import argparse
import logging
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from configs import Config, update_configs
from src.datasets import Scan3RSceneBatchDataset
from torch.utils.data import DataLoader
from src.engine import EpochBasedTrainer
from src.models.masked_inpainting_unet import MaskedInpaintingUNet
from utils import common
from utils.visualisation import save_vox_as_ply, side_by_side


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg: Config, parser: argparse.ArgumentParser = None) -> None:
        super().__init__(cfg, parser)
        self.cfg = cfg
        self.root_dir = cfg.data.root_dir
        self.G = getattr(cfg.train.loss, "grid_size", 64)
        self.use_dino_feats: bool = getattr(cfg.train.loss, "use_aligned_dino_feats", True)
        self.dino_feat_dim: int = getattr(cfg.train.loss, "dino_feat_dim", 1024)
        self.dino_grid_channels: int = getattr(cfg.train.loss, "dino_grid_channels", 16)
        train_loader = self._build_loader(cfg, split="train", batch_size=cfg.train.batch_size, shuffle=True)
        val_loader = self._build_loader(cfg, split="val", batch_size=cfg.val.batch_size, shuffle=False)
        self.register_loader(train_loader, val_loader)
        base_ch = getattr(getattr(cfg.train, "model", None), "base_channels", 32)
        in_channels = 2 + (self.dino_grid_channels if self.use_dino_feats else 0)
        model = MaskedInpaintingUNet(
            in_channels=in_channels,
            base_channels=base_ch,
            dino_feat_dim=self.dino_feat_dim if self.use_dino_feats else 0,
            dino_grid_channels=self.dino_grid_channels if self.use_dino_feats else 0,
        )
        model = model.to(self.device)
        self.register_model(model)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.train.optim.lr,
            weight_decay=cfg.train.optim.weight_decay,
        )
        self.register_optimizer(optimizer)
        self.logger.info("Masked inpainting trainer initialised")

    def _build_loader(self, cfg: Config, split: str, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = Scan3RSceneBatchDataset(cfg, split=split)
        num_workers = cfg.train.num_workers if shuffle else cfg.val.num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._minimal_collate,
            pin_memory=True,
            drop_last=False,
        )

    @staticmethod
    def _minimal_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        scene_ids = [sample["scan_id"] for sample in batch]
        frame_ids = [sample["frame_idx"] for sample in batch]
        image_frames: Dict[str, List[str]] = {}
        for sid, fid in zip(scene_ids, frame_ids):
            image_frames.setdefault(sid, []).append(fid)
        scene_graphs = {
            "scene_ids": np.array([[sid] for sid in scene_ids]),
            "frame_ids": frame_ids,
            "image_frames": image_frames,
        }
        return {"scene_graphs": scene_graphs}

    def _load_aligned_pack(self, scene_id: str, frame_id: str) -> Dict[str, Any]:
        root_dir_scratch = "/cluster/scratch/wangyih/3RScan"
        base = osp.join(root_dir_scratch, "files", "gs_annotations", scene_id, "scene_level_structure")
        cand = f"student_pack_aligned_{frame_id}.npz"
        path = osp.join(base, cand)
        if not osp.exists(path):
            files = [f for f in os.listdir(base) if f.startswith("student_pack_aligned_")]
            if not files:
                raise FileNotFoundError(f"No aligned packs for {scene_id}")
            path = osp.join(base, files[0])
        return np.load(path)

    def _rasterize_idx(self, idx_np: np.ndarray) -> torch.Tensor:
        occ = np.zeros((self.G, self.G, self.G), dtype=np.uint8)
        if idx_np.size:
            occ[idx_np[:, 0], idx_np[:, 1], idx_np[:, 2]] = 1
        return torch.from_numpy(occ).float()

    def _indices_to_occ(self, idx_t: torch.Tensor, G: int) -> torch.Tensor:
        occ = torch.zeros(G, G, G, dtype=torch.float32)
        if idx_t.numel() > 0:
            occ[idx_t[:, 0], idx_t[:, 1], idx_t[:, 2]] = 1.0
        return occ

    def _idx_to_centers(self, idx: torch.Tensor, G: int) -> torch.Tensor:
        return (idx.float() + 0.5) / G - 0.5

    def _centers_to_idx(self, centers: torch.Tensor, G: int) -> torch.Tensor:
        idx = torch.floor((centers + 0.5) * G).long()
        return torch.clamp(idx, 0, G - 1)

    def _remap_seed_idx_with_bbox(
        self,
        seed_idx: torch.Tensor,
        mean_src: torch.Tensor,
        scale_src: torch.Tensor,
        mean_dst: torch.Tensor,
        scale_dst: torch.Tensor,
        b: int,
        G: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if seed_idx.numel() == 0:
            return seed_idx
        c_seed = self._idx_to_centers(seed_idx, G)
        s_src = torch.clamp(scale_src[b].view(1, 1), min=eps)
        m_src = mean_src[b].view(1, 3)
        world = c_seed * (2.0 * s_src) + m_src
        s_dst = torch.clamp(scale_dst[b].view(1, 1), min=eps)
        m_dst = mean_dst[b].view(1, 3)
        c_dst = (world - m_dst) / (2.0 * s_dst)
        return self._centers_to_idx(c_dst, G)

    def _scatter_voxel_mean(self, idx_t: torch.Tensor, feat_t: torch.Tensor, G: int) -> torch.Tensor:
        if idx_t.numel() == 0 or feat_t.numel() == 0:
            C = feat_t.shape[-1] if feat_t.ndim == 2 else self.dino_grid_channels
            return torch.zeros(1, C, G, G, G, device=feat_t.device, dtype=feat_t.dtype)
        M, C = feat_t.shape
        lin = (idx_t[:, 0] * G * G + idx_t[:, 1] * G + idx_t[:, 2]).long()
        feat_sum = torch.zeros(C, G * G * G, device=feat_t.device, dtype=feat_t.dtype)
        counts = torch.zeros(G * G * G, device=feat_t.device, dtype=feat_t.dtype)
        feat_sum.index_add_(1, lin, feat_t.T)
        counts.index_add_(0, lin, torch.ones(M, device=feat_t.device, dtype=feat_t.dtype))
        mask = counts > 0
        feat_sum[:, mask] = feat_sum[:, mask] / counts[mask]
        grid = feat_sum.view(C, G, G, G).unsqueeze(0)
        return grid

    def _make_batch(self, data_dict: Dict[str, Any]):
        sg = data_dict["scene_graphs"]
        scene_ids = [sid[0] for sid in sg["scene_ids"]]
        frame_ids = sg.get("frame_ids", None)
        image_frames = sg.get("image_frames", {})
        occ_gt_list = []
        seed_list = []
        dino_grid_list: List[torch.Tensor] = []
        for idx, sid in enumerate(scene_ids):
            if frame_ids is not None and idx < len(frame_ids):
                fid = str(frame_ids[idx])
            else:
                frames = image_frames.get(sid, [])
                fid = str(frames[0]) if frames else "000000"
            pack = self._load_aligned_pack(sid, fid)
            occ_gt = self._rasterize_idx(pack["vox_idx_gt_occ"])
            seed_idx = pack.get("seed_idx", np.zeros((0, 3), np.int32))
            mean_gt = torch.from_numpy(pack["mean_gt"].astype(np.float32))
            scale_gt = torch.tensor(float(pack["scale_gt"]), dtype=torch.float32).view(1)
            mean_seed0 = torch.from_numpy(pack["seed_box_init_mean"].astype(np.float32))
            scale_seed0 = torch.tensor(float(pack["seed_box_init_scale"]), dtype=torch.float32).view(1)
            if seed_idx.size > 0:
                seed_idx_t = torch.from_numpy(seed_idx).int()
                idx_remap = self._remap_seed_idx_with_bbox(
                    seed_idx=seed_idx_t,
                    mean_src=mean_seed0.unsqueeze(0),
                    scale_src=scale_seed0.view(1, 1),
                    mean_dst=mean_gt.unsqueeze(0),
                    scale_dst=scale_gt.view(1, 1),
                    b=0,
                    G=self.G,
                )
            else:
                idx_remap = torch.zeros((0, 3), dtype=torch.int32)
            seeds = self._indices_to_occ(idx_remap, self.G)
            occ_gt_list.append(occ_gt.unsqueeze(0).unsqueeze(0))
            seed_list.append(seeds.unsqueeze(0).unsqueeze(0))
            if self.use_dino_feats:
                feats_np = pack.get("feats", np.zeros((0, self.dino_feat_dim), np.float32))
                if feats_np.size > 0 and idx_remap.numel() > 0:
                    feats_t = torch.from_numpy(feats_np).to(self.device, dtype=torch.float32)
                    projected = self.model.project_dino_feats(feats_t)
                    idx_dev = idx_remap.to(self.device, dtype=torch.long)
                    grid_feats = self._scatter_voxel_mean(idx_dev, projected, self.G)
                else:
                    grid_feats = torch.zeros(
                        1,
                        self.dino_grid_channels,
                        self.G,
                        self.G,
                        self.G,
                        device=self.device,
                        dtype=torch.float32,
                    )
                dino_grid_list.append(grid_feats)
        occ_gt = torch.cat(occ_gt_list, dim=0).to(self.device)
        seed_occ = torch.cat(seed_list, dim=0).to(self.device)
        mask = (seed_occ > 0).float()
        dino_grid = None
        if self.use_dino_feats:
            dino_grid = torch.cat(dino_grid_list, dim=0)
        return occ_gt, seed_occ, mask, dino_grid

    def _compute_iou(self, logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pred = (probs > thr).float()
        inter = (pred * target).sum(dim=(1, 2, 3, 4))
        union = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) - inter
        return (inter / (union + 1e-6)).mean()

    def train_step(self, epoch: int, iteration: int, data_dict: Dict[str, Any]):
        occ_gt, seed_occ, mask, dino_grid = self._make_batch(data_dict)
        inputs = [seed_occ, mask]
        if self.use_dino_feats and dino_grid is not None:
            inputs.append(dino_grid)
        x_in = torch.cat(inputs, dim=1)
        logits = self.model(x_in, mask)
        miss_mask = (1.0 - mask)
        bce = F.binary_cross_entropy_with_logits(logits, occ_gt, reduction="none")
        loss = (bce * miss_mask).sum() / miss_mask.sum().clamp_min(1.0)
        iou = self._compute_iou(logits.detach(), occ_gt)
        loss_dict = {"loss": loss, "masked_bce": loss, "IoU @ 0.5": iou.detach()}
        output = {"logits": logits.detach().cpu(), "occ_gt": occ_gt.detach().cpu(), "seed_occ": seed_occ.detach().cpu()}
        return output, loss_dict

    def val_step(self, epoch: int, iteration: int, data_dict: Dict[str, Any]):
        with torch.no_grad():
            return self.train_step(epoch, iteration, data_dict)

    def visualize(self, output_dict: Dict[str, Any], epoch: int, mode: str = "train") -> None:
        outdir = osp.join(self.cfg.output_dir, "events")
        os.makedirs(outdir, exist_ok=True)
        logits = output_dict["logits"]
        occ_gt = output_dict["occ_gt"]
        seed_occ = output_dict["seed_occ"]
        B = logits.shape[0]
        thr = 0.5
        for i in range(min(B, 4)):
            pred_idx = (torch.sigmoid(logits[i, 0]) > thr).nonzero(as_tuple=False)
            gt_idx = (occ_gt[i, 0] > thr).nonzero(as_tuple=False)
            seed_idx = (seed_occ[i, 0] > thr).nonzero(as_tuple=False)
            save_vox_as_ply(gt_idx, self.G, f"{outdir}/{mode}_gt_{i}.ply")
            save_vox_as_ply(pred_idx, self.G, f"{outdir}/{mode}_pred_{i}.ply")
            save_vox_as_ply(seed_idx, self.G, f"{outdir}/{mode}_seed_{i}.ply")
            sbs = side_by_side(occ_gt[i, 0], logits[i, 0], max_slices=6)
            save_image(sbs, f"{outdir}/{mode}_slices_{i}.png", normalize=True)
            self.writer.add_image(
                f"{mode}/slices_{i}_gt_pred",
                sbs.unsqueeze(1),
                global_step=epoch,
                dataformats="NCHW",
            )


def parse_args(parser: argparse.ArgumentParser = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    args, unknown = parser.parse_known_args()
    return parser, args, unknown


def main() -> None:
    common.init_log(level=logging.INFO)
    parser, args, unknown = parse_args()
    cfg = update_configs(args.config, unknown)
    trainer = Trainer(cfg, parser)
    trainer.run()


if __name__ == "__main__":
    main()
