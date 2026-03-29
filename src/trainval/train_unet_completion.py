import argparse
import logging
import os
import os.path as osp
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from configs import Config, update_configs
from src.datasets import Scan3RSceneBatchDataset
from src.engine import EpochBasedTrainer
from src.models.unet3d_completion import UNetCompletionModel
from utils import common, scan3r
from utils.visualisation import save_vox_as_ply, side_by_side, slice_mosaic
from torch.utils.data import DataLoader

logging.getLogger("PIL").setLevel(logging.WARNING)


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg: Config, parser: argparse.ArgumentParser = None) -> None:
        super().__init__(cfg, parser)
        self.cfg = cfg
        self.root_dir = cfg.data.root_dir
        self.G = 64
        self.recon_weight = getattr(cfg.train.loss, "recon_weight", 1.0)
        self.lambda_dice = getattr(cfg.train.loss, "dice_weight", 1.0)

        start = time.time()
        train_loader = self._build_loader(cfg, split="train", batch_size=cfg.train.batch_size, shuffle=True)
        val_loader = self._build_loader(cfg, split="val", batch_size=cfg.val.batch_size, shuffle=False)
        self.logger.info(f"Data loader built in {time.time() - start:.2f}s")
        self.register_loader(train_loader, val_loader)

        model = self.create_model()
        self.register_model(model)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.optim.lr,
            eps=1e-8,
            weight_decay=cfg.train.optim.weight_decay,
        )
        self.register_optimizer(optimizer)
        self.logger.info("Initialization complete.")

    def create_model(self) -> UNetCompletionModel:
        model = UNetCompletionModel().to(self.device)
        # snapshot = getattr(self.cfg.train, "unet_completion_snapshot", None)
        snapshot = '/cluster/scratch/wangyih/overfitting_dataset/pretrained/training_unet_completion/2025-11-5_300_scenes_unet_fix_compressor/snapshots/epoch-25.pth.tar'
        if snapshot and osp.exists(snapshot):
            state = torch.load(snapshot, map_location=self.device)
            model_state = state.get("model", state)
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            self.logger.info(
                f"Loaded UNet snapshot {snapshot} (missing={missing}, unexpected={unexpected})"
            )
        return model

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

    # ----------------- dataset helpers (copied & simplified) -----------------
    def _load_aligned_pack(self, root_dir: str, scene_id: str, frame_id: str):
        base = osp.join(root_dir, "files", "gs_annotations", scene_id, "scene_level_structure_no_dilation")
        p = osp.join(base, f"student_pack_aligned_{frame_id}.npz")
        if not osp.exists(p):
            cands = [f for f in os.listdir(base) if f.startswith("student_pack_aligned_")]
            if not cands:
                raise FileNotFoundError(f"No aligned pack in {base}")
            p = osp.join(base, cands[0])
        return np.load(p, allow_pickle=False)

    def _rasterize_idx(self, idx_np: np.ndarray, G: int) -> torch.Tensor:
        occ = np.zeros((G, G, G), dtype=np.uint8)
        if idx_np.size:
            occ[idx_np[:, 0], idx_np[:, 1], idx_np[:, 2]] = 1
        return torch.from_numpy(occ).unsqueeze(0).float()

    def _build_inputs(self, pack, expect_G: int):
        G = int(pack["G"])
        if G != expect_G:
            raise ValueError(f"G mismatch: expected {expect_G}, got {G}")
        occ_gt = self._rasterize_idx(pack["vox_idx_gt_occ"], G)
        seed_idx = pack.get("seed_idx", np.zeros((0, 3), np.int32))
        feats = pack.get("feats", np.zeros((0, 1024), np.float32))
        mean_gt = torch.from_numpy(pack["mean_gt"].astype(np.float32))
        scale_gt = torch.tensor(float(pack["scale_gt"]), dtype=torch.float32).view(1)
        mean_seed0 = torch.from_numpy(pack["seed_box_init_mean"].astype(np.float32))
        scale_seed0 = torch.tensor(float(pack["seed_box_init_scale"]), dtype=torch.float32).view(1)
        return occ_gt, seed_idx, feats, mean_gt, scale_gt, mean_seed0, scale_seed0

    def _make_batch(self, data_dict: Dict[str, Any]):
        sg = data_dict["scene_graphs"]
        scene_ids = [sid[0] for sid in sg["scene_ids"]]
        frame_ids_seq = sg.get("frame_ids")
        image_frames = sg.get("image_frames", {})
        G = self.G

        occ_gt_list = []
        seed_idx_list, feats_list = [], []
        mean_gt_list, scale_gt_list = [], []
        mean_seed0_list, scale_seed0_list = [], []

        for idx, sid in enumerate(scene_ids):
            if frame_ids_seq is not None and idx < len(frame_ids_seq):
                fid = str(frame_ids_seq[idx])
            else:
                frames = image_frames.get(sid, [])
                if frames:
                    fid = str(frames[0])
                else:
                    frame_ids = scan3r.load_frame_idxs(osp.join(self.cfg.data.root_dir, "scenes"), sid)
                    fid = frame_ids[0] if frame_ids else "000000"
            try:
                pack = self._load_aligned_pack('/cluster/scratch/wangyih/3RScan', sid, fid)
                occ_gt, seed_idx, feats, mean_gt, scale_gt, mean_seed0, scale_seed0 = self._build_inputs(pack, G)
            except (zipfile.BadZipFile, ValueError, OSError) as exc:
                self.logger.warning(f"Skipping corrupted pack for {sid}/{fid}: {exc}")
                continue
            seed_idx_list.append(torch.from_numpy(seed_idx).int())
            feats_list.append(torch.from_numpy(feats).float())
            occ_gt_list.append(occ_gt.unsqueeze(0))
            mean_gt_list.append(mean_gt.unsqueeze(0))
            scale_gt_list.append(scale_gt.unsqueeze(0))
            mean_seed0_list.append(mean_seed0.unsqueeze(0))
            scale_seed0_list.append(scale_seed0.unsqueeze(0))

        if not occ_gt_list:
            raise RuntimeError("Failed to build batch: all packs corrupted or missing.")
        occ_gt = torch.cat(occ_gt_list, dim=0).to(self.device)
        mean_gt = torch.cat(mean_gt_list, dim=0).to(self.device)
        scale_gt = torch.cat(scale_gt_list, dim=0).to(self.device)
        mean_seed0 = torch.cat(mean_seed0_list, dim=0).to(self.device)
        scale_seed0 = torch.cat(scale_seed0_list, dim=0).to(self.device)
        return occ_gt, seed_idx_list, feats_list, mean_gt, scale_gt, mean_seed0, scale_seed0

    # ----------------- geometry helpers -----------------
    def scatter_voxel_mean(self, idx_t: torch.Tensor, feat_t: torch.Tensor, G: int):
        if idx_t.numel() == 0:
            C = feat_t.shape[-1] if feat_t.ndim == 2 else 64
            grid_feats = torch.zeros(1, C, G, G, G, device=self.device)
            seed_occ = torch.zeros(1, 1, G, G, G, device=self.device)
            return grid_feats, seed_occ
        M, C = feat_t.shape
        lin = (idx_t[:, 0] * G * G + idx_t[:, 1] * G + idx_t[:, 2]).long()
        Csum = torch.zeros(C, G * G * G, device=self.device)
        cnt = torch.zeros(G * G * G, device=self.device)
        Csum.index_add_(1, lin, feat_t.T)
        cnt.index_add_(0, lin, torch.ones(M, device=self.device))
        mask = cnt > 0
        Csum[:, mask] = Csum[:, mask] / cnt[mask]
        grid_feats = Csum.view(C, G, G, G).unsqueeze(0)
        seed_occ = torch.zeros(1, 1, G, G, G, device=self.device)
        seed_occ.view(1, 1, -1)[0, 0, torch.unique(lin)] = 1.0
        return grid_feats, seed_occ

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
        c_seed = self._idx_to_centers(seed_idx, G)
        s_src = torch.clamp(scale_src[b].view(1, 1), min=eps)
        m_src = mean_src[b].view(1, 3)
        world_can = c_seed * (2.0 * s_src) + m_src
        s_dst = torch.clamp(scale_dst[b].view(1, 1), min=eps)
        m_dst = mean_dst[b].view(1, 3)
        c_dst = (world_can - m_dst) / (2.0 * s_dst)
        return self._centers_to_idx(c_dst, G)

    def remap_occ(self, occ_src, mean_src, scale_src, mean_dst, scale_dst, G):
        B = occ_src.shape[0]
        occ_dst = torch.zeros_like(occ_src)
        for b in range(B):
            idx_src = (occ_src[b, 0] > 0.5).nonzero(as_tuple=False)
            if idx_src.numel() == 0:
                continue
            idx_dst = self._remap_seed_idx_with_bbox(idx_src, mean_src, scale_src, mean_dst, scale_dst, b, G)
            occ_dst[b, 0, idx_dst[:, 0], idx_dst[:, 1], idx_dst[:, 2]] = 1.0
        return occ_dst

    # ----------------- train / val -----------------
    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        occ_gt, seed_idx_list, feats_list, mean_gt, scale_gt, mean_seed0, scale_seed0 = self._make_batch(data_dict)
        B = occ_gt.shape[0]
        mean_dst, scale_dst = mean_gt, scale_gt

        x_list, occ_vis_list = [], []
        for b in range(B):
            idx_t = seed_idx_list[b].to(self.device).long()
            feats_t = feats_list[b].to(self.device).float()
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                feats_comp = self.model.feature_compressor(feats_t).float()
            idx_dst = self._remap_seed_idx_with_bbox(idx_t, mean_seed0, scale_seed0, mean_dst, scale_dst, b, self.G)
            grid_dino, seed_occ = self.scatter_voxel_mean(idx_dst.int(), feats_comp, self.G)
            x_list.append(torch.cat([seed_occ, grid_dino], dim=1))
            occ_vis_list.append(seed_occ)

        x_in = torch.cat(x_list, dim=0)
        occ_vis = torch.cat(occ_vis_list, dim=0)
        occ_gt = occ_gt.to(self.device)

        logits = self.model(x_in)
        mask_complete = (1 - occ_vis) + 0.3 * (F.max_pool3d(occ_vis, 3, 1, 1) - occ_vis).clamp_min(0)
        mask_complete = mask_complete.clamp_max(1.0)
        bce = F.binary_cross_entropy_with_logits(logits, occ_gt, reduction="none")
        vae_rec = (bce * mask_complete).sum() / (mask_complete.sum() + 1e-6)

        probs_dice = torch.sigmoid(logits)
        gt = (occ_gt > 0.5).float()
        P = probs_dice * mask_complete
        Gt = gt * mask_complete
        inter = (P * Gt).sum(dim=(1, 2, 3, 4))
        pred = P.sum(dim=(1, 2, 3, 4))
        target = Gt.sum(dim=(1, 2, 3, 4))
        dice_loss = (1.0 - (2 * inter + 1e-6) / (pred + target + 1e-6)).mean()

        loss = self.recon_weight * vae_rec + self.lambda_dice * dice_loss

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            iou_metrics = self._compute_iou(probs, occ_gt)

        loss_dict = {
            "loss": loss,
            "bce": vae_rec,
            "dice": dice_loss,
            "IoU @ 0.3": iou_metrics[0.3].item(),
            "IoU @ 0.5": iou_metrics[0.5].item(),
            "IoU @ 0.7": iou_metrics[0.7].item(),
        }

        output_dict = {
            "logits": logits[: min(B, 4)].detach().cpu(),
            "occ_gt": occ_gt[: min(B, 4)].detach().cpu(),
            "occ_vis": occ_vis[: min(B, 4)].detach().cpu(),
            "sample_logits": None,
        }
        return output_dict, loss_dict

    def _compute_iou(self, probs: torch.Tensor, gt: torch.Tensor, thresholds=(0.3, 0.5, 0.7)):
        metrics = {}
        gt_bin = (gt > 0.5).float()
        for thr in thresholds:
            pr = (probs >= thr).float()
            inter = (pr * gt_bin).sum(dim=(1, 2, 3, 4))
            union = pr.sum(dim=(1, 2, 3, 4)) + gt_bin.sum(dim=(1, 2, 3, 4)) - inter
            metrics[thr] = (inter / (union + 1e-6)).mean()
        return metrics

    def val_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with torch.no_grad():
            return self.train_step(epoch, iteration, data_dict)

    # ----------------- visualization -----------------
    def visualize(self, output_dict: Dict[str, Any], epoch: int, mode: str = "train") -> None:
        outdir = f"{self.cfg.output_dir}/events"
        os.makedirs(outdir, exist_ok=True)
        logits = output_dict["logits"]
        occ_gt = output_dict["occ_gt"]
        occ_vis = output_dict["occ_vis"]
        B = logits.shape[0]
        thr = 0.5
        for i in range(B):
            pr_idx = (logits[i, 0].sigmoid() > thr).nonzero(as_tuple=False)
            gt_idx = (occ_gt[i, 0] > thr).nonzero(as_tuple=False)
            vis_idx = (occ_vis[i, 0] > thr).nonzero(as_tuple=False)
            save_vox_as_ply(gt_idx, self.G, f"{outdir}/{mode}_gt_{i}.ply")
            save_vox_as_ply(pr_idx, self.G, f"{outdir}/{mode}_pred_{i}.ply")
            save_vox_as_ply(vis_idx, self.G, f"{outdir}/{mode}_input_{i}.ply")
            sbs = side_by_side(occ_gt[i, 0], logits[i, 0], max_slices=6)
            self.writer.add_image(
                f"{mode}/slices_{i}_gt_pred",
                sbs.unsqueeze(1),  # (2,1,H,W)
                global_step=epoch,
                dataformats="NCHW",
            )
            save_image(sbs, f"{outdir}/{mode}_slices_{i}.png", normalize=True)


def parse_args(parser: argparse.ArgumentParser = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--load_encoder", default=None, type=str)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    args, unknown_args = parser.parse_known_args()
    return parser, args, unknown_args


def main() -> None:
    common.init_log(level=logging.INFO)
    parser, args, unknown = parse_args()
    cfg = update_configs(args.config, unknown)
    trainer = Trainer(cfg, parser)
    trainer.run()


if __name__ == "__main__":
    main()
