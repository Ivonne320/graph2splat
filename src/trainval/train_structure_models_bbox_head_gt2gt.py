import argparse
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import zipfile

from configs import Config, update_configs
from src.datasets import Scan3RSceneBatchDataset
from src.engine import EpochBasedTrainer
from src.models.structure_model_with_bbox_head import StructureModel
from utils import common, scan3r
from utils.visualisation import save_vox_as_ply, side_by_side


class Trainer(EpochBasedTrainer):
    """
    Pretrain the TRELLIS structure VAE on GT occupancy -> GT occupancy (no seeds).
    """

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
        self.logger.info(f"Data loader ready in {time.time() - start:.2f}s")
        self.register_loader(train_loader, val_loader)

        model = StructureModel(cfg=self.cfg.autoencoder, device=self.device)
        self.register_model(model)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.optim.lr,
            eps=1e-3,
            weight_decay=cfg.train.optim.weight_decay,
        )
        self.register_optimizer(optimizer)
        self.logger.info("Initialization complete.")

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
        frame_ids = [sample.get("frame_idx") for sample in batch]
        image_frames: Dict[str, List[str]] = {}
        for sid, fid in zip(scene_ids, frame_ids):
            if fid is not None:
                image_frames.setdefault(sid, []).append(fid)
        scene_graphs = {
            "scene_ids": np.array([[sid] for sid in scene_ids]),
            "frame_ids": frame_ids,
            "image_frames": {sid: list(dict.fromkeys(fids)) for sid, fids in image_frames.items()},
        }
        return {"scene_graphs": scene_graphs}

    # ---------------- Data helpers ----------------
    def _load_aligned_pack(self, root_dir: str, scene_id: str, frame_id: str):
        base = os.path.join(root_dir, "files", "gs_annotations", scene_id, "scene_level_structure")
        p = os.path.join(base, f"student_pack_aligned_{frame_id}.npz")
        if not os.path.exists(p):
            cands = [
                f for f in os.listdir(base) if f.startswith("student_pack_aligned_") and f.endswith(".npz")
            ]
            if not cands:
                raise FileNotFoundError(f"No aligned pack found in {base}")
            p = os.path.join(base, cands[0])
        return np.load(p)

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
        mean_gt = torch.from_numpy(pack["mean_gt"].astype(np.float32))
        scale_gt = torch.tensor(float(pack["scale_gt"]), dtype=torch.float32).view(1)
        return occ_gt, mean_gt, scale_gt

    def _make_batch(self, data_dict: Dict[str, Any]):
        sg = data_dict["scene_graphs"]
        scene_ids = [sid[0] for sid in sg["scene_ids"]]
        frame_ids_seq = sg.get("frame_ids")
        image_frames = sg.get("image_frames", {})

        occ_gt_list = []
        mean_gt_list, scale_gt_list = [], []

        for idx, sid in enumerate(scene_ids):
            if frame_ids_seq is not None and idx < len(frame_ids_seq) and frame_ids_seq[idx] is not None:
                fid = str(frame_ids_seq[idx])
            else:
                frames = image_frames.get(sid, [])
                if frames:
                    fid = str(frames[0])
                else:
                    frame_ids = scan3r.load_frame_idxs(os.path.join(self.cfg.data.root_dir, "scenes"), sid)
                    fid = frame_ids[0] if frame_ids else "000000"
            try:
                pack = self._load_aligned_pack('/cluster/scratch/wangyih/3RScan', sid, fid)
                occ_gt, mean_gt, scale_gt = self._build_inputs(pack, self.G)
            except (zipfile.BadZipFile, ValueError, OSError) as exc:
                self.logger.warning(f"Skipping corrupted pack {sid}/{fid}: {exc}")
                continue
            occ_gt_list.append(occ_gt.unsqueeze(0))
            mean_gt_list.append(mean_gt.unsqueeze(0))
            scale_gt_list.append(scale_gt.unsqueeze(0))

        if not occ_gt_list:
            raise RuntimeError("All packs in batch failed to load; cannot proceed.")

        occ_gt = torch.cat(occ_gt_list, dim=0).to(self.device)
        mean_gt = torch.cat(mean_gt_list, dim=0).to(self.device)
        scale_gt = torch.cat(scale_gt_list, dim=0).to(self.device)
        return occ_gt, mean_gt, scale_gt

    # --------------- Training --------------------
    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        occ_gt, _, _ = self._make_batch(data_dict)
        x_in = occ_gt  # GT occupancy as input

        with torch.amp.autocast(self.device.type, enabled=False):
            z, mu, _logvar, _feat = self.model.encoder(
                x_in, sample_posterior=False, return_raw=True, return_feat=True
            )
            logits = self.model.decoder(mu)

        logits32 = logits.float()
        occ_gt32 = occ_gt.float()
        bce = F.binary_cross_entropy_with_logits(logits32, occ_gt32, reduction="mean")

        probs = torch.sigmoid(logits32)
        gt = (occ_gt32 > 0.5).float()
        inter = (probs * gt).sum(dim=(1, 2, 3, 4))
        pred = probs.sum(dim=(1, 2, 3, 4))
        target = gt.sum(dim=(1, 2, 3, 4))
        dice_loss = (1.0 - (2 * inter + 1e-6) / (pred + target + 1e-6)).mean()

        loss = self.recon_weight * bce + self.lambda_dice * dice_loss

        with torch.no_grad():
            def iou_at(thr):
                pr = (probs >= thr).float()
                inter = (pr * gt).sum(dim=(1, 2, 3, 4))
                union = pr.sum(dim=(1, 2, 3, 4)) + gt.sum(dim=(1, 2, 3, 4)) - inter
                return (inter / (union + 1e-6)).mean()

            iou03 = iou_at(0.3)
            iou05 = iou_at(0.5)
            iou07 = iou_at(0.7)

        loss_dict = {
            "loss": loss,
            "bce": bce,
            "dice": dice_loss,
            "IoU @ 0.3": iou03.item(),
            "IoU @ 0.5": iou05.item(),
            "IoU @ 0.7": iou07.item(),
        }

        output_dict = {
            "logits": logits[: min(occ_gt.shape[0], 4)].detach().cpu(),
            "occ_gt": occ_gt[: min(occ_gt.shape[0], 4)].detach().cpu(),
        }
        return output_dict, loss_dict

    def val_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with torch.no_grad():
            return self.train_step(epoch, iteration, data_dict)

    def visualize(self, output_dict: Dict[str, Any], epoch: int, mode: str = "train"):
        outdir = f"{self.cfg.output_dir}/events"
        os.makedirs(outdir, exist_ok=True)
        logits = output_dict["logits"]
        occ_gt = output_dict["occ_gt"]
        thr = 0.5
        for i in range(logits.shape[0]):
            pr_idx = (logits[i, 0].sigmoid() > thr).nonzero(as_tuple=False)
            gt_idx = (occ_gt[i, 0] > thr).nonzero(as_tuple=False)
            save_vox_as_ply(pr_idx, self.G, f"{outdir}/{mode}_pred_{i}.ply")
            save_vox_as_ply(gt_idx, self.G, f"{outdir}/{mode}_gt_{i}.ply")
            sbs = side_by_side(occ_gt[i, 0], logits[i, 0], max_slices=6)
            save_image(sbs, f"{outdir}/{mode}_slices_{i}.png", normalize=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--snapshot", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    args, unknown = parser.parse_known_args()
    return parser, args, unknown


def main():
    common.init_log(level=logging.INFO)
    parser, args, unknown = parse_args()
    cfg = update_configs(args.config, unknown)
    trainer = Trainer(cfg, parser)
    trainer.run()


if __name__ == "__main__":
    main()
