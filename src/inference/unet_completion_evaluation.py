import argparse
import logging
import os
import os.path as osp
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import zipfile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import Config, update_configs
from src.datasets import Scan3RSceneBatchDataset
from src.models.unet3d_completion import UNetCompletionModel
from utils import common, scan3r
from utils.visualisation import save_vox_as_ply, side_by_side
from torchvision.utils import save_image

logging.getLogger("PIL").setLevel(logging.WARNING)
_LOGGER = logging.getLogger(__name__)


class UNetCompletionEvaluator:
    def __init__(
        self,
        cfg: Config,
        split: str = "val",
        checkpoint: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        scene_ids: Optional[Iterable[str]] = None,
        save_vis: bool = False,
        max_batches: Optional[int] = None,
        remap_seed_feats: bool = True,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = 64
        self.save_vis = save_vis
        self.remap_seed_feats = remap_seed_feats
        self.target_scene_ids = set(scene_ids) if scene_ids else None
        self.max_batches = max_batches

        self.model = UNetCompletionModel().to(self.device).eval()
        self._load_checkpoint(checkpoint)

        split_cfg = getattr(cfg, "val", None)
        default_batch_size = getattr(split_cfg, "batch_size", 1) if split_cfg else 1
        default_num_workers = getattr(split_cfg, "num_workers", 0) if split_cfg else 0

        self.dataset = Scan3RSceneBatchDataset(cfg, split=split)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size or default_batch_size,
            shuffle=False,
            num_workers=num_workers if num_workers is not None else default_num_workers,
            collate_fn=lambda batch: batch,
            pin_memory=True,
            drop_last=False,
        )
        self.max_batches = max_batches

    def _load_checkpoint(self, checkpoint: Optional[str]) -> None:
        if checkpoint is None:
            inference_cfg = getattr(self.cfg, "inference", None)
            checkpoint = getattr(inference_cfg, "unet_completion_path", None)
        if checkpoint is None or not osp.exists(checkpoint):
            raise ValueError(
                "UNet checkpoint not provided. Use --checkpoint or cfg.inference.unet_completion_path."
            )
        state = torch.load(checkpoint, map_location=self.device)
        model_state = state.get("model", state)
        self.model.load_state_dict(model_state, strict=False)
        _LOGGER.info("Loaded checkpoint from %s", checkpoint)

    # ------- Pack loading utilities -------
    def _load_aligned_pack(self, scene_id: str, frame_id: str):
        base = osp.join(
            "/cluster/scratch/wangyih/3RScan",
            "files",
            "gs_annotations",
            scene_id,
            "scene_level_structure",
        )
        p = osp.join(base, f"student_pack_aligned_{frame_id}.npz")
        if not osp.exists(p):
            cand = [
                f for f in os.listdir(base) if f.startswith("student_pack_aligned_")
            ]
            if not cand:
                raise FileNotFoundError(f"No aligned pack in {base}")
            p = osp.join(base, cand[0])
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
        seed_idx = pack.get("seed_idx", np.zeros((0, 3), np.int32))
        feats = pack.get("feats", np.zeros((0, 1024), np.float32))
        mean_gt = torch.from_numpy(pack["mean_gt"].astype(np.float32))
        scale_gt = torch.tensor(float(pack["scale_gt"]), dtype=torch.float32).view(1)
        mean_seed0 = torch.from_numpy(pack["seed_box_init_mean"].astype(np.float32))
        scale_seed0 = torch.tensor(float(pack["seed_box_init_scale"]), dtype=torch.float32).view(1)
        return occ_gt, seed_idx, feats, mean_gt, scale_gt, mean_seed0, scale_seed0

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
        G: int,
        b: int,
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

    # ------- Evaluation -------
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total = 0
        iou_sums = {0.3: 0.0, 0.5: 0.0, 0.7: 0.0}
        total_tp = total_fp = total_fn = 0.0

        total_bar = (
            self.max_batches
            if self.max_batches is not None
            else len(self.dataloader)
        )
        for batch_idx, raw_batch in enumerate(
            tqdm(self.dataloader, desc="Evaluating", total=total_bar)
        ):
            batch_res = self._process_batch(raw_batch)
            if batch_res is None:
                continue
            iou_batch, bsz, tp, fp, fn = batch_res
            total += bsz
            total_tp += tp
            total_fp += fp
            total_fn += fn
            for k, v in iou_batch.items():
                iou_sums[k] += v * bsz
            if self.max_batches is not None and (batch_idx + 1) >= self.max_batches:
                break

        if total == 0:
            raise RuntimeError("No samples evaluated.")
        metrics = {f"IoU@{thr}": val / total for thr, val in iou_sums.items()}
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = (
            2 * precision * recall / (precision + recall + 1e-8)
            if (precision + recall) > 0
            else 0.0
        )
        metrics.update(
            {
                "voxel_precision": precision,
                "voxel_recall": recall,
                "voxel_f1": f1,
                "voxel_tp": total_tp,
                "voxel_fp": total_fp,
                "voxel_fn": total_fn,
            }
        )
        _LOGGER.info("Evaluation complete: %s", metrics)
        return metrics

    def _process_batch(
        self, raw_batch: List[Dict[str, Any]]
    ) -> Optional[Tuple[Dict[float, float], int, float, float, float]]:
        scene_ids = [sample["scan_id"] for sample in raw_batch]
        if self.target_scene_ids:
            keep = [sid for sid in scene_ids if sid in self.target_scene_ids]
            if not keep:
                return None

        frame_ids = [sample.get("frame_idx") for sample in raw_batch]
        batch = {
            "scene_graphs": {
                "scene_ids": np.array([[sid] for sid in scene_ids]),
                "frame_ids": frame_ids,
            }
        }
        image_frames = {}
        for sid, fid in zip(scene_ids, frame_ids):
            if fid is None:
                continue
            image_frames.setdefault(sid, []).append(fid)
        batch["scene_graphs"]["image_frames"] = {
            sid: list(dict.fromkeys(fids)) for sid, fids in image_frames.items()
        }

        occ_gt, seed_idx_list, feats_list, mean_gt, scale_gt, mean_seed0, scale_seed0 = self._make_batch(batch)
        B = occ_gt.shape[0]

        x_list, occ_vis_list = [], []
        for b in range(B):
            idx_np = seed_idx_list[b]
            feats_np = feats_list[b]
            idx_t = torch.from_numpy(idx_np).to(self.device).long()
            feats_t = torch.from_numpy(feats_np).to(self.device).float()
            with torch.no_grad():
                feats_comp = self.model.feature_compressor(feats_t).float()
            if self.remap_seed_feats and idx_t.numel() > 0:
                idx_dst = self._remap_seed_idx_with_bbox(idx_t, mean_seed0, scale_seed0, mean_gt, scale_gt, self.G, b)
            else:
                idx_dst = idx_t
            grid_dino, seed_occ = self.scatter_voxel_mean(idx_dst.int(), feats_comp, self.G)
            x_list.append(torch.cat([seed_occ, grid_dino], dim=1))
            occ_vis_list.append(seed_occ)

        x_in = torch.cat(x_list, dim=0)
        occ_vis = torch.cat(occ_vis_list, dim=0)
        occ_gt = occ_gt.to(self.device)

        with torch.no_grad():
            logits = self.model(x_in)
            probs = torch.sigmoid(logits)
            metrics = self._compute_iou(probs, occ_gt)

        tp = fp = fn = 0.0
        for b in range(B):
            tp_b, fp_b, fn_b = self._accumulate_voxel_stats(probs[b], occ_gt[b])
            tp += tp_b
            fp += fp_b
            fn += fn_b

        if self.save_vis:
            self._dump_visuals(scene_ids, occ_gt, logits, occ_vis)
        return metrics, B, tp, fp, fn

    def _make_batch(self, data_dict: Dict[str, Any]):
        sg = data_dict["scene_graphs"]
        scene_ids = [sid[0] for sid in sg["scene_ids"]]
        # _LOGGER.info("scene_ids %s", scene_ids)
        frame_ids_seq = sg.get("frame_ids")
        # _LOGGER.info("frame_ids_seq %s", frame_ids_seq)
        image_frames = sg.get("image_frames", {})
        # _LOGGER.info("image_frames %s", image_frames)
        frame_ids = sg.get("frame_ids",{})
        # _LOGGER.info("frame_ids %s", frame_ids)

        occ_gt_list = []
        seed_idx_list, feats_list = [], []
        mean_gt_list, scale_gt_list = [], []
        mean_seed0_list, scale_seed0_list = [], []

        for idx, sid in enumerate(scene_ids):
            fid = None
            if frame_ids_seq is not None and idx < len(frame_ids_seq):
                fid = frame_ids_seq[idx]
            if fid is None:
                frames = image_frames.get(sid, [])
                if frames:
                    fid = frames[0]
                else:
                    frame_ids = scan3r.load_frame_idxs(
                        osp.join(self.cfg.data.root_dir, "scenes"), sid
                    )
                    fid = frame_ids[0] if frame_ids else "000000"
            try:
                pack = self._load_aligned_pack(sid, str(fid))
                occ_gt, seed_idx, feats, mean_gt, scale_gt, mean_seed0, scale_seed0 = self._build_inputs(pack, self.G)
            except (zipfile.BadZipFile, ValueError, OSError) as exc:
                _LOGGER.warning("Skipping corrupted pack %s/%s: %s", sid, fid, exc)
                continue
            occ_gt_list.append(occ_gt.unsqueeze(0))
            seed_idx_list.append(seed_idx)
            feats_list.append(feats)
            mean_gt_list.append(mean_gt.unsqueeze(0))
            scale_gt_list.append(scale_gt.unsqueeze(0))
            mean_seed0_list.append(mean_seed0.unsqueeze(0))
            scale_seed0_list.append(scale_seed0.unsqueeze(0))

        if not occ_gt_list:
            raise RuntimeError("No valid packs found in batch during evaluation.")
        occ_gt = torch.cat(occ_gt_list, dim=0).to(self.device)
        mean_gt = torch.cat(mean_gt_list, dim=0).to(self.device)
        scale_gt = torch.cat(scale_gt_list, dim=0).to(self.device)
        mean_seed0 = torch.cat(mean_seed0_list, dim=0).to(self.device)
        scale_seed0 = torch.cat(scale_seed0_list, dim=0).to(self.device)
        return occ_gt, seed_idx_list, feats_list, mean_gt, scale_gt, mean_seed0, scale_seed0

    def _compute_iou(self, probs: torch.Tensor, gt: torch.Tensor, thresholds=(0.3, 0.5, 0.7)):
        metrics = {}
        gt_bin = (gt > 0.5).float()
        for thr in thresholds:
            pr = (probs >= thr).float()
            inter = (pr * gt_bin).sum(dim=(1, 2, 3, 4))
            union = pr.sum(dim=(1, 2, 3, 4)) + gt_bin.sum(dim=(1, 2, 3, 4)) - inter
            metrics[thr] = (inter / (union + 1e-6)).mean().item()
        return metrics

    def _accumulate_voxel_stats(self, probs: torch.Tensor, gt: torch.Tensor, thr: float = 0.5):
        pr = (probs >= thr).float()
        gt_bin = (gt > 0.5).float()
        tp = (pr * gt_bin).sum().item()
        fp = (pr * (1 - gt_bin)).sum().item()
        fn = ((1 - pr) * gt_bin).sum().item()
        return tp, fp, fn

    def _dump_visuals(self, scene_ids: List[str], occ_gt, logits, occ_vis):
        outdir = f"{self.cfg.output_dir}/unet_eval"
        os.makedirs(outdir, exist_ok=True)
        thr = 0.5
        for i, sid in enumerate(scene_ids[:4]):
            pr_idx = (logits[i, 0].sigmoid() > thr).nonzero(as_tuple=False)
            gt_idx = (occ_gt[i, 0] > thr).nonzero(as_tuple=False)
            vis_idx = (occ_vis[i, 0] > thr).nonzero(as_tuple=False)
            save_vox_as_ply(gt_idx, self.G, f"{outdir}/{sid}_gt.ply")
            save_vox_as_ply(pr_idx, self.G, f"{outdir}/{sid}_pred.ply")
            save_vox_as_ply(vis_idx, self.G, f"{outdir}/{sid}_input.ply")
            sbs = side_by_side(occ_gt[i, 0].cpu(), logits[i, 0].cpu(), max_slices=6)
            save_image(sbs, os.path.join(outdir, f"{sid}_slices.png"), normalize=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="UNet checkpoint path")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--scene_ids", type=str, nargs="*", default=None)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args, unknown = parser.parse_known_args()
    return parser, args, unknown


def main():
    common.init_log(level=logging.INFO)
    parser, args, unknown = parse_args()
    cfg = update_configs(args.config, unknown)
    evaluator = UNetCompletionEvaluator(
        cfg,
        split=args.split,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scene_ids=args.scene_ids,
        save_vis=args.save_vis,
        max_batches=args.max_batches,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
