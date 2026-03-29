import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import Config, update_configs
from src.datasets import Scan3RSceneBatchDataset
from src.models.structure_model_with_bbox_head import StructureModel
from utils import common, scan3r, torch_util

_LOGGER = logging.getLogger(__name__)


class StructureModelEvaluator:
    """Evaluate structure completion model on bounding box IoU and voxel metrics."""

    def __init__(
        self,
        cfg: Config,
        split: str = "val",
        checkpoint: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        max_batches: Optional[int] = None,
        scene_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint).to(self.device).eval()
        self.G = 64
        self.max_batches = max_batches
        self.target_scene_ids = set(scene_ids) if scene_ids else None

        val_cfg = getattr(cfg, "val", None)
        default_batch_size = getattr(val_cfg, "batch_size", 1) if val_cfg else 1
        default_num_workers = getattr(val_cfg, "num_workers", 0) if val_cfg else 0

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

    def _load_model(self, checkpoint: Optional[str]) -> StructureModel:
        model = StructureModel(cfg=self.cfg.autoencoder, device=self.device)
        if checkpoint is None:
            inference_cfg = getattr(self.cfg, "inference", None)
            checkpoint = getattr(inference_cfg, "structure_model_path", None)
        if checkpoint is None:
            raise ValueError(
                "No checkpoint specified. Provide --checkpoint or set cfg.inference.structure_model_path."
            )
        state = torch.load(checkpoint, map_location=self.device)
        state_dict = state.get("model", state)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    # ---------- Data preparation helpers ----------

    def _load_aligned_pack(self, scene_id: str, frame_id: str) -> np.lib.npyio.NpzFile:
        root_dir_scratch = "/cluster/scratch/wangyih/3RScan"
        base = osp.join(
            root_dir_scratch,
            "files",
            "gs_annotations",
            scene_id,
            "scene_level_structure",
        )
        pack_path = osp.join(base, f"student_pack_aligned_{frame_id}.npz")
        if not osp.exists(pack_path):
            cand = [
                f
                for f in os.listdir(base)
                if f.startswith("student_pack_aligned_") and f.endswith(".npz")
            ]
            if not cand:
                raise FileNotFoundError(f"No aligned pack found under {base}")
            pack_path = osp.join(base, cand[0])
        return np.load(pack_path)

    @staticmethod
    def _rasterize_idx(idx_np: np.ndarray, G: int) -> torch.Tensor:
        occ = np.zeros((G, G, G), dtype=np.uint8)
        if idx_np.size:
            occ[idx_np[:, 0], idx_np[:, 1], idx_np[:, 2]] = 1
        return torch.from_numpy(occ).unsqueeze(0).float()

    def _build_inputs(
        self, pack: np.lib.npyio.NpzFile, expected_G: int
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        G = int(pack["G"])
        if G != expected_G:
            raise ValueError(f"G mismatch: expected {expected_G}, got {G}")

        occ_gt = self._rasterize_idx(pack["vox_idx_gt_occ"], G)

        seed_idx = pack.get("seed_idx", np.zeros((0, 3), np.int32))
        feats = pack.get("feats", np.zeros((0, 1024), np.float32))

        mean_gt = torch.from_numpy(pack["mean_gt"].astype(np.float32))
        scale_gt = torch.tensor(float(pack["scale_gt"]), dtype=torch.float32).view(1)
        mean_seed0 = torch.from_numpy(pack["seed_box_init_mean"].astype(np.float32))
        scale_seed0 = torch.tensor(
            float(pack["seed_box_init_scale"]), dtype=torch.float32
        ).view(1)

        return (
            occ_gt,
            seed_idx,
            feats,
            mean_gt,
            scale_gt,
            mean_seed0,
            scale_seed0,
        )

    def _choose_frame(self, scene_id: str, frame_candidates: Optional[List[str]]) -> str:
        if frame_candidates and len(frame_candidates) > 0:
            return random.choice(frame_candidates)

        frame_ids = scan3r.load_frame_idxs(
            osp.join(self.cfg.data.root_dir, "scenes"), scene_id
        )
        if len(frame_ids) > 60:
            frame_ids = frame_ids[:60]
        if not frame_ids:
            raise RuntimeError(f"No frames available for scene {scene_id}")
        return random.choice(frame_ids)

    def _build_scene_graph_batch(self, raw_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        scene_ids = [sample["scan_id"] for sample in raw_batch]
        scene_graphs = {
            "scene_ids": np.array([[sid] for sid in scene_ids]),
            "obj_intrinsics": {sid: self.dataset.image_intrinsics[sid] for sid in scene_ids},
        }
        image_frames = defaultdict(list)
        for sample in raw_batch:
            sid = sample["scan_id"]
            fid = sample.get("frame_idx")
            if fid is not None:
                image_frames[sid].append(fid)
        scene_graphs["image_frames"] = {
            sid: list(dict.fromkeys(fids)) for sid, fids in image_frames.items()
        }
        return {"scene_graphs": scene_graphs}

    def _make_batch(
        self, data_dict: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor,
        List[np.ndarray],
        List[np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[str],
    ]:
        sg = data_dict["scene_graphs"]
        scene_ids_arr = sg["scene_ids"]
        scene_ids = [sid[0] for sid in scene_ids_arr]
        image_frames = sg.get("image_frames", defaultdict(list))

        occ_gt_list = []
        seed_idx_list: List[np.ndarray] = []
        feats_list: List[np.ndarray] = []
        mean_gt_list = []
        scale_gt_list = []
        mean_seed0_list = []
        scale_seed0_list = []

        kept_scene_ids: List[str] = []

        for sid in scene_ids:
            if self.target_scene_ids and sid not in self.target_scene_ids:
                continue
            fid = self._choose_frame(sid, image_frames.get(sid))
            pack = self._load_aligned_pack(sid, fid)
            (
                occ_gt,
                seed_idx,
                feats,
                mean_gt,
                scale_gt,
                mean_seed0,
                scale_seed0,
            ) = self._build_inputs(pack, self.G)

            occ_gt_list.append(occ_gt.unsqueeze(0))
            seed_idx_list.append(seed_idx)
            feats_list.append(feats)
            mean_gt_list.append(mean_gt.unsqueeze(0))
            scale_gt_list.append(scale_gt.unsqueeze(0))
            mean_seed0_list.append(mean_seed0.unsqueeze(0))
            scale_seed0_list.append(scale_seed0.unsqueeze(0))
            kept_scene_ids.append(sid)

        if not kept_scene_ids:
            empty = torch.zeros(0, 1, self.G, self.G, self.G, device=self.device)
            return (
                empty,
                [],
                [],
                torch.zeros(0, 3, device=self.device),
                torch.zeros(0, 1, device=self.device),
                torch.zeros(0, 3, device=self.device),
                torch.zeros(0, 1, device=self.device),
                [],
            )

        occ_gt = torch.cat(occ_gt_list, 0).to(self.device)  # (B,1,G,G,G)
        mean_gt = torch.cat(mean_gt_list, 0).to(self.device)
        scale_gt = torch.cat(scale_gt_list, 0).to(self.device)
        mean_seed0 = torch.cat(mean_seed0_list, 0).to(self.device)
        scale_seed0 = torch.cat(scale_seed0_list, 0).to(self.device)

        return (
            occ_gt,
            seed_idx_list,
            feats_list,
            mean_gt,
            scale_gt,
            mean_seed0,
            scale_seed0,
            kept_scene_ids,
        )

    # ---------- Geometry helpers ----------

    @staticmethod
    def _idx_to_centers(idx: torch.Tensor, G: int) -> torch.Tensor:
        return (idx.float() + 0.5) / G - 0.5

    @staticmethod
    def _centers_to_idx(centers: torch.Tensor, G: int) -> torch.Tensor:
        idx = torch.floor((centers + 0.5) * G).long()
        return torch.clamp(idx, 0, G - 1)

    def _remap_seed_idx_with_bbox(
        self,
        seed_idx: torch.Tensor,
        mean_src: torch.Tensor,
        scale_src: torch.Tensor,
        mean_dst: torch.Tensor,
        scale_dst: torch.Tensor,
        batch_index: int,
    ) -> torch.Tensor:
        if seed_idx.numel() == 0:
            return seed_idx
        c_seed = self._idx_to_centers(seed_idx, self.G)
        s_src = scale_src[batch_index].view(1, 1).clamp_min(1e-6)
        m_src = mean_src[batch_index].view(1, 3)
        world = c_seed * (2.0 * s_src) + m_src

        s_dst = scale_dst[batch_index].view(1, 1).clamp_min(1e-6)
        m_dst = mean_dst[batch_index].view(1, 3)
        centers_dst = (world - m_dst) / (2.0 * s_dst)
        return self._centers_to_idx(centers_dst, self.G)

    def remap_occ(
        self,
        occ_src: torch.Tensor,
        mean_src: torch.Tensor,
        scale_src: torch.Tensor,
        mean_dst: torch.Tensor,
        scale_dst: torch.Tensor,
    ) -> torch.Tensor:
        B = occ_src.shape[0]
        occ_dst = torch.zeros_like(occ_src)
        for b in range(B):
            idx_src = (occ_src[b, 0] > 0.5).nonzero(as_tuple=False)
            if idx_src.numel() == 0:
                continue
            idx_dst = self._remap_seed_idx_with_bbox(
                idx_src, mean_src, scale_src, mean_dst, scale_dst, b
            )
            occ_dst[b, 0, idx_dst[:, 0], idx_dst[:, 1], idx_dst[:, 2]] = 1.0
        return occ_dst

    @staticmethod
    def scatter_voxel_mean(
        idx_t: torch.Tensor, feat_t: torch.Tensor, G: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx_t.numel() == 0:
            C = feat_t.shape[-1] if feat_t.ndim == 2 else 64
            grid_feats = torch.zeros(
                1, C, G, G, G, device=feat_t.device, dtype=feat_t.dtype
            )
            seed_occ = torch.zeros(
                1, 1, G, G, G, device=feat_t.device, dtype=feat_t.dtype
            )
            return grid_feats, seed_occ

        M, C = feat_t.shape
        lin = (idx_t[:, 0] * G * G + idx_t[:, 1] * G + idx_t[:, 2]).long()

        Csum = torch.zeros(C, G * G * G, device=feat_t.device, dtype=feat_t.dtype)
        cnt = torch.zeros(G * G * G, device=feat_t.device, dtype=feat_t.dtype)

        Csum.index_add_(1, lin, feat_t.T)
        cnt.index_add_(0, lin, torch.ones(M, device=feat_t.device, dtype=feat_t.dtype))

        mask = cnt > 0
        Csum[:, mask] = Csum[:, mask] / cnt[mask]
        grid_feats = Csum.view(C, G, G, G).unsqueeze(0)

        seed_occ = torch.zeros(
            1, 1, G, G, G, device=feat_t.device, dtype=feat_t.dtype
        )
        uniq = torch.unique(lin)
        seed_occ.view(1, 1, -1)[0, 0, uniq] = 1.0
        return grid_feats, seed_occ

    # ---------- Metrics ----------

    @staticmethod
    def _bbox_iou(
        mean_pred: torch.Tensor,
        scale_pred: torch.Tensor,
        mean_gt: torch.Tensor,
        scale_gt: torch.Tensor,
    ) -> torch.Tensor:
        scale_pred = scale_pred.view(-1, 1).clamp_min(1e-6)
        scale_gt = scale_gt.view(-1, 1).clamp_min(1e-6)

        pred_min = mean_pred - scale_pred
        pred_max = mean_pred + scale_pred
        gt_min = mean_gt - scale_gt
        gt_max = mean_gt + scale_gt

        inter_min = torch.maximum(pred_min, gt_min)
        inter_max = torch.minimum(pred_max, gt_max)
        inter_size = torch.clamp(inter_max - inter_min, min=0.0)
        inter_vol = inter_size.prod(dim=1)

        side_pred = 2.0 * scale_pred.squeeze(-1)
        side_gt = 2.0 * scale_gt.squeeze(-1)
        vol_pred = side_pred.pow(3)
        vol_gt = side_gt.pow(3)
        union = torch.clamp(vol_pred + vol_gt - inter_vol, min=1e-6)
        return inter_vol / union

    @staticmethod
    def _accumulate_voxel_stats(
        probs: torch.Tensor, gt: torch.Tensor, thr: float = 0.5
    ) -> Tuple[int, int, int]:
        pred = (probs >= thr)
        gt_mask = gt >= 0.5
        tp = torch.logical_and(pred, gt_mask).sum().item()
        fp = torch.logical_and(pred, torch.logical_not(gt_mask)).sum().item()
        fn = torch.logical_and(torch.logical_not(pred), gt_mask).sum().item()
        return tp, fp, fn

    # ---------- Evaluation ----------

    def evaluate(self) -> Dict[str, float]:
        total_tp = total_fp = total_fn = 0
        bbox_ious = []
        num_batches = 0

        total_batches = len(self.dataloader)
        if self.max_batches is not None:
            total_batches = min(total_batches, self.max_batches)

        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(
                tqdm(self.dataloader, total=total_batches, desc="Evaluating")
            ):
                if batch_idx >= total_batches:
                    break
                if not raw_batch:
                    continue
                num_batches += 1
                data_dict = self._build_scene_graph_batch(raw_batch)

                (
                    occ_gt,
                    seed_idx_list,
                    feats_list,
                    mean_gt,
                    scale_gt,
                    mean_seed0,
                    scale_seed0,
                    scene_ids,
                ) = self._make_batch(data_dict)

                B = occ_gt.shape[0]
                if B == 0:
                    continue
                feats_comp_list: List[torch.Tensor] = []
                idx_t_list: List[torch.Tensor] = []

                for b in range(B):
                    idx_np = seed_idx_list[b]
                    feats_np = feats_list[b]
                    idx_t = torch.from_numpy(idx_np).to(self.device).long()
                    feats_t = torch.from_numpy(feats_np).to(self.device).float()
                    with torch.amp.autocast(
                        "cuda",
                        enabled=self.device.type == "cuda",
                        dtype=torch.float16,
                    ):
                        feats_comp = self.model.comp(feats_t).float()
                    feats_comp_list.append(feats_comp)
                    idx_t_list.append(idx_t)

                mean_pred, scale_pred = self.model.forward_bbox_from_seeds_batch(
                    feats_comp_list=feats_comp_list,
                    idx_list=idx_t_list,
                    G=self.G,
                    mean_seed0=mean_seed0,
                    scale_seed0=scale_seed0,
                )
                bbox_ious.extend(
                    self._bbox_iou(mean_pred, scale_pred, mean_gt, scale_gt.view(-1))
                    .detach()
                    .cpu()
                    .tolist()
                )

                mean_dst = mean_gt
                scale_dst = scale_gt.view(B, 1)

                occ_gt_aligned = self.remap_occ(
                    occ_src=occ_gt,
                    mean_src=mean_gt,
                    scale_src=scale_gt.view(B, 1),
                    mean_dst=mean_dst,
                    scale_dst=scale_dst,
                )

                x_dst_list = []
                for b in range(B):
                    idx_dst = self._remap_seed_idx_with_bbox(
                        seed_idx=idx_t_list[b],
                        mean_src=mean_seed0,
                        scale_src=scale_seed0,
                        mean_dst=mean_dst,
                        scale_dst=scale_dst,
                        batch_index=b,
                    )
                    grid_dino_dst, seed_occ_dst = self.scatter_voxel_mean(
                        idx_dst.int(), feats_comp_list[b], self.G
                    )
                    x_dst = torch.cat([seed_occ_dst, grid_dino_dst], dim=1)
                    x_dst_list.append(x_dst)

                x_in = torch.cat(x_dst_list, dim=0).to(self.device)

                with torch.amp.autocast(self.device.type, enabled=False):
                    z, mu, _logvar, _feat = self.model.encoder(
                        x_in,
                        sample_posterior=False,
                        return_raw=True,
                        return_feat=True,
                    )
                    logits = self.model.decoder(mu)

                probs = torch.sigmoid(logits.float())
                gt = occ_gt_aligned.float()

                for b in range(B):
                    tp, fp, fn = self._accumulate_voxel_stats(
                        probs[b], gt[b], thr=0.5
                    )
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = (
            2 * precision * recall / (precision + recall + 1e-8)
            if (precision + recall) > 0
            else 0.0
        )

        bbox_iou_mean = float(np.mean(bbox_ious)) if bbox_ious else 0.0

        return {
            "bbox_iou": bbox_iou_mean,
            "voxel_precision": precision,
            "voxel_recall": recall,
            "voxel_f1": f1,
            "voxel_tp": total_tp,
            "voxel_fp": total_fp,
            "voxel_fn": total_fn,
        }


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate structure model on bounding box and voxel metrics."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--scene_ids", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_known_args()


def main() -> None:
    common.init_log(level=logging.INFO)
    args, unknown = parse_args()
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)

    evaluator = StructureModelEvaluator(
        cfg=cfg,
        split=args.split,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        scene_ids=args.scene_ids,
    )

    metrics = evaluator.evaluate()
    _LOGGER.info(
        "bbox IoU: %.4f | voxel Precision: %.4f | Recall: %.4f | F1: %.4f",
        metrics["bbox_iou"],
        metrics["voxel_precision"],
        metrics["voxel_recall"],
        metrics["voxel_f1"],
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        _LOGGER.info("Saved metrics to %s", args.output)


if __name__ == "__main__":
    main()
