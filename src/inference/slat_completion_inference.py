
import argparse
import copy
import logging
import os
import os.path as osp
import math
import json
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple
from src.models.losses.reconstruction import LPIPS
import torch.nn.functional as F
from utils.loss_utils import ssim
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from configs import Config, update_configs
from gaussian_renderer import render
from scene.cameras import MiniCam2
from src.datasets import Scan3RSceneBatchDataset
from src.models.latent_autoencoder import LatentAutoencoder
from src.models.unet3d_completion import UNetCompletionModel
from src.modules.sparse.basic import SparseTensor
from utils import common, scan3r
from utils.graphics_utils import focal2fov
from utils.visualisation import save_vox_as_ply


LOGGER = logging.getLogger(__name__)


class SlatCompletionInference:
    def __init__(self, cfg: Config, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder_cfg = getattr(cfg.autoencoder, "encoder", None)
        self.grid_res = getattr(encoder_cfg, "resolution", 128)
        self.latent_dim = getattr(encoder_cfg, "latent_channels", 16)
        self.lpips = LPIPS(device="cuda" if self.device.type == "cuda" else "cpu").eval()
        self.occ_threshold = args.occ_threshold
        self.max_latents = args.max_latents
        self.eval_threshold = getattr(args, "eval_threshold", 0.5)
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._pipe_cfg = Namespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )

        self.unet: Optional[UNetCompletionModel] = None
        self.seed_feat_dim: Optional[int] = None
        self.latent_model: Optional[LatentAutoencoder] = None
        self.dataset: Optional[Scan3RSceneBatchDataset] = None

    @staticmethod
    def _idx_to_centers(idx: torch.Tensor, G: int) -> torch.Tensor:
        if idx.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32, device=idx.device)
        return (idx.float() + 0.5) / G - 0.5

    @staticmethod
    def _centers_to_idx(centers: torch.Tensor, G: int) -> torch.Tensor:
        if centers.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.int32, device=centers.device)
        idx = torch.floor((centers + 0.5) * G).long()
        return torch.clamp(idx, 0, G - 1)
    def _compute_photometric_metrics(
            self,
            pred_img: torch.Tensor,
            gt_img: torch.Tensor,
        ) -> Dict[str, float]:
        """
        pred_img, gt_img: (3, H, W) in [0,1]
        Returns per-frame mse/ssim/lpips.
        """
        pred = pred_img.clamp(0.0, 1.0)
        gt = gt_img.clamp(0.0, 1.0)

        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)

        mse_val = F.mse_loss(pred, gt, reduction="mean").item()
        ssim_val = ssim(pred, gt).item()
        lpips_val = self.lpips(pred, gt).item()

        return {
            "mse": mse_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
        }
    def _remap_seed_idx_with_bbox(
        self,
        seed_idx: torch.Tensor,
        mean_src: torch.Tensor,
        scale_src: torch.Tensor,
        mean_dst: torch.Tensor,
        scale_dst: torch.Tensor,
        G: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if seed_idx.numel() == 0:
            return seed_idx.new_zeros((0, 3))

        c_seed = self._idx_to_centers(seed_idx, G)

        s_src = scale_src.view(-1)
        if s_src.numel() == 1:
            s_src = s_src.repeat(3)
        elif s_src.numel() > 3:
            s_src = s_src[:3]
        s_src = torch.clamp(s_src.view(1, 3), min=eps)

        m_src = mean_src.view(1, 3)
        world = c_seed * (2.0 * s_src) + m_src

        s_dst = scale_dst.view(-1)
        if s_dst.numel() == 1:
            s_dst = s_dst.repeat(3)
        elif s_dst.numel() > 3:
            s_dst = s_dst[:3]
        s_dst = torch.clamp(s_dst.view(1, 3), min=eps)

        m_dst = mean_dst.view(1, 3)
        c_dst = (world - m_dst) / (2.0 * s_dst)
        return self._centers_to_idx(c_dst, G)

    def scatter_voxel_mean(self, idx_t: torch.Tensor, feat_t: torch.Tensor, G: int):
        if idx_t.numel() == 0 or feat_t.numel() == 0:
            C = feat_t.shape[-1] if feat_t.ndim == 2 else (self.seed_feat_dim or 64)
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

    def _get_dataset(self, split: str = "train", use_obj_id_filter: bool = False) -> Scan3RSceneBatchDataset:
        if (
            self.dataset is None
            or self.dataset.split != split
            or getattr(self.dataset, "use_obj_id_filter", False) != use_obj_id_filter
        ):
            cfg_local = copy.deepcopy(self.cfg)
            setattr(cfg_local.data, "use_student_structure", True)
            setattr(cfg_local.data, "use_obj_id_filter", use_obj_id_filter)
            self.dataset = Scan3RSceneBatchDataset(cfg_local, split=split)
        return self.dataset

    def _build_training_style_student_sample(
        self, scene_id: str, frame_id: str, split: str = "train", use_obj_id_filter: bool = False
    ) -> Dict[str, Any]:
        dataset = self._get_dataset(split=split, use_obj_id_filter=use_obj_id_filter)
        sample = dataset._build_student_pack_splat(scene_id, frame_id)
        sample["frame_id_used"] = str(frame_id).zfill(6)
        return sample

    def _infer_feat_dim_from_sample(self, sample: Dict[str, Any]) -> int:
        raw = sample.get("student_seed_feats_raw")
        if raw is None:
            return getattr(self.cfg.data, "student_feat_dim", 1024)
        if isinstance(raw, torch.Tensor) and raw.ndim == 2 and raw.shape[1] > 0:
            return int(raw.shape[1])
        return getattr(self.cfg.data, "student_feat_dim", 1024)
        # return 8

    def _init_teacher(self) -> None:
        if self.latent_model is not None:
            return

        model = LatentAutoencoder(cfg=self.cfg.autoencoder, device=self.device)
        checkpoint = self.args.teacher_checkpoint
        if checkpoint is None:
            raise ValueError("--teacher_checkpoint must be provided")

        state = torch.load(checkpoint, map_location=self.device)
        missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
        LOGGER.info(
            "Teacher load: missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )

        model.eval()
        inferred_dim = self._infer_teacher_latent_dim(model.encoder)
        if inferred_dim is not None and inferred_dim != self.latent_dim:
            LOGGER.warning(
                "Teacher latent dim %d overrides config latent dim %d",
                inferred_dim,
                self.latent_dim,
            )
            self.latent_dim = inferred_dim

        self.latent_model = model

    @staticmethod
    def _infer_teacher_latent_dim(encoder) -> Optional[int]:
        out_layer = getattr(encoder, "out_layer", None)
        if out_layer is None:
            return None
        out_features = getattr(out_layer, "out_features", None)
        if isinstance(out_features, int) and out_features > 0:
            return out_features // 2
        return None

    def _init_unet(self, feat_dim: int) -> None:
        if self.unet is not None:
            return

        self._init_teacher()
        self.seed_feat_dim = feat_dim
        LOGGER.info(f"feat_dim {feat_dim}")

        model = UNetCompletionModel(
            feat_in=feat_dim,
            out_channels=1 + self.latent_dim,
        ).to(self.device)

        checkpoint = self.args.unet_checkpoint
        if checkpoint is None:
            raise ValueError("--unet_checkpoint is required")

        state = torch.load(checkpoint, map_location=self.device)
        missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
        LOGGER.info(
            "UNet load: missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
        if missing:
            LOGGER.info("UNet missing keys (first 20): %s", missing[:20])
        if unexpected:
            LOGGER.info("UNet unexpected keys (first 20): %s", unexpected[:20])

        model.train()
        # model.eval()
        self.unet = model

    def _prepare_inputs_from_training_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.unet is None:
            raise RuntimeError("UNet must be initialized first")

        G = int(sample["student_grid_resolution"])
        if G != self.grid_res:
            raise ValueError(f"Grid mismatch: sample {G}, model {self.grid_res}")

        gt_idx = sample["student_gt_indices"].to(self.device).long()
        seed_idx_raw = sample["student_seed_indices_raw"].to(self.device).long()
        seed_feats_raw = sample["student_seed_feats_raw"].to(self.device).float()
        mean_seed = sample["student_seed_mean"].to(self.device).float()
        scale_seed = sample["student_seed_scale"].to(self.device).float()
        mean_gt = sample["student_mean_gt"].to(self.device).float()
        scale_gt = sample["student_scale_gt"].to(self.device).float()

        occ_gt = torch.zeros(1, 1, G, G, G, device=self.device)
        if gt_idx.numel():
            occ_gt[0, 0, gt_idx[:, 0], gt_idx[:, 1], gt_idx[:, 2]] = 1.0

        with torch.no_grad():
            comp = self.unet.feature_compressor(seed_feats_raw).float()

        idx_dst = self._remap_seed_idx_with_bbox(
            seed_idx_raw,
            mean_seed,
            scale_seed,
            mean_gt,
            scale_gt,
            G,
        )

        grid_feats, seed_occ = self.scatter_voxel_mean(idx_dst.int(), comp.float(), G)
        x_in = torch.cat([seed_occ, grid_feats], dim=1)

        meta = {
            "mean_gt": mean_gt.detach().cpu().numpy().astype(np.float32),
            "scale_gt": scale_gt.detach().cpu().numpy().astype(np.float32),
            "R_frame": sample.get("R_cans", torch.eye(3)).detach().cpu().numpy().astype(np.float32),
            "frame_id_used": sample.get("frame_id_used"),
        }

        return {
            "x_in": x_in,
            "occ_gt": occ_gt,
            "gt_idx": gt_idx,
            "seed_occ": seed_occ,
            "meta": meta,
        }

    def _run_unet(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.unet is None:
            raise RuntimeError("UNet not initialized")
        with torch.no_grad():
            logits_full = self.unet(x_in)
        logits = logits_full[:, :1]
        latents = logits_full[:, 1:]
        return logits, latents

    def _logits_to_sparse_latent(
        self, latents: torch.Tensor, logits: torch.Tensor
    ) -> SparseTensor:
        thr = torch.tensor(self.occ_threshold, device=logits.device, dtype=logits.dtype)
        thr = torch.logit(thr.clamp(1e-4, 1.0 - 1e-4))

        mask = logits > thr
        coords = mask.nonzero(as_tuple=False)
        scores = logits[mask].sigmoid()

        if coords.shape[0] == 0:
            LOGGER.warning(
                "No voxels exceed occ threshold %.2f; falling back to top responses.",
                self.occ_threshold,
            )
            flat = logits.flatten()
            k = flat.numel() if self.max_latents is None else min(self.max_latents, flat.numel())
            vals, idxs = torch.topk(flat, k)
            coords = torch.stack(torch.unravel_index(idxs, logits.shape), dim=1)
            scores = vals.sigmoid()

        if self.max_latents is not None and coords.shape[0] > self.max_latents:
            vals, order = torch.topk(scores, self.max_latents)
            coords = coords[order]

        feats = latents[:, coords[:, 0], coords[:, 1], coords[:, 2]].permute(1, 0)
        batch_col = torch.zeros(coords.shape[0], 1, dtype=torch.long, device=coords.device)
        coords4 = torch.cat([batch_col, coords.long()], dim=1)
        return SparseTensor(feats=feats.contiguous(), coords=coords4.int()).to(self.device)

    def _normalize_rotation(self, rot: Optional[np.ndarray]) -> np.ndarray:
        if rot is None:
            return np.eye(3, dtype=np.float32)
        rot = np.asarray(rot)
        if rot.ndim == 3:
            rot = rot[0]
        if rot.ndim == 1 and rot.size == 9:
            rot = rot.reshape(3, 3)
        if rot.shape != (3, 3):
            return np.eye(3, dtype=np.float32)
        return rot.astype(np.float32)

    def _apply_pack_alignment(self, gauss, mean_gt: np.ndarray, scale_gt: np.ndarray) -> None:
        device = gauss.get_xyz.device
        gauss.rescale(torch.tensor([2.0, 2.0, 2.0], device=device))
        gauss.translate(torch.tensor([-1.0, -1.0, -1.0], device=device))

        scale_arr = np.asarray(scale_gt, dtype=np.float32)
        scale_vec = torch.from_numpy(scale_arr).to(device=device).view(-1)
        if scale_vec.numel() == 1:
            scale_vec = scale_vec.repeat(3)
        elif scale_vec.numel() > 3:
            scale_vec = scale_vec[:3]
        gauss.rescale(scale_vec)

        translation = torch.as_tensor(mean_gt, device=device).view(-1)
        if translation.numel() == 1:
            translation = translation.repeat(3)
        gauss.translate(translation)

        self._clamp_gaussian_scale(gauss, scale_gt)

    def _clamp_gaussian_scale(self, reconstruction, bbox_scale: np.ndarray) -> None:
        device = reconstruction.get_xyz.device
        dtype = reconstruction.get_xyz.dtype

        if not torch.is_tensor(bbox_scale):
            bbox_scale = torch.tensor(bbox_scale, device=device, dtype=dtype)

        bbox_scale = bbox_scale.to(device=device, dtype=dtype).flatten()
        if bbox_scale.numel() == 0:
            return
        if bbox_scale.numel() == 1:
            bbox_scale = bbox_scale.repeat(3)
        elif bbox_scale.numel() > 3:
            bbox_scale = bbox_scale[:3]

        max_scale = (bbox_scale / self.grid_res).clamp_min(1e-5)
        current_scale = reconstruction.get_scaling
        clamped = torch.minimum(current_scale, max_scale.view(1, 3))
        reconstruction.from_scaling(clamped)

    def _render_prediction(
        self,
        gauss,
        scene_id: str,
        frame_id: str,
        rot_can: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        root = self.cfg.data.root_dir
        intr = scan3r.load_intrinsics(osp.join(root, "scenes"), scene_id)

        H = int(intr["height"])
        W = int(intr["width"])
        fx = float(intr["intrinsic_mat"][0, 0])
        fy = float(intr["intrinsic_mat"][1, 1])

        fovx = focal2fov(fx, W)
        fovy = focal2fov(fy, H)

        frame = str(frame_id).zfill(6)
        poses = scan3r.load_frame_poses(root, scene_id, (frame,))
        extr = poses[frame]

        R_h = np.eye(4, dtype=np.float32)
        R_h[:3, :3] = rot_can
        if getattr(self.args, "pose_is_w2c", False):
            extr = extr @ R_h.T
        else:
            extr = R_h @ extr

        cam_to_world = np.linalg.inv(extr)
        camera = MiniCam2(
            width=W,
            height=H,
            fovy=fovy,
            fovx=fovx,
            znear=0.01,
            zfar=100.0,
            R=cam_to_world[:3, :3].T,
            T=cam_to_world[:3, 3],
        )

        xyz = gauss.get_xyz
        if xyz.numel() == 0 or not torch.isfinite(xyz).all():
            raise RuntimeError(f"Gaussians for {scene_id}/{frame_id} are empty or contain NaN/Inf — skipping render")

        with torch.no_grad():
            rendered = render(
                camera,
                gauss,
                pipe=self._pipe_cfg,
                bg_color=torch.tensor((0.0, 0.0, 0.0), device=self.device),
            )["render"]

        img_path = osp.join(
            root,
            "scenes",
            scene_id,
            "sequence",
            f"frame-{frame}.color.jpg",
        )
        gt = Image.open(img_path).convert("RGB")
        gt_t = torch.from_numpy(np.array(gt)).permute(2, 0, 1).float() / 255.0
        return gt_t.to(rendered.device), rendered

    def _predict_reconstruction(
        self, scene_id: str, frame_id: str, split: str = "train", save_debug: bool = False,
        use_obj_id_filter: bool = False,
    ):
        sample = self._build_training_style_student_sample(
            scene_id, frame_id, split=split, use_obj_id_filter=use_obj_id_filter
        )
        feat_dim = self._infer_feat_dim_from_sample(sample)
        self._init_unet(feat_dim)

        prepared = self._prepare_inputs_from_training_sample(sample)
        x_in = prepared["x_in"]
        seed_occ = x_in[0, 0]  # (G,G,G)

        idx_seed = (seed_occ > 0).nonzero(as_tuple=False)

        logits, latents = self._run_unet(x_in)

        pred_idx = (logits[0, 0].sigmoid() > self.occ_threshold).nonzero(as_tuple=False)
        if save_debug:
            save_vox_as_ply(
                idx_seed.detach().cpu(),
                self.grid_res,
                osp.join(self.output_dir, f"{scene_id}_{str(frame_id).zfill(6)}_x_in.ply"),
            )
            save_vox_as_ply(
                pred_idx.detach().cpu(),
                self.grid_res,
                osp.join(self.output_dir, f"{scene_id}_{str(frame_id).zfill(6)}_pred_occ.ply"),
            )
            save_vox_as_ply(
                prepared["gt_idx"].detach().cpu(),
                self.grid_res,
                osp.join(self.output_dir, f"{scene_id}_{str(frame_id).zfill(6)}_gt_occ.ply"),
            )

        sparse_latent = self._logits_to_sparse_latent(latents[0], logits[0, 0])

        if self.latent_model is None:
            raise RuntimeError("Teacher decoder not initialized")

        with torch.no_grad():
            reconstruction = self.latent_model.decode(sparse_latent)

        mean_gt = prepared["meta"]["mean_gt"]
        scale_gt = prepared["meta"]["scale_gt"]
        for gauss in reconstruction:
            self._apply_pack_alignment(gauss, mean_gt, scale_gt)

        return reconstruction, prepared, logits

    @staticmethod
    def _compute_prf_metrics(probs: torch.Tensor, gt: torch.Tensor, threshold: float) -> Dict[str, float]:
        gt_bin = (gt > 0.5).float()
        pred = (probs >= threshold).float()
        tp = (pred * gt_bin).sum().item()
        fp = (pred * (1.0 - gt_bin)).sum().item()
        fn = ((1.0 - pred) * gt_bin).sum().item()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        fscore = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
        }

    # def evaluate_split(
    #     self,
    #     split: str,
    #     threshold: Optional[float] = None,
    #     max_samples: Optional[int] = None,
    # ) -> Dict[str, float]:
    #     eval_thr = self.eval_threshold if threshold is None else threshold
    #     dataset = self._get_dataset(split=split)

    #     total_tp = total_fp = total_fn = 0.0
    #     processed = 0

    #     for item in dataset.data_items:
    #         if max_samples is not None and processed >= max_samples:
    #             break

    #         scan_id = item.get("scan_id")
    #         frame_idx = item.get("frame_idx")
    #         if scan_id is None or frame_idx is None:
    #             continue

    #         fid = str(frame_idx).zfill(6)

    #         try:
    #             sample = self._build_training_style_student_sample(scan_id, fid, split=split)
    #         except FileNotFoundError as err:
    #             LOGGER.warning("Skipping %s/%s: %s", scan_id, fid, err)
    #             continue

    #         feat_dim = self._infer_feat_dim_from_sample(sample)
    #         self._init_unet(feat_dim)

    #         prepared = self._prepare_inputs_from_training_sample(sample)
    #         logits, _ = self._run_unet(prepared["x_in"])
    #         probs = torch.sigmoid(logits)

    #         metrics = self._compute_prf_metrics(probs, prepared["occ_gt"], eval_thr)
    #         total_tp += metrics["tp"]
    #         total_fp += metrics["fp"]
    #         total_fn += metrics["fn"]
    #         processed += 1

    #     if processed == 0:
    #         raise RuntimeError(f"No samples evaluated for split {split}")

    #     precision = total_tp / (total_tp + total_fp + 1e-6)
    #     recall = total_tp / (total_tp + total_fn + 1e-6)
    #     fscore = 2 * precision * recall / (precision + recall + 1e-6)

    #     results = {
    #         "precision": precision,
    #         "recall": recall,
    #         "fscore": fscore,
    #         "samples": processed,
    #     }

    #     LOGGER.info(
    #         "[%s split] precision=%.4f recall=%.4f F-score=%.4f over %d samples",
    #         split,
    #         precision,
    #         recall,
    #         fscore,
    #         processed,
    #     )
    #     return results
    def _compute_iou(
        self,
        probs: torch.Tensor,
        gt: torch.Tensor,
        thresholds=(0.3, 0.5, 0.7),
    ) -> Dict[float, torch.Tensor]:
        metrics = {}
        gt_bin = (gt > 0.5).float()
        for thr in thresholds:
            pr = (probs >= thr).float()
            inter = (pr * gt_bin).sum(dim=(1, 2, 3, 4))
            union = pr.sum(dim=(1, 2, 3, 4)) + gt_bin.sum(dim=(1, 2, 3, 4)) - inter
            metrics[thr] = (inter / (union + 1e-6)).mean()
        return metrics
    def evaluate_split(
        self,
        split: str,
        threshold: Optional[float] = None,
        max_samples: Optional[int] = None,
        save_json: bool = True,
    ) -> Dict[str, float]:
        eval_thr = self.eval_threshold if threshold is None else threshold
        use_obj_id_filter = getattr(self.args, "use_obj_id_filter", False)
        dataset = self._get_dataset(split=split, use_obj_id_filter=use_obj_id_filter)

        total_tp = total_fp = total_fn = 0.0
        total_iou_03 = 0.0
        total_iou_05 = 0.0
        total_iou_07 = 0.0

        total_mse = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        photometric_count = 0
        processed = 0

        per_sample = []

        eval_other_views = getattr(self.args, "eval_other_views", False)
        eval_other_views_num = int(getattr(self.args, "eval_other_views_num", 3))
        use_obj_id_filter = getattr(self.args, "use_obj_id_filter", False)

        for item in dataset.data_items:
            if max_samples is not None and processed >= max_samples:
                break

            scan_id = item.get("scan_id")
            frame_idx = item.get("frame_idx")
            if scan_id is None or frame_idx is None:
                continue

            fid = str(frame_idx).zfill(6)

            try:
                reconstruction, prepared, logits = self._predict_reconstruction(
                    scan_id,
                    fid,
                    split=split,
                    save_debug=False,
                    use_obj_id_filter=use_obj_id_filter,
                )
            except FileNotFoundError as err:
                LOGGER.warning("Skipping %s/%s: %s", scan_id, fid, err)
                continue
            except Exception as err:
                LOGGER.warning("Failed on %s/%s: %s", scan_id, fid, err)
                continue

            # ------------------------------------------------------------
            # voxel metrics in the GT-aligned grid, same convention as training
            # ------------------------------------------------------------
            probs = torch.sigmoid(logits)

            voxel_metrics = self._compute_prf_metrics(
                probs,
                prepared["occ_gt"],
                eval_thr,
            )
            iou_metrics = self._compute_iou(
                probs,
                prepared["occ_gt"],
                thresholds=(0.3, 0.5, 0.7),
            )

            total_tp += voxel_metrics["tp"]
            total_fp += voxel_metrics["fp"]
            total_fn += voxel_metrics["fn"]

            total_iou_03 += float(iou_metrics[0.3].item())
            total_iou_05 += float(iou_metrics[0.5].item())
            total_iou_07 += float(iou_metrics[0.7].item())

            if getattr(self.args, "save_eval_vox_debug", False):
                pred_idx = (probs[0, 0] >= eval_thr).nonzero(as_tuple=False)
                gt_idx = prepared["gt_idx"]
                frame_tag = str(fid).zfill(6)
                save_vox_as_ply(
                    pred_idx.detach().cpu(),
                    self.grid_res,
                    osp.join(self.output_dir, f"{scan_id}_{frame_tag}_eval_pred_occ.ply"),
                )
                save_vox_as_ply(
                    gt_idx.detach().cpu(),
                    self.grid_res,
                    osp.join(self.output_dir, f"{scan_id}_{frame_tag}_eval_gt_occ.ply"),
                )

            # ------------------------------------------------------------
            # photometric metrics in world/render space
            # ------------------------------------------------------------
            render_frame_id = prepared["meta"].get("frame_id_used", fid)
            R_frame = self._normalize_rotation(prepared["meta"].get("R_frame"))
            if getattr(self.args, "invert_pack_rotation", False):
                R_frame = R_frame.T

            photo_entries = []

            try:
                gt_img, pred_img = self._render_prediction(
                    reconstruction[0], scan_id, render_frame_id, R_frame
                )
                photo_metrics = self._compute_photometric_metrics(pred_img, gt_img)

                total_mse += photo_metrics["mse"]
                total_ssim += photo_metrics["ssim"]
                total_lpips += photo_metrics["lpips"]
                photometric_count += 1

                photo_entries.append(
                    {
                        "frame_id": render_frame_id,
                        "kind": "input_view",
                        "mse": photo_metrics["mse"],
                        "psnr": 10.0 * math.log10(1.0 / max(photo_metrics["mse"], 1e-8)),
                        "ssim": photo_metrics["ssim"],
                        "lpips": photo_metrics["lpips"],
                    }
                )
            except Exception as err:
                LOGGER.warning(
                    "Photometric eval failed on input view %s/%s: %s",
                    scan_id,
                    render_frame_id,
                    err,
                )

            if eval_other_views and eval_other_views_num > 0:
                other_fids = self._sample_other_scene_frames(
                    scan_id,
                    render_frame_id,
                    split,
                    eval_other_views_num,
                )

                for other_fid in other_fids:
                    try:
                        gt_img, pred_img = self._render_prediction(
                            reconstruction[0], scan_id, other_fid, R_frame
                        )
                        photo_metrics = self._compute_photometric_metrics(pred_img, gt_img)

                        total_mse += photo_metrics["mse"]
                        total_ssim += photo_metrics["ssim"]
                        total_lpips += photo_metrics["lpips"]
                        photometric_count += 1

                        photo_entries.append(
                            {
                                "frame_id": other_fid,
                                "kind": "other_view",
                                "mse": photo_metrics["mse"],
                                "psnr": 10.0 * math.log10(1.0 / max(photo_metrics["mse"], 1e-8)),
                                "ssim": photo_metrics["ssim"],
                                "lpips": photo_metrics["lpips"],
                            }
                        )
                    except Exception as err:
                        LOGGER.warning(
                            "Photometric eval failed on extra view %s/%s: %s",
                            scan_id,
                            other_fid,
                            err,
                        )

            processed += 1

            sample_record = {
                "scene_id": scan_id,
                "frame_id": fid,
                "render_frame_id": render_frame_id,
                "precision": voxel_metrics["precision"],
                "recall": voxel_metrics["recall"],
                "fscore": voxel_metrics["fscore"],
                "iou_03": float(iou_metrics[0.3].item()),
                "iou_05": float(iou_metrics[0.5].item()),
                "iou_07": float(iou_metrics[0.7].item()),
                "photometric_views": photo_entries,
            }

            input_views = [x for x in photo_entries if x["kind"] == "input_view"]
            if input_views:
                sample_record["psnr"] = input_views[0]["psnr"]
                sample_record["ssim"] = input_views[0]["ssim"]
                sample_record["lpips"] = input_views[0]["lpips"]

            per_sample.append(sample_record)

            if photo_entries:
                mean_sample_psnr = sum(x["psnr"] for x in photo_entries) / len(photo_entries)
                mean_sample_ssim = sum(x["ssim"] for x in photo_entries) / len(photo_entries)
                mean_sample_lpips = sum(x["lpips"] for x in photo_entries) / len(photo_entries)
                LOGGER.info(
                    "[%s %d] scene=%s frame=%s | F-score=%.4f | IoU@0.3=%.4f IoU@0.5=%.4f IoU@0.7=%.4f | PSNR=%.3f SSIM=%.4f LPIPS=%.4f over %d rendered views",
                    split,
                    processed,
                    scan_id,
                    fid,
                    voxel_metrics["fscore"],
                    float(iou_metrics[0.3].item()),
                    float(iou_metrics[0.5].item()),
                    float(iou_metrics[0.7].item()),
                    mean_sample_psnr,
                    mean_sample_ssim,
                    mean_sample_lpips,
                    len(photo_entries),
                )
            else:
                LOGGER.info(
                    "[%s %d] scene=%s frame=%s | F-score=%.4f | IoU@0.3=%.4f IoU@0.5=%.4f IoU@0.7=%.4f | no photometric views scored",
                    split,
                    processed,
                    scan_id,
                    fid,
                    voxel_metrics["fscore"],
                    float(iou_metrics[0.3].item()),
                    float(iou_metrics[0.5].item()),
                    float(iou_metrics[0.7].item()),
                )

        if processed == 0:
            raise RuntimeError(f"No samples evaluated for split {split}")

        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        fscore = 2 * precision * recall / (precision + recall + 1e-6)

        mean_iou_03 = total_iou_03 / processed
        mean_iou_05 = total_iou_05 / processed
        mean_iou_07 = total_iou_07 / processed

        if photometric_count > 0:
            mean_mse = total_mse / photometric_count
            mean_psnr = 10.0 * math.log10(1.0 / max(mean_mse, 1e-8))
            mean_ssim = total_ssim / photometric_count
            mean_lpips = total_lpips / photometric_count
        else:
            mean_psnr = 0.0
            mean_ssim = 0.0
            mean_lpips = 0.0

        results = {
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
            "iou_03": mean_iou_03,
            "iou_05": mean_iou_05,
            "iou_07": mean_iou_07,
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "lpips": mean_lpips,
            "samples": processed,
            "photometric_views": photometric_count,
        }

        LOGGER.info(
            "[%s split] voxel precision=%.4f recall=%.4f F-score=%.4f | IoU@0.3=%.4f IoU@0.5=%.4f IoU@0.7=%.4f | PSNR=%.3f SSIM=%.4f LPIPS=%.4f over %d samples and %d rendered views",
            split,
            precision,
            recall,
            fscore,
            mean_iou_03,
            mean_iou_05,
            mean_iou_07,
            mean_psnr,
            mean_ssim,
            mean_lpips,
            processed,
            photometric_count,
        )

        if save_json:
            # suffix = "_other_frame" if eval_other_views else ""
            suffix = "_with_photo_89"
            out_path = osp.join(self.output_dir, f"eval_{split}_metrics{suffix}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary": results,
                        "per_sample": per_sample,
                    },
                    f,
                    indent=2,
                )
            LOGGER.info("Saved evaluation metrics to %s", out_path)

        return results
    def _sample_other_scene_frames(
        self,
        scene_id: str,
        current_frame_id: str,
        split: str,
        num_views: int,
    ) -> list[str]:
        dataset = self._get_dataset(split=split)

        frame_ids = []
        current_frame_id = str(current_frame_id).zfill(6)

        for item in dataset.data_items:
            sid = item.get("scan_id")
            fid = item.get("frame_idx")
            if sid != scene_id or fid is None:
                continue
            fid = str(fid).zfill(6)
            if fid != current_frame_id:
                frame_ids.append(fid)

        frame_ids = sorted(set(frame_ids))
        if len(frame_ids) <= num_views:
            return frame_ids

        rng = np.random.default_rng(0)
        chosen = rng.choice(frame_ids, size=num_views, replace=False)
        return sorted([str(x) for x in chosen])

    # def evaluate_split(
    #     self,
    #     split: str,
    #     threshold: Optional[float] = None,
    #     max_samples: Optional[int] = None,
    #     save_json: bool = True,
    # ) -> Dict[str, float]:
    #     eval_thr = self.eval_threshold if threshold is None else threshold
    #     dataset = self._get_dataset(split=split)

    #     total_tp = total_fp = total_fn = 0.0
    #     total_mse = 0.0
    #     total_ssim = 0.0
    #     total_lpips = 0.0
    #     photometric_count = 0
    #     processed = 0

    #     per_sample = []

    #     eval_other_views = getattr(self.args, "eval_other_views", False)
    #     eval_other_views_num = int(getattr(self.args, "eval_other_views_num", 3))

    #     for item in dataset.data_items:
    #         if max_samples is not None and processed >= max_samples:
    #             break

    #         scan_id = item.get("scan_id")
    #         frame_idx = item.get("frame_idx")
    #         if scan_id is None or frame_idx is None:
    #             continue

    #         fid = str(frame_idx).zfill(6)

    #         try:
    #             reconstruction, prepared, logits = self._predict_reconstruction(
    #                 scan_id, fid, split=split
    #             )
    #         except FileNotFoundError as err:
    #             LOGGER.warning("Skipping %s/%s: %s", scan_id, fid, err)
    #             continue
    #         except Exception as err:
    #             LOGGER.warning("Failed on %s/%s: %s", scan_id, fid, err)
    #             continue

    #         # voxel metrics on the original sample only
    #         probs = torch.sigmoid(logits)
    #         voxel_metrics = self._compute_prf_metrics(
    #             probs, prepared["occ_gt"], eval_thr
    #         )

    #         total_tp += voxel_metrics["tp"]
    #         total_fp += voxel_metrics["fp"]
    #         total_fn += voxel_metrics["fn"]

    #         render_frame_id = prepared["meta"].get("frame_id_used", fid)
    #         R_frame = self._normalize_rotation(prepared["meta"].get("R_frame"))
    #         if getattr(self.args, "invert_pack_rotation", False):
    #             R_frame = R_frame.T

    #         photo_entries = []

    #         # original frame photometric metrics
    #         try:
    #             gt_img, pred_img = self._render_prediction(
    #                 reconstruction[0], scan_id, render_frame_id, R_frame
    #             )
    #             photo_metrics = self._compute_photometric_metrics(pred_img, gt_img)

    #             total_mse += photo_metrics["mse"]
    #             total_ssim += photo_metrics["ssim"]
    #             total_lpips += photo_metrics["lpips"]
    #             photometric_count += 1

    #             photo_entries.append(
    #                 {
    #                     "frame_id": render_frame_id,
    #                     "kind": "input_view",
    #                     "mse": photo_metrics["mse"],
    #                     "psnr": 10.0 * math.log10(1.0 / max(photo_metrics["mse"], 1e-8)),
    #                     "ssim": photo_metrics["ssim"],
    #                     "lpips": photo_metrics["lpips"],
    #                 }
    #             )
    #         except Exception as err:
    #             LOGGER.warning(
    #                 "Photometric eval failed on input view %s/%s: %s",
    #                 scan_id,
    #                 render_frame_id,
    #                 err,
    #             )

    #         # extra random views from same scene
    #         if eval_other_views and eval_other_views_num > 0:
    #             other_fids = self._sample_other_scene_frames(
    #                 scan_id,
    #                 render_frame_id,
    #                 split,
    #                 eval_other_views_num,
    #             )

    #             for other_fid in other_fids:
    #                 try:
    #                     gt_img, pred_img = self._render_prediction(
    #                         reconstruction[0], scan_id, other_fid, R_frame
    #                     )
    #                     photo_metrics = self._compute_photometric_metrics(pred_img, gt_img)

    #                     total_mse += photo_metrics["mse"]
    #                     total_ssim += photo_metrics["ssim"]
    #                     total_lpips += photo_metrics["lpips"]
    #                     photometric_count += 1

    #                     photo_entries.append(
    #                         {
    #                             "frame_id": other_fid,
    #                             "kind": "other_view",
    #                             "mse": photo_metrics["mse"],
    #                             "psnr": 10.0 * math.log10(1.0 / max(photo_metrics["mse"], 1e-8)),
    #                             "ssim": photo_metrics["ssim"],
    #                             "lpips": photo_metrics["lpips"],
    #                         }
    #                     )
    #                 except Exception as err:
    #                     LOGGER.warning(
    #                         "Photometric eval failed on extra view %s/%s: %s",
    #                         scan_id,
    #                         other_fid,
    #                         err,
    #                     )

    #         processed += 1

    #         sample_record = {
    #             "scene_id": scan_id,
    #             "frame_id": fid,
    #             "render_frame_id": render_frame_id,
    #             "precision": voxel_metrics["precision"],
    #             "recall": voxel_metrics["recall"],
    #             "fscore": voxel_metrics["fscore"],
    #             "photometric_views": photo_entries,
    #         }

    #         # convenience summary for the input view if present
    #         input_views = [x for x in photo_entries if x["kind"] == "input_view"]
    #         if input_views:
    #             sample_record["psnr"] = input_views[0]["psnr"]
    #             sample_record["ssim"] = input_views[0]["ssim"]
    #             sample_record["lpips"] = input_views[0]["lpips"]

    #         per_sample.append(sample_record)

    #         if photo_entries:
    #             mean_sample_psnr = sum(x["psnr"] for x in photo_entries) / len(photo_entries)
    #             mean_sample_ssim = sum(x["ssim"] for x in photo_entries) / len(photo_entries)
    #             mean_sample_lpips = sum(x["lpips"] for x in photo_entries) / len(photo_entries)
    #             LOGGER.info(
    #                 "[%s %d] scene=%s frame=%s | F-score=%.4f | PSNR=%.3f SSIM=%.4f LPIPS=%.4f over %d rendered views",
    #                 split,
    #                 processed,
    #                 scan_id,
    #                 fid,
    #                 voxel_metrics["fscore"],
    #                 mean_sample_psnr,
    #                 mean_sample_ssim,
    #                 mean_sample_lpips,
    #                 len(photo_entries),
    #             )
    #         else:
    #             LOGGER.info(
    #                 "[%s %d] scene=%s frame=%s | F-score=%.4f | no photometric views scored",
    #                 split,
    #                 processed,
    #                 scan_id,
    #                 fid,
    #                 voxel_metrics["fscore"],
    #             )

    #     if processed == 0:
    #         raise RuntimeError(f"No samples evaluated for split {split}")

    #     precision = total_tp / (total_tp + total_fp + 1e-6)
    #     recall = total_tp / (total_tp + total_fn + 1e-6)
    #     fscore = 2 * precision * recall / (precision + recall + 1e-6)

    #     if photometric_count > 0:
    #         mean_mse = total_mse / photometric_count
    #         mean_psnr = 10.0 * math.log10(1.0 / max(mean_mse, 1e-8))
    #         mean_ssim = total_ssim / photometric_count
    #         mean_lpips = total_lpips / photometric_count
    #     else:
    #         mean_psnr = 0.0
    #         mean_ssim = 0.0
    #         mean_lpips = 0.0

    #     results = {
    #         "precision": precision,
    #         "recall": recall,
    #         "fscore": fscore,
    #         "psnr": mean_psnr,
    #         "ssim": mean_ssim,
    #         "lpips": mean_lpips,
    #         "samples": processed,
    #         "photometric_views": photometric_count,
    #     }

    #     LOGGER.info(
    #         "[%s split] precision=%.4f recall=%.4f F-score=%.4f | PSNR=%.3f SSIM=%.4f LPIPS=%.4f over %d samples and %d rendered views",
    #         split,
    #         precision,
    #         recall,
    #         fscore,
    #         mean_psnr,
    #         mean_ssim,
    #         mean_lpips,
    #         processed,
    #         photometric_count,
    #     )

    #     if save_json:
    #         out_path = osp.join(self.output_dir, f"eval_{split}_metrics_other_frame.json")
    #         with open(out_path, "w", encoding="utf-8") as f:
    #             json.dump(
    #                 {
    #                     "summary": results,
    #                     "per_sample": per_sample,
    #                 },
    #                 f,
    #                 indent=2,
    #             )
    #         LOGGER.info("Saved evaluation metrics to %s", out_path)

    #     return results

    def run(self, scene_id: str, frame_id: str, split: str = "train") -> None:
        reconstruction, prepared, _ = self._predict_reconstruction(scene_id, frame_id, split=split, save_debug=True)

        frame_str = str(frame_id).zfill(6)
        if reconstruction:
            for idx, gauss in enumerate(reconstruction):
                suffix = f"_b{idx}" if len(reconstruction) > 1 else ""
                ply_path = osp.join(
                    self.output_dir,
                    f"{scene_id}_{frame_str}_pred{suffix}.ply",
                )
                gauss.save_ply(ply_path)
                LOGGER.info("Saved Gaussian PLY to %s", ply_path)

        render_frame_id = prepared["meta"].get("frame_id_used", frame_str)
        R_frame = self._normalize_rotation(prepared["meta"].get("R_frame"))
        if getattr(self.args, "invert_pack_rotation", False):
            R_frame = R_frame.T

        gt_img, pred_img = self._render_prediction(
            reconstruction[0], scene_id, render_frame_id, R_frame
        )

        pred_path = osp.join(self.output_dir, f"{scene_id}_{render_frame_id}_pred.png")
        gt_path = osp.join(self.output_dir, f"{scene_id}_{render_frame_id}_gt.png")
        compare_path = osp.join(
            self.output_dir,
            f"{scene_id}_{render_frame_id}_compare.png",
        )

        save_image(pred_img.clamp(0, 1).unsqueeze(0).cpu(), pred_path)
        save_image(gt_img.clamp(0, 1).unsqueeze(0).cpu(), gt_path)

        comp = torch.stack([gt_img.clamp(0, 1), pred_img.clamp(0, 1)], dim=0)
        save_image(comp.cpu(), compare_path, nrow=2)

        LOGGER.info("Saved rendered prediction to %s", pred_path)


def parse_args() -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run SLAT completion inference aligned with training pipeline")
    parser.add_argument("--config", required=True, type=str, help="Path to config YAML")
    parser.add_argument("--unet_checkpoint", required=True, type=str, help="Trained SLAT UNet checkpoint")
    parser.add_argument("--teacher_checkpoint", required=True, type=str, help="Latent autoencoder checkpoint")
    parser.add_argument("--scene_id", type=str, default=None, help="Scene ID to process")
    parser.add_argument("--frame_id", type=str, default=None, help="Frame ID (e.g. 000123)")
    parser.add_argument("--occ_threshold", type=float, default=0.3, help="Occupancy threshold for selecting voxels")
    parser.add_argument("--max_latents", type=int, default=20000, help="Maximum voxels passed to decoder")
    parser.add_argument("--output_dir", type=str, default="outputs/slat_completion", help="Directory for outputs")
    parser.add_argument(
        "--eval_other_views",
        action="store_true",
        help="If enabled, also evaluate photometric metrics on other random frames from the same scene.",
    )
    parser.add_argument(
        "--eval_other_views_num",
        type=int,
        default=3,
        help="Number of other random scene frames to evaluate when --eval_other_views is enabled.",
    )
    parser.add_argument(
        "--invert_pack_rotation",
        action="store_true",
        help="Render using inverse of pack rotation",
    )
    parser.add_argument(
        "--pose_is_w2c",
        action="store_true",
        help="Interpret loaded poses as world-to-camera when applying pack rotation",
    )
    parser.add_argument("--eval_split", type=str, default=None, help="Evaluate split instead of single-scene inference")
    parser.add_argument("--eval_threshold", type=float, default=0.5, help="Threshold for eval precision/recall/F-score")
    parser.add_argument("--max_eval", type=int, default=None, help="Optional max samples for eval")
    parser.add_argument("--student_pack_root", type=str, default=None, help="Root directory for student packs")
    parser.add_argument("--render_views", type=int, default=0, help="Number of GT views to render")
    parser.add_argument("--render_from_gt_annotations", action="store_true", help="Use GT annotation render mode")
    parser.add_argument("--save_multiview_dir", type=str, default=None, help="Optional multiview output dir")
    parser.add_argument(
        "--use_obj_id_filter",
        action="store_true",
        default=False,
        help="Evaluate with obj-id filter: GT voxels restricted to objects visible in the input frame (matches training with use_obj_id_filter=true)",
    )
    args, unknown = parser.parse_known_args()
    
    return args, unknown


def main() -> None:
    common.init_log(level=logging.INFO)
    args, unknown = parse_args()
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)

    runner = SlatCompletionInference(cfg, args)

    if args.eval_split:
        runner.evaluate_split(args.eval_split, threshold=args.eval_threshold, max_samples=args.max_eval)
    else:
        if args.scene_id is None or args.frame_id is None:
            raise ValueError("--scene_id and --frame_id are required for single-scene inference")
        runner.run(args.scene_id, args.frame_id)


if __name__ == "__main__":
    main()