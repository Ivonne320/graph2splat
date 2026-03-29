import argparse
import json
import logging
import math
import os.path as osp
import random
from argparse import Namespace
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gaussian_renderer import render
from scene.cameras import MiniCam

from configs import Config, update_configs
from src.datasets import Scan3RSceneBatchDataset
from src.models.latent_autoencoder import LatentAutoencoder
from src.models.losses.reconstruction import LPIPS
from utils import common, scan3r, torch_util
from utils.graphics_utils import focal2fov
from utils.loss_utils import ssim

_LOGGER = logging.getLogger(__name__)


class PhotometricEvaluator:
    """Evaluate photometric metrics (PSNR, SSIM, LPIPS) for reconstructed scenes."""

    def __init__(
        self,
        cfg: Config,
        split: str,
        checkpoint: Optional[str] = None,
        max_frames_per_scene: Optional[int] = None,
        scene_ids: Optional[Iterable[str]] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        single_frame: bool = False,
        single_frame_views: int = 5,
        single_frame_frame_limit: Optional[int] = None,
        geom_threshold: float = 0.02,
        geom_max_points: int = 10000,
    ) -> None:
        self.cfg = cfg
        self.cfg.data.preload_slat = getattr(self.cfg.data, "preload_slat", False)
        self.split = split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint)
        self.model.eval()
        self.lpips = LPIPS(
            device="cuda" if self.device.type == "cuda" else "cpu"
        ).eval()
        self.max_frames_per_scene = max_frames_per_scene
        self.target_scene_ids = set(scene_ids) if scene_ids else None
        self.single_frame_mode = (
            single_frame
            if single_frame is not None
            else getattr(self.cfg.data, "scene_level_single_frame", False)
        )
        setattr(self.cfg.data, "scene_level_single_frame", self.single_frame_mode)
        self.single_frame_views = single_frame_views
        cfg_limit = getattr(self.cfg.data, "single_frame_frame_limit", None)
        self.single_frame_frame_limit = (
            single_frame_frame_limit if single_frame_frame_limit is not None else cfg_limit
        )
        if self.single_frame_frame_limit is None:
            self.single_frame_frame_limit = 50
        self._single_frame_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.rng = random.Random(getattr(self.cfg, "seed", 0))
        self.np_rng = np.random.default_rng(getattr(self.cfg, "seed", 0))
        self.geom_threshold = geom_threshold
        self.geom_max_points = geom_max_points
        self.gt_points_cache: Dict[str, np.ndarray] = {}

        val_cfg = getattr(self.cfg, "val", None)
        default_batch_size = getattr(val_cfg, "batch_size", 1) if val_cfg else 1
        default_num_workers = getattr(val_cfg, "num_workers", 0) if val_cfg else 0

        self.dataset = Scan3RSceneBatchDataset(self.cfg, split=self.split)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size or default_batch_size,
            shuffle=False,
            num_workers=num_workers if num_workers is not None else default_num_workers,
            collate_fn=self.dataset.collate_fn,
            pin_memory=True,
            drop_last=False,
        )

        self.pipe_cfg = Namespace(
            debug=False, compute_cov3D_python=False, convert_SHs_python=False
        )

    def _load_model(self, checkpoint: Optional[str]) -> LatentAutoencoder:
        model = LatentAutoencoder(cfg=self.cfg.autoencoder, device=self.device)

        if checkpoint is None:
            inference_cfg = getattr(self.cfg, "inference", None)
            checkpoint = getattr(inference_cfg, "slat_model_path", None)

        if checkpoint is None:
            raise ValueError(
                "No checkpoint provided. Specify --checkpoint or set cfg.inference.slat_model_path."
            )

        ckpt = torch.load(checkpoint, map_location=self.device)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict)
        return model

    def run(self) -> Dict[str, Dict[str, float]]:
        per_scene_acc = defaultdict(lambda: {"mse": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0})
        overall_acc = {"mse": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0}
        per_frame_acc: Dict[str, Dict[str, float]] = {}
        geom_scene_acc = defaultdict(
            lambda: {"chamfer": 0.0, "precision": 0.0, "recall": 0.0, "fscore": 0.0, "count": 0}
        )
        geom_overall_acc = {"chamfer": 0.0, "precision": 0.0, "recall": 0.0, "fscore": 0.0, "count": 0}
        geom_per_frame_acc: Dict[str, Dict[str, float]] = {}

        with torch.no_grad():
            for batch in self.dataloader:
                if not batch:
                    continue
                if self.single_frame_mode:
                    self._evaluate_single_frame_batch(
                        batch,
                        per_scene_acc,
                        overall_acc,
                        per_frame_acc,
                        geom_scene_acc,
                        geom_overall_acc,
                        geom_per_frame_acc,
                    )
                else:
                    batch = self._move_to_device(batch)
                    embedding = self.model.encode(batch)
                    reconstruction = self.model.decode(embedding)
                    self._evaluate_batch(
                        batch,
                        reconstruction,
                        per_scene_acc,
                        overall_acc,
                        geom_scene_acc,
                        geom_overall_acc,
                    )

        results = self._finalize_metrics(
            per_scene_acc,
            overall_acc,
            per_frame_acc,
            geom_scene_acc,
            geom_overall_acc,
            geom_per_frame_acc,
        )
        return results

    def _move_to_device(self, data_dict: dict) -> dict:
        if self.device.type == "cuda":
            return torch_util.to_cuda(data_dict)
        return data_dict

    def _evaluate_batch(
        self,
        data_dict: dict,
        reconstruction,
        per_scene_acc: dict,
        overall_acc: dict,
        geom_scene_acc: Optional[dict] = None,
        geom_overall_acc: Optional[dict] = None,
    ) -> None:
        scene_graphs = data_dict["scene_graphs"]
        raw_scene_ids = scene_graphs["scene_ids"]
        scene_ids = [self._normalize_scene_id(sid) for sid in raw_scene_ids]

        translations = scene_graphs["mean_obj_splat"]
        scales = scene_graphs["scale_obj_splat"]
        R_cans = scene_graphs["R_cans"]
        image_frames = scene_graphs["image_frames"]
        intrinsics = scene_graphs["obj_intrinsics"]

        for idx, sid in enumerate(scene_ids):
            if self.target_scene_ids and sid not in self.target_scene_ids:
                continue

            frames = image_frames.get(sid, [])
            if not frames:
                continue
            if self.max_frames_per_scene is not None:
                frames = frames[: self.max_frames_per_scene]

            recon = reconstruction[idx]
            translation = translations[idx]
            scale = scales[idx]
            self._apply_scene_alignment(recon, translation, scale)

            if geom_scene_acc is not None and geom_overall_acc is not None:
                geom_metrics = self._compute_geometry_metrics(recon, sid)
                if geom_metrics is not None:
                    self._accumulate_geometry(geom_scene_acc[sid], geom_metrics)
                    self._accumulate_geometry(geom_overall_acc, geom_metrics)

            R_can = R_cans[idx].detach().cpu().numpy()
            extrinsics_frames = scan3r.load_frame_poses(
                self.cfg.data.root_dir, sid, tuple(frames)
            )

            intr = intrinsics[sid]
            H, W = int(intr["height"]), int(intr["width"])
            fx, fy = intr["intrinsic_mat"][0, 0], intr["intrinsic_mat"][1, 1]
            fovx = focal2fov(fx, W)
            fovy = focal2fov(fy, H)

            R_h = np.eye(4, dtype=np.float32)
            R_h[:3, :3] = R_can

            for fid in frames:
                metrics = self._evaluate_frame(
                    sid, fid, recon, extrinsics_frames[fid], R_h, fovx, fovy, W, H
                )

                if metrics is None:
                    continue

                mse_val, ssim_val, lpips_val = metrics
                self._accumulate(per_scene_acc[sid], mse_val, ssim_val, lpips_val)
                self._accumulate(overall_acc, mse_val, ssim_val, lpips_val)

    def _get_scene_frame_ids(self, scene_id: str) -> List[str]:
        frame_ids = scan3r.load_frame_idxs(self.dataset.scenes_dir, scene_id)
        if (
            self.single_frame_frame_limit is not None
            and len(frame_ids) > self.single_frame_frame_limit
        ):
            frame_ids = frame_ids[: self.single_frame_frame_limit]
        return frame_ids

    def _load_single_frame_scene_data(self, scene_id: str) -> List[Dict[str, Any]]:
        if scene_id in self._single_frame_cache:
            return self._single_frame_cache[scene_id]

        try:
            splat_dict = self.dataset._load_splats_single_frame(
                scene_id, preload_slat=self.cfg.data.preload_slat
            )
        except FileNotFoundError:
            _LOGGER.warning("Missing single-frame splats for scene %s", scene_id)
            self._single_frame_cache[scene_id] = []
            return []

        mean = splat_dict["mean_obj_splat"]
        scale = splat_dict["scale_obj_splat"]
        R_cans = splat_dict["R_cans"]
        splat = splat_dict["tot_obj_splat"]

        if isinstance(mean, torch.Tensor) and mean.dim() == 1:
            mean = mean.unsqueeze(0)
        if isinstance(scale, torch.Tensor) and scale.dim() == 0:
            scale = scale.unsqueeze(0)

        frame_ids = self._get_scene_frame_ids(scene_id)
        frame_count = min(
            len(frame_ids),
            mean.shape[0] if isinstance(mean, torch.Tensor) else len(mean),
            scale.shape[0] if isinstance(scale, torch.Tensor) else len(scale),
            R_cans.shape[0] if isinstance(R_cans, torch.Tensor) else len(R_cans),
            splat.shape[0],
        )

        frames: List[Dict[str, Any]] = []
        for idx in range(frame_count):
            fid = frame_ids[idx]
            frames.append(
                {
                    "scene_id": scene_id,
                    "fid": fid,
                    "splat": splat[idx].cpu(),
                    "mean": mean[idx].clone().cpu(),
                    "scale": scale[idx].clone().cpu(),
                    "R_can": R_cans[idx].clone().cpu(),
                }
            )

        self._single_frame_cache[scene_id] = frames
        return frames

    def _reconstruct_single_frame(
        self, frame_info: Dict[str, Any]
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_splat = frame_info["splat"].to(self.device)
        mean = frame_info["mean"].to(self.device)
        scale = frame_info["scale"].to(self.device)
        R_can = frame_info["R_can"].to(self.device)

        if scale.dim() == 0:
            scale_tensor = scale.view(1, 1)
        elif scale.dim() == 1:
            scale_tensor = scale.unsqueeze(0)
        else:
            scale_tensor = scale

        scene_graph = {
            "tot_obj_splat": frame_splat,
            "mean_obj_splat": mean.unsqueeze(0),
            "scale_obj_splat": scale_tensor,
            "R_cans": R_can.unsqueeze(0),
        }
        data_dict = {"scene_graphs": scene_graph}
        embedding = self.model.encode(data_dict)
        reconstruction = self.model.decode(embedding)[0]
        return reconstruction, mean, scale, R_can

    def _evaluate_single_frame_batch(
        self,
        data_dict: dict,
        per_scene_acc: dict,
        overall_acc: dict,
        per_frame_acc: Dict[str, Dict[str, float]],
        geom_scene_acc: dict,
        geom_overall_acc: dict,
        geom_per_frame_acc: Dict[str, Dict[str, float]],
    ) -> None:
        scene_graphs = data_dict["scene_graphs"]
        raw_scene_ids = scene_graphs["scene_ids"]
        scene_ids = [self._normalize_scene_id(sid) for sid in raw_scene_ids]
        intrinsics_map = scene_graphs["obj_intrinsics"]

        for sid in scene_ids:
            if self.target_scene_ids and sid not in self.target_scene_ids:
                continue

            frame_infos = self._load_single_frame_scene_data(sid)
            if not frame_infos:
                continue

            intrinsics = intrinsics_map[sid]
            frame_ids_all = self._get_scene_frame_ids(sid)
            if not frame_ids_all:
                continue

            extrinsics_frames = scan3r.load_frame_poses(
                self.cfg.data.root_dir, sid, tuple(frame_ids_all)
            )

            H, W = int(intrinsics["height"]), int(intrinsics["width"])
            fx, fy = intrinsics["intrinsic_mat"][0, 0], intrinsics["intrinsic_mat"][1, 1]
            fovx = focal2fov(fx, W)
            fovy = focal2fov(fy, H)

            for frame_info in frame_infos:
                fid = frame_info["fid"]
                other_fids = [
                    f for f in frame_ids_all if f != fid and f in extrinsics_frames
                ]
                if not other_fids:
                    continue

                if len(other_fids) > self.single_frame_views:
                    view_fids = self.rng.sample(other_fids, self.single_frame_views)
                else:
                    view_fids = other_fids

                reconstruction, mean_dev, scale_dev, R_can_dev = self._reconstruct_single_frame(
                    frame_info
                )
                self._apply_scene_alignment(reconstruction, mean_dev, scale_dev)

                R_h = np.eye(4, dtype=np.float32)
                R_h[:3, :3] = R_can_dev.detach().cpu().numpy()

                frame_key = f"{sid}/{fid}"
                if frame_key not in per_frame_acc:
                    per_frame_acc[frame_key] = {
                        "mse": 0.0,
                        "ssim": 0.0,
                        "lpips": 0.0,
                        "count": 0,
                    }
                frame_geom_acc = geom_per_frame_acc.setdefault(
                    frame_key,
                    {"chamfer": 0.0, "precision": 0.0, "recall": 0.0, "fscore": 0.0, "count": 0},
                )

                geom_metrics = self._compute_geometry_metrics(reconstruction, sid)
                if geom_metrics is not None:
                    self._accumulate_geometry(geom_scene_acc[sid], geom_metrics)
                    self._accumulate_geometry(geom_overall_acc, geom_metrics)
                    self._accumulate_geometry(frame_geom_acc, geom_metrics)

                for view_fid in view_fids:
                    metrics = self._evaluate_frame(
                        sid,
                        view_fid,
                        reconstruction,
                        extrinsics_frames[view_fid],
                        R_h,
                        fovx,
                        fovy,
                        W,
                        H,
                    )
                    if metrics is None:
                        continue
                    mse_val, ssim_val, lpips_val = metrics
                    self._accumulate(per_scene_acc[sid], mse_val, ssim_val, lpips_val)
                    self._accumulate(overall_acc, mse_val, ssim_val, lpips_val)
                    self._accumulate(per_frame_acc[frame_key], mse_val, ssim_val, lpips_val)

    def _evaluate_frame(
        self,
        scene_id: str,
        frame_id: int,
        reconstruction,
        extrinsics: np.ndarray,
        R_h: np.ndarray,
        fovx: float,
        fovy: float,
        width: int,
        height: int,
    ) -> Optional[tuple[float, float, float]]:
        img_path = osp.join(
            self.cfg.data.root_dir,
            "scenes",
            scene_id,
            "sequence",
            f"frame-{frame_id}.color.jpg",
        )

        if not osp.isfile(img_path):
            _LOGGER.warning("Missing image for %s frame %s", scene_id, frame_id)
            return None

        with Image.open(img_path) as pil_image:
            image_tensor = torch.from_numpy(np.array(pil_image, copy=False))
        image = image_tensor.permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0).to(self.device)

        extrinsics = R_h @ extrinsics
        pose_camera_to_world = np.linalg.inv(extrinsics)
        viewpoint_camera = MiniCam(
            width=width,
            height=height,
            fovy=fovy,
            fovx=fovx,
            znear=0.01,
            zfar=100.0,
            R=pose_camera_to_world[:3, :3].T,
            T=pose_camera_to_world[:3, 3],
        )

        render_out = render(
            viewpoint_camera,
            reconstruction,
            pipe=self.pipe_cfg,
            bg_color=torch.tensor(
                (0.0, 0.0, 0.0), device=reconstruction.get_xyz.device
            ),
        )
        predicted = render_out["render"]
        if predicted.dim() == 3:
            predicted = predicted.unsqueeze(0)
        predicted = predicted.clamp(0.0, 1.0)

        mse_val = F.mse_loss(predicted, image, reduction="mean").item()
        ssim_val = ssim(predicted, image).item()
        lpips_val = self.lpips(predicted, image).item()
        return mse_val, ssim_val, lpips_val

    def _compute_geometry_metrics(
        self, reconstruction, scene_id: str
    ) -> Optional[Dict[str, float]]:
        pred_xyz = reconstruction.get_xyz
        if pred_xyz is None or pred_xyz.numel() == 0:
            return None
        pred_points = pred_xyz.detach().cpu().numpy()
        pred_points = self._sample_points_np(pred_points, self.geom_max_points)
        if pred_points.shape[0] == 0:
            return None

        gt_points = self._load_gt_points(scene_id)
        if gt_points.size == 0:
            return None

        pred_t = torch.from_numpy(pred_points).float()
        gt_t = torch.from_numpy(gt_points).float()

        d_pred_to_gt = self._min_distances(pred_t, gt_t)
        d_gt_to_pred = self._min_distances(gt_t, pred_t)

        chamfer = float(d_pred_to_gt.mean().item() + d_gt_to_pred.mean().item())

        precision = float(
            (d_pred_to_gt < self.geom_threshold).sum().item()
            / max(1, d_pred_to_gt.shape[0])
        )
        recall = float(
            (d_gt_to_pred < self.geom_threshold).sum().item()
            / max(1, d_gt_to_pred.shape[0])
        )
        if precision + recall == 0:
            fscore = 0.0
        else:
            fscore = 2 * precision * recall / (precision + recall)

        return {
            "chamfer": chamfer,
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
        }

    def _load_gt_points(self, scene_id: str) -> np.ndarray:
        if scene_id not in self.gt_points_cache:
            points = self.dataset.load_mesh(scene_id, obj_id=-1)
            if isinstance(points, torch.Tensor):
                points_np = points.cpu().numpy()
            else:
                points_np = np.asarray(points)
            points_np = self._sample_points_np(points_np, self.geom_max_points)
            self.gt_points_cache[scene_id] = points_np
        return self.gt_points_cache[scene_id]

    def _sample_points_np(self, points: np.ndarray, max_points: int) -> np.ndarray:
        if points.shape[0] == 0 or points.shape[0] <= max_points:
            return points.astype(np.float32, copy=False)
        idx = self.np_rng.choice(points.shape[0], size=max_points, replace=False)
        return points[idx].astype(np.float32, copy=False)

    @staticmethod
    def _min_distances(src: torch.Tensor, dst: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
        if src.numel() == 0 or dst.numel() == 0:
            return torch.empty(0)
        mins: List[torch.Tensor] = []
        for start in range(0, src.shape[0], chunk_size):
            end = min(start + chunk_size, src.shape[0])
            chunk = src[start:end]
            dist = torch.cdist(chunk, dst)
            mins.append(dist.min(dim=1).values)
        return torch.cat(mins, dim=0)

    @staticmethod
    def _accumulate_geometry(acc: Dict[str, float], metrics: Dict[str, float]) -> None:
        acc["chamfer"] += metrics["chamfer"]
        acc["precision"] += metrics["precision"]
        acc["recall"] += metrics["recall"]
        acc["fscore"] += metrics["fscore"]
        acc["count"] += 1

    def _apply_scene_alignment(self, reconstruction, translation, scale) -> None:
        device = reconstruction.get_xyz.device
        reconstruction.rescale(torch.tensor([2.0, 2.0, 2.0], device=device))
        reconstruction.translate(torch.tensor([-1.0, -1.0, -1.0], device=device))

        scale_vec = scale.flatten()
        if scale_vec.numel() == 1:
            scale_vec = scale_vec.repeat(3)
        reconstruction.rescale(scale_vec.to(device))

        translation_vec = translation.flatten()
        if translation_vec.numel() == 1:
            translation_vec = translation_vec.repeat(3)
        reconstruction.translate(translation_vec.to(device))

    def _accumulate(
        self, acc: Dict[str, float], mse_val: float, ssim_val: float, lpips_val: float
    ) -> None:
        acc["mse"] += mse_val
        acc["ssim"] += ssim_val
        acc["lpips"] += lpips_val
        acc["count"] += 1
    def _finalize_metrics(
        self,
        per_scene_acc: dict,
        overall_acc: dict,
        per_frame_acc: Optional[Dict[str, Dict[str, float]]] = None,
        geom_scene_acc: Optional[dict] = None,
        geom_overall_acc: Optional[dict] = None,
        geom_per_frame_acc: Optional[dict] = None,
    ) -> Dict[str, Dict[str, float]]:
        per_scene_results = {}
        for sid, acc in per_scene_acc.items():
            if acc["count"] == 0:
                continue
            mse_mean = acc["mse"] / acc["count"]
            psnr = 10.0 * math.log10(1.0 / max(mse_mean, 1e-8))
            per_scene_results[sid] = {
                "psnr": psnr,
                "ssim": acc["ssim"] / acc["count"],
                "lpips": acc["lpips"] / acc["count"],
                "frames": acc["count"],
            }

        if overall_acc["count"] > 0:
            mse_mean = overall_acc["mse"] / overall_acc["count"]
            overall_psnr = 10.0 * math.log10(1.0 / max(mse_mean, 1e-8))
            overall_metrics = {
                "psnr": overall_psnr,
                "ssim": overall_acc["ssim"] / overall_acc["count"],
                "lpips": overall_acc["lpips"] / overall_acc["count"],
                "frames": overall_acc["count"],
            }
        else:
            overall_metrics = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "frames": 0}

        per_frame_results = {}
        if per_frame_acc:
            for key, acc in per_frame_acc.items():
                if acc["count"] == 0:
                    continue
                mse_mean = acc["mse"] / acc["count"]
                per_frame_results[key] = {
                    "psnr": 10.0 * math.log10(1.0 / max(mse_mean, 1e-8)),
                    "ssim": acc["ssim"] / acc["count"],
                    "lpips": acc["lpips"] / acc["count"],
                    "frames": acc["count"],
                }

        results = {"per_scene": per_scene_results, "overall": overall_metrics}
        if per_frame_results:
            results["per_frame"] = per_frame_results

        geometry_results = self._finalize_geometry(
            geom_scene_acc, geom_overall_acc, geom_per_frame_acc
        )
        if geometry_results:
            results["geometry"] = geometry_results
        return results

    @staticmethod
    def _summarize_geometry_acc(acc: Dict[str, float]) -> Dict[str, float]:
        if acc["count"] == 0:
            return {
                "chamfer": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "fscore": 0.0,
                "count": 0,
            }
        inv = 1.0 / acc["count"]
        return {
            "chamfer": acc["chamfer"] * inv,
            "precision": acc["precision"] * inv,
            "recall": acc["recall"] * inv,
            "fscore": acc["fscore"] * inv,
            "count": acc["count"],
        }

    def _finalize_geometry(
        self,
        geom_scene_acc: Optional[dict],
        geom_overall_acc: Optional[dict],
        geom_per_frame_acc: Optional[dict],
    ) -> Dict[str, Dict[str, float]]:
        if geom_scene_acc is None or geom_overall_acc is None:
            return {}

        per_scene = {}
        for sid, acc in geom_scene_acc.items():
            if acc["count"] == 0:
                continue
            per_scene[sid] = self._summarize_geometry_acc(acc)

        overall = self._summarize_geometry_acc(geom_overall_acc)

        geometry_results: Dict[str, Dict[str, float]] = {
            "per_scene": per_scene,
            "overall": overall,
        }

        if overall["count"] == 0 and not per_scene:
            return {}

        if geom_per_frame_acc:
            per_frame = {}
            for key, acc in geom_per_frame_acc.items():
                if acc["count"] == 0:
                    continue
                per_frame[key] = self._summarize_geometry_acc(acc)
            if per_frame:
                geometry_results["per_frame"] = per_frame

        return geometry_results

    @staticmethod
    def _normalize_scene_id(scene_id) -> str:
        if isinstance(scene_id, str):
            return scene_id
        if isinstance(scene_id, (list, tuple)):
            return PhotometricEvaluator._normalize_scene_id(scene_id[0])
        if isinstance(scene_id, np.ndarray):
            return PhotometricEvaluator._normalize_scene_id(scene_id.item())
        if torch.is_tensor(scene_id):
            scene_id = scene_id.detach().cpu()
            if scene_id.numel() == 1:
                return str(scene_id.item())
            return PhotometricEvaluator._normalize_scene_id(scene_id[0])
        return str(scene_id)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate photometric metrics for scene reconstructions."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (defaults to cfg.inference.slat_model_path).",
    )
    parser.add_argument("--split", type=str, default="val", help="Dataset split.")
    parser.add_argument(
        "--scene_ids",
        type=str,
        nargs="+",
        default=None,
        help="Subset of scene ids to evaluate.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames per scene to evaluate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override evaluation dataloader batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override dataloader worker count.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    parser.add_argument(
        "--single_frame",
        action="store_true",
        help="Evaluate single-frame reconstructions (uses single-frame splats).",
    )
    parser.add_argument(
        "--single_frame_views",
        type=int,
        default=5,
        help="Number of alternate views to sample per frame when single-frame evaluation is enabled.",
    )
    parser.add_argument(
        "--single_frame_frame_limit",
        type=int,
        default=None,
        help="Optional limit for number of frames per scene when sampling single-frame data.",
    )
    parser.add_argument(
        "--geom_threshold",
        type=float,
        default=0.02,
        help="Distance threshold (meters) for geometry precision/recall/F-score.",
    )
    parser.add_argument(
        "--geom_max_points",
        type=int,
        default=10000,
        help="Maximum number of points sampled from prediction and GT for geometry metrics.",
    )
    return parser.parse_known_args()


def log_results(results: Dict[str, Dict[str, float]]) -> None:
    overall = results["overall"]
    _LOGGER.info(
        "Overall metrics — PSNR: %.3f, SSIM: %.4f, LPIPS: %.4f over %d frames",
        overall["psnr"],
        overall["ssim"],
        overall["lpips"],
        overall["frames"],
    )
    for sid, metrics in sorted(results["per_scene"].items()):
        _LOGGER.info(
            "Scene %s — PSNR: %.3f, SSIM: %.4f, LPIPS: %.4f (%d frames)",
            sid,
            metrics["psnr"],
            metrics["ssim"],
            metrics["lpips"],
            metrics["frames"],
        )

    per_frame = results.get("per_frame")
    if per_frame:
        total_views = sum(item["frames"] for item in per_frame.values())
        _LOGGER.info("Evaluated %d single-frame view pairs", total_views)

    geometry = results.get("geometry")
    if geometry:
        overall_geom = geometry.get("overall", {})
        _LOGGER.info(
            "Geometry — Chamfer: %.4f, Precision: %.4f, Recall: %.4f, F-score: %.4f (count=%d)",
            overall_geom.get("chamfer", 0.0),
            overall_geom.get("precision", 0.0),
            overall_geom.get("recall", 0.0),
            overall_geom.get("fscore", 0.0),
            overall_geom.get("count", 0),
        )
        for sid, metrics in sorted(geometry.get("per_scene", {}).items()):
            _LOGGER.info(
                "Geometry %s — Chamfer: %.4f, Precision: %.4f, Recall: %.4f, F-score: %.4f (count=%d)",
                sid,
                metrics.get("chamfer", 0.0),
                metrics.get("precision", 0.0),
                metrics.get("recall", 0.0),
                metrics.get("fscore", 0.0),
                metrics.get("count", 0),
            )

def main() -> None:
    common.init_log(level=logging.INFO)
    args, unknown_args = parse_args()
    cfg = update_configs(args.config, unknown_args, do_ensure_dir=False)
    evaluator = PhotometricEvaluator(
        cfg=cfg,
        split=args.split,
        checkpoint=args.checkpoint,
        max_frames_per_scene=args.max_frames,
        scene_ids=args.scene_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        single_frame=args.single_frame,
        single_frame_views=args.single_frame_views,
        single_frame_frame_limit=args.single_frame_frame_limit,
        geom_threshold=args.geom_threshold,
        geom_max_points=args.geom_max_points,
    )
    results = evaluator.run()
    log_results(results)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        _LOGGER.info("Saved metrics to %s", args.output)


if __name__ == "__main__":
    main()
