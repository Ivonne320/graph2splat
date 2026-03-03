import argparse
import copy
import logging
import os
import os.path as osp
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from configs import Config, update_configs
from gaussian_renderer import render
from scene.cameras import MiniCam
from src.models.latent_autoencoder import LatentAutoencoder
from src.models.unet3d_completion import UNetCompletionModel
from src.modules.sparse.basic import SparseTensor
from src.datasets import Scan3RSceneBatchDataset
from utils import common, scan3r, torch_util
from utils.graphics_utils import focal2fov


LOGGER = logging.getLogger(__name__)


class SlatCompletionInference:
    def __init__(self, cfg: Config, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_cfg = getattr(cfg.autoencoder, "encoder", None)
        self.grid_res = getattr(encoder_cfg, "resolution", 128)
        self.latent_dim = getattr(encoder_cfg, "latent_channels", 16)
        default_root = "/cluster/scratch/wangyih/3RScan"
        self.student_pack_root = args.student_pack_root or default_root
        subdir_cfg = getattr(cfg.data, "student_pack_subdir", "scene_level_structure_no_dilation_128")
        if isinstance(subdir_cfg, (list, tuple)):
            self.student_pack_subdirs = list(subdir_cfg)
        else:
            self.student_pack_subdirs = [subdir_cfg]
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

    @staticmethod
    def _idx_to_centers(idx: torch.Tensor, G: int) -> torch.Tensor:
        if idx.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32, device=idx.device if idx.is_cuda else None)
        return (idx.float() + 0.5) / G - 0.5

    @staticmethod
    def _centers_to_idx(centers: torch.Tensor, G: int) -> torch.Tensor:
        if centers.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.int32, device=centers.device if centers.is_cuda else None)
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
            C = feat_t.shape[-1] if feat_t.ndim == 2 else self.seed_feat_dim or 64
            # LOGGER.info(f"C:{C}")
            grid_feats = torch.zeros(1, C, G, G, G, device=self.device)
            seed_occ = torch.zeros(1, 1, G, G, G, device=self.device)
            return grid_feats, seed_occ
        M, C = feat_t.shape
        # LOGGER.info(f"C:{C}")
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

    def _load_student_pack(self, scene_id: str, frame_id: str) -> Tuple[Dict[str, np.ndarray], str]:
        fid = str(frame_id).zfill(6)
        for subdir in self.student_pack_subdirs:
            base = osp.join(
                self.student_pack_root,
                "files",
                "gs_annotations",
                scene_id,
                subdir,
            )
            candidate = osp.join(base, f"student_pack_aligned_{fid}.npz")
            if osp.isfile(candidate):
                pack_dict = self._load_pack_with_meta(candidate, base)
                return pack_dict, candidate
            if osp.isdir(base):
                alt = [f for f in os.listdir(base) if f.startswith("student_pack_aligned_")]
                if alt:
                    alt_path = osp.join(base, sorted(alt)[0])
                    LOGGER.warning(
                        "Exact frame %s not found under %s; using %s instead.",
                        fid,
                        base,
                        alt[0],
                    )
                    pack_dict = self._load_pack_with_meta(alt_path, base)
                    return pack_dict, alt_path
        raise FileNotFoundError(
            f"No student pack for scene {scene_id} frame {fid} in {self.student_pack_subdirs}"
        )

    def _load_pack_with_meta(self, pack_path: str, base_dir: str) -> Dict[str, np.ndarray]:
        data = np.load(pack_path, allow_pickle=False)
        pack = {k: data[k] for k in data.files}
        pack["__base__"] = base_dir
        pack["__path__"] = pack_path
        meta_path = osp.join(base_dir, "scene_occ_meta.npz")
        if osp.isfile(meta_path):
            meta = np.load(meta_path, allow_pickle=False)
            for key in ("vox_idx_gt_occ", "mean_gt", "scale_gt"):
                if key not in pack and key in meta.files:
                    pack[key] = meta[key]
        return pack

    def _init_teacher(self) -> None:
        if self.latent_model is not None:
            return
        model = LatentAutoencoder(cfg=self.cfg.autoencoder, device=self.device)
        checkpoint = self.args.teacher_checkpoint
        if checkpoint is None:
            raise ValueError("--teacher_checkpoint must be provided for decoding")
        state = torch.load(checkpoint, map_location=self.device)
        model.load_state_dict(state.get("model", state), strict=False)
        model.eval()
        inferred_dim = self._infer_teacher_latent_dim(model.encoder)
        if inferred_dim is not None and inferred_dim != self.latent_dim:
            LOGGER.warning(
                "Teacher latent dim %d overrides UNet config %d.",
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
        model = UNetCompletionModel(
            feat_in=feat_dim,
            out_channels=1 + self.latent_dim,
        ).to(self.device)
        checkpoint = self.args.unet_checkpoint
        if checkpoint is None:
            raise ValueError("--unet_checkpoint is required for inference")
        state = torch.load(checkpoint, map_location=self.device)
        model.load_state_dict(state.get("model", state), strict=False)
        model.eval()
        self.unet = model

    def _prepare_inputs(self, pack) -> Dict[str, Any]:
        if self.unet is None or self.seed_feat_dim is None:
            raise RuntimeError("UNet must be initialized before preparing inputs")
        G = int(pack["G"])
        # LOGGER.info(f"G:{G}")
        if G != self.grid_res:
            raise ValueError(f"Pack grid {G} does not match model resolution {self.grid_res}")

        gt_idx_np = pack.get("vox_idx_gt_occ", np.zeros((0, 3), np.int32)).astype(np.int64)
        gt_idx = torch.from_numpy(gt_idx_np)
        seed_idx_np = pack.get("seed_idx", np.zeros((0, 3), np.int32)).astype(np.int64)
        seed_idx_raw = torch.from_numpy(seed_idx_np).to(self.device)
        feats_np = pack.get(
            "feats", np.zeros((0, self.seed_feat_dim), np.float32)
        ).astype(np.float32)
        feats = torch.from_numpy(feats_np).to(self.device)
        mean_seed = torch.from_numpy(
            pack.get("seed_box_init_mean", np.zeros(3, np.float32)).astype(np.float32)
        ).to(self.device)
        scale_seed_val = float(pack.get("seed_box_init_scale", 1.0))
        scale_seed = torch.tensor(scale_seed_val, dtype=torch.float32, device=self.device)
        mean_gt = torch.from_numpy(pack.get("mean_gt", np.zeros(3, np.float32)).astype(np.float32)).to(self.device)
        scale_gt_np = np.asarray(pack.get("scale_gt", 1.0), dtype=np.float32)
        scale_gt = torch.from_numpy(scale_gt_np).view(-1).to(self.device)
        if scale_gt.numel() == 1:
            scale_gt = scale_gt.repeat(3)
        elif scale_gt.numel() > 3:
            scale_gt = scale_gt[:3]

        occ_gt = torch.zeros(1, 1, G, G, G, device=self.device)
        if gt_idx.numel():
            occ_gt[0, 0, gt_idx[:, 0], gt_idx[:, 1], gt_idx[:, 2]] = 1.0

        if feats.shape[0] == 0:
            comp = torch.zeros(0, self.unet.compressed_channels, device=self.device)
        else:
            with torch.no_grad():
                comp = self.unet.feature_compressor(feats).float()

        idx_dst = self._remap_seed_idx_with_bbox(
            seed_idx_raw.long(), mean_seed.float(), scale_seed.float(), mean_gt.float(), scale_gt.float(), G
        )
        grid_feats, seed_occ = self.scatter_voxel_mean(idx_dst.int(), comp.float(), G)

        x_in = torch.cat([seed_occ, grid_feats], dim=1)

        meta = {
            "mean_gt": pack.get("mean_gt", np.zeros(3, np.float32)).astype(np.float32),
            "scale_gt": np.asarray(pack.get("scale_gt", 1.0), dtype=np.float32),
            "R_frame": pack.get("R_frame"),
            "frame_id_used": self._normalize_frame_id_value(pack.get("frame_id_used")),
        }
        return {
            "x_in": x_in,
            "occ_gt": occ_gt,
            "gt_idx": gt_idx,
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
            scores = vals

        feats = latents[:, coords[:, 0], coords[:, 1], coords[:, 2]].permute(1, 0)
        batch_col = torch.zeros(coords.shape[0], 1, dtype=torch.long, device=coords.device)
        coords4 = torch.cat([batch_col, coords.long()], dim=1)
        sparse = SparseTensor(feats=feats.contiguous(), coords=coords4.int())
        return sparse.to(self.device)

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

    @staticmethod
    def _normalize_frame_id_value(frame_id: Any) -> Optional[str]:
        if frame_id is None:
            return None
        if isinstance(frame_id, np.ndarray):
            if frame_id.size == 1:
                frame_id = frame_id.reshape(-1)[0]
            else:
                frame_id = frame_id.flatten()[0]
        if isinstance(frame_id, bytes):
            frame_id = frame_id.decode("utf-8")
        frame_str = str(frame_id)
        return frame_str.zfill(6) if frame_str.isdigit() else frame_str

    def _apply_pack_alignment(self, gauss, mean_gt: np.ndarray, scale_gt: float) -> None:
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

    def _apply_scene_alignment(self, gauss, translation: torch.Tensor, scale: torch.Tensor) -> None:
        device = gauss.get_xyz.device
        gauss.rescale(torch.tensor([2.0, 2.0, 2.0], device=device))
        gauss.translate(torch.tensor([-1.0, -1.0, -1.0], device=device))
        scale_vec = scale.flatten().to(device)
        if scale_vec.numel() == 1:
            scale_vec = scale_vec.repeat(3)
        elif scale_vec.numel() > 3:
            scale_vec = scale_vec[:3]
        gauss.rescale(scale_vec)
        translation_vec = translation.flatten().to(device)
        if translation_vec.numel() == 1:
            translation_vec = translation_vec.repeat(3)
        gauss.translate(translation_vec)

    def _clamp_gaussian_scale(self, reconstruction, bbox_scale: float) -> None:
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
        max_scale = 1 * (bbox_scale / self.grid_res).clamp_min(1e-5)
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
        camera = MiniCam(
            width=W,
            height=H,
            fovy=fovy,
            fovx=fovx,
            znear=0.01,
            zfar=100.0,
            R=cam_to_world[:3, :3].T,
            T=cam_to_world[:3, 3],
        )

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

    def _render_scene_graph_views(
        self,
        reconstruction,
        scene_graphs: Dict[str, Any],
        scene_id: str,
        frame_id: Optional[str],
    ) -> None:
        translations = scene_graphs["mean_obj_splat"]
        scales = scene_graphs["scale_obj_splat"]
        image_frames = scene_graphs["image_frames"]
        obj_intrinsics = scene_graphs["obj_intrinsics"]
        obj_2d_masks = scene_graphs.get("obj_2D_masks", {})

        frames = image_frames.get(scene_id, [])
        if not frames:
            LOGGER.warning("No frames recorded for scene %s", scene_id)
            return
        if frame_id is not None:
            frames = [frame_id]
        elif self.args.render_views and self.args.render_views > 0:
            frames = frames[: self.args.render_views]

        extra_dir = self.args.save_multiview_dir or self.output_dir
        os.makedirs(extra_dir, exist_ok=True)

        extrinsics_frames = scan3r.load_frame_poses(
            self.cfg.data.root_dir,
            scene_id,
            tuple(frames),
        )
        intrinsics = obj_intrinsics[scene_id]
        H, W = int(intrinsics["height"]), int(intrinsics["width"])
        fx, fy = intrinsics["intrinsic_mat"][0, 0], intrinsics["intrinsic_mat"][1, 1]
        fovx = focal2fov(fx, W)
        fovy = focal2fov(fy, H)
        K = intrinsics["intrinsic_mat"]

        for b, sid in enumerate(scene_graphs["scene_ids"]):
            if sid[0] != scene_id:
                continue
            recon_b = reconstruction[b]
            self._apply_scene_alignment(recon_b, translations[b], scales[b])
            self._clamp_gaussian_scale(recon_b, scales[b])
            for fid in frames:
                img_path = f"{self.cfg.data.root_dir}/scenes/{scene_id}/sequence/frame-{fid}.color.jpg"
                image = Image.open(img_path)
                image_t = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
                image_t = image_t.to(self.device, non_blocking=True)

                extrinsics = extrinsics_frames[fid]
                pose_camera_to_world = np.linalg.inv(extrinsics)
                camera = MiniCam(
                    width=W,
                    height=H,
                    fovy=fovy,
                    fovx=fovx,
                    znear=0.01,
                    zfar=100.0,
                    R=pose_camera_to_world[:3, :3].T,
                    T=pose_camera_to_world[:3, 3],
                    K=K,
                )

                rendered = render(
                    camera,
                    recon_b,
                    pipe=self._pipe_cfg,
                    bg_color=torch.tensor((0.0, 0.0, 0.0), device=self.device),
                )["render"]

                mask_np = obj_2d_masks.get(scene_id, {}).get(fid)
                if mask_np is not None:
                    mask = torch.as_tensor(mask_np, device=self.device)
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    mask = mask.to(image_t.dtype)
                    rendered = rendered * mask
                    image_t = image_t * mask

                frame_str = str(fid).zfill(6)
                pred_path = osp.join(extra_dir, f"{scene_id}_{frame_str}_gtScene_pred.png")
                gt_path = osp.join(extra_dir, f"{scene_id}_{frame_str}_gtScene_img.png")
                save_image(rendered.clamp(0, 1).unsqueeze(0).cpu(), pred_path)
                save_image(image_t.clamp(0, 1).unsqueeze(0).cpu(), gt_path)
                LOGGER.info("Saved scene-graph render for %s frame %s", scene_id, frame_str)

    def _predict_reconstruction(self, scene_id: str, frame_id: str):
        pack, pack_path = self._load_student_pack(scene_id, frame_id)
        feat_dim = pack.get("feats", np.zeros((0, 1024), np.float32)).shape[1]
        if feat_dim == 0:
            feat_dim = getattr(self.cfg.data, "student_feat_dim", 1024)
        self._init_unet(feat_dim)

        prepared = self._prepare_inputs(pack)
        x_in = prepared["x_in"]
        logits, latents = self._run_unet(x_in)
        sparse_latent = self._logits_to_sparse_latent(latents[0], logits[0, 0])

        if self.latent_model is None:
            raise RuntimeError("Teacher decoder not initialized")
        with torch.no_grad():
            reconstruction = self.latent_model.decode(sparse_latent)

        mean_gt = prepared["meta"]["mean_gt"]
        scale_gt = prepared["meta"]["scale_gt"]
        for gauss in reconstruction:
            self._apply_pack_alignment(gauss, mean_gt, scale_gt)
        return reconstruction, pack_path, prepared

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

    def evaluate_split(
        self,
        split: str,
        threshold: Optional[float] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        eval_thr = self.eval_threshold if threshold is None else threshold
        dataset = Scan3RSceneBatchDataset(self.cfg, split=split)
        total_tp = total_fp = total_fn = 0.0
        processed = 0
        for item in dataset.data_items:
            if max_samples is not None and processed >= max_samples:
                break
            scan_id = item.get("scan_id")
            frame_idx = item.get("frame_idx")
            if scan_id is None or frame_idx is None:
                continue
            fid = str(frame_idx).zfill(6)
            try:
                pack, _ = self._load_student_pack(scan_id, fid)
            except FileNotFoundError as err:
                LOGGER.warning("Skipping %s/%s: %s", scan_id, fid, err)
                continue

            feat_dim = pack.get("feats", np.zeros((0, 1024), np.float32)).shape[1]
            if feat_dim == 0:
                feat_dim = getattr(self.cfg.data, "student_feat_dim", 1024)
            self._init_unet(feat_dim)

            prepared = self._prepare_inputs(pack)
            logits, _ = self._run_unet(prepared["x_in"])
            probs = torch.sigmoid(logits)
            metrics = self._compute_prf_metrics(probs, prepared["occ_gt"], eval_thr)
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]
            processed += 1

        if processed == 0:
            raise RuntimeError(f"No samples evaluated for split {split}")

        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        fscore = 2 * precision * recall / (precision + recall + 1e-6)
        results = {
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
            "samples": processed,
        }
        LOGGER.info(
            "[%s split] precision=%.4f recall=%.4f F-score=%.4f over %d samples",
            split,
            precision,
            recall,
            fscore,
            processed,
        )
        return results

    def run(self, scene_id: str, frame_id: str) -> None:
        reconstruction, pack_path, prepared = self._predict_reconstruction(scene_id, frame_id)
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

        render_frame_id = frame_str
        frame_id_used = prepared["meta"].get("frame_id_used")
        if frame_id_used is not None:
            if frame_id_used != frame_str:
                LOGGER.warning(
                    "Pack frame_id_used=%s differs from requested %s; rendering with pack frame.",
                    frame_id_used,
                    frame_str,
                )
            render_frame_id = frame_id_used

        R_frame = self._normalize_rotation(prepared["meta"].get("R_frame"))
        if getattr(self.args, "invert_pack_rotation", False):
            R_frame = R_frame.T
        gt_img, pred_img = self._render_prediction(reconstruction[0], scene_id, render_frame_id, R_frame)

        pred_path = osp.join(
            self.output_dir,
            f"{scene_id}_{render_frame_id}_pred.png",
        )
        gt_path = osp.join(
            self.output_dir,
            f"{scene_id}_{render_frame_id}_gt.png",
        )
        save_image(pred_img.clamp(0, 1).unsqueeze(0).cpu(), pred_path)
        save_image(gt_img.clamp(0, 1).unsqueeze(0).cpu(), gt_path)
        comp = torch.stack([gt_img.clamp(0, 1), pred_img.clamp(0, 1)], dim=0)
        save_image(
            comp.cpu(),
            osp.join(
                self.output_dir,
                f"{scene_id}_{render_frame_id}_compare.png",
            ),
            nrow=2,
        )
        LOGGER.info(
            "Saved prediction %s (student pack %s)",
            pred_path,
            pack_path,
        )

    def _load_scene_graph_entry(self, scene_id: str):
        cfg_scene = copy.deepcopy(self.cfg)
        setattr(cfg_scene.data, "use_student_structure", False)
        # setattr(cfg_scene.data, "scene_level_single_frame", False)
        dataset = Scan3RSceneBatchDataset(cfg_scene, split="train")
        indices = [i for i, item in enumerate(dataset.data_items) if item["scan_id"] == scene_id]
        if not indices:
            raise ValueError(f"Scene {scene_id} not found in dataset split.")
        samples = [dataset[idx] for idx in indices]
        batch = dataset.collate_fn(samples)
        return batch["scene_graphs"]

    def run_scene_graph_mode(self, scene_id: str, frame_id: Optional[str]) -> None:
        scene_graphs = self._load_scene_graph_entry(scene_id)
        self._init_teacher()
        data_dict = {"scene_graphs": scene_graphs}
        if self.device.type == "cuda":
            data_dict = torch_util.to_cuda(data_dict)
        with torch.no_grad():
            embedding = self.latent_model.encode(data_dict)
            reconstruction = self.latent_model.decode(embedding)
        self._render_scene_graph_views(reconstruction, scene_graphs, scene_id, frame_id)


def parse_args() -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run SLAT completion inference from student packs")
    parser.add_argument("--config", required=True, type=str, help="Path to config YAML")
    parser.add_argument("--unet_checkpoint", required=True, type=str, help="Trained SLAT UNet checkpoint")
    parser.add_argument("--teacher_checkpoint", required=True, type=str, help="Latent autoencoder checkpoint")
    parser.add_argument("--scene_id", required=True, type=str, help="Scene ID to process")
    parser.add_argument("--frame_id", type=str, default=None, help="Frame ID (e.g. 000123) for single-view mode")
    parser.add_argument("--student_pack_root", type=str, default=None, help="Root directory for student packs")
    parser.add_argument("--occ_threshold", type=float, default=0.3, help="Occupancy threshold for selecting voxels")
    parser.add_argument("--max_latents", type=int, default=20000, help="Maximum voxels passed to decoder (None disables)")
    parser.add_argument("--output_dir", type=str, default="outputs/slat_completion", help="Directory for rendered results")
    parser.add_argument("--render_views", type=int, default=0, help="Number of scene-level GT views to render (0=single pack view)")
    parser.add_argument("--render_from_gt_annotations", action="store_true", help="Use scene-level GT annotations for multi-view rendering like train_scene_gs")
    parser.add_argument("--save_multiview_dir", type=str, default=None, help="Optional output subdir for multiview renders")
    parser.add_argument(
        "--invert_pack_rotation",
        action="store_true",
        help="Render using the inverse of pack R_frame (debug for canonical rotation direction).",
    )
    parser.add_argument(
        "--pose_is_w2c",
        action="store_true",
        help="Interpret scan3r poses as world-to-camera when applying pack rotation.",
    )
    parser.add_argument("--eval_split", type=str, default=None, help="Evaluate voxel metrics on the given dataset split instead of running single-scene inference")
    parser.add_argument("--eval_threshold", type=float, default=0.5, help="Occupancy threshold for voxel precision/recall evaluation")
    parser.add_argument("--max_eval", type=int, default=None, help="Optional max number of samples to evaluate")
    args, unknown = parser.parse_known_args()
    return args, unknown


def main() -> None:
    common.init_log(level=logging.INFO)
    args, unknown = parse_args()
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    runner = SlatCompletionInference(cfg, args)
    if args.eval_split:
        runner.evaluate_split(args.eval_split, threshold=args.eval_threshold, max_samples=args.max_eval)
    elif args.render_from_gt_annotations:
        runner.run_scene_graph_mode(args.scene_id, args.frame_id)
    else:
        if args.frame_id is None:
            raise ValueError("--frame_id is required for single-view inference")
        runner.run(args.scene_id, args.frame_id)


if __name__ == "__main__":
    main()
