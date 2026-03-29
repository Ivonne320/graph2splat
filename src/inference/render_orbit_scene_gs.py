#!/usr/bin/env python3
"""
Render orbit views for a checkpoint trained with src/trainval/train_scene_gs.py.

Example:
python scripts/inference/render_orbit_scene_gs.py \
    --config scripts/train_val/train_batch_scene_gs.yaml \
    --snapshot /path/to/snapshots/epoch-10.pth.tar \
    --index 0 \
    --output ./orbit_vis
"""
import argparse
import copy
import os
import os.path as osp
from typing import List, Optional

import imageio
import numpy as np
import torch
from argparse import Namespace
from gaussian_renderer import render
from scene.cameras import MiniCam
from torchvision.utils import save_image

from configs import update_configs
from src.datasets import Scan3RObjectDataset, Scan3RSceneBatchDataset
from src.models.latent_autoencoder import LatentAutoencoder
from src.models.latent_autoencoder_rgb_skip import LatentAutoencoderRgbSkip
from src.representations.gaussian.gaussian_model import Gaussian
from utils import common, scan3r, torch_util
from utils.gaussian_splatting import GaussianSplat
from utils.graphics_utils import focal2fov, getProjectionMatrix
from utils.general_utils import inverse_sigmoid


def apply_scene_alignment(reconstruction: Gaussian, translation: torch.Tensor, scale: torch.Tensor) -> None:
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


def clamp_gaussian_scale(reconstruction: Gaussian, bbox_scale: torch.Tensor) -> None:
    device = reconstruction.get_xyz.device
    dtype = reconstruction.get_xyz.dtype
    bbox_scale = torch.as_tensor(bbox_scale, device=device, dtype=dtype).flatten()
    if bbox_scale.numel() == 0:
        return
    if bbox_scale.numel() == 1:
        bbox_scale = bbox_scale.repeat(3)
    elif bbox_scale.numel() > 3:
        bbox_scale = bbox_scale[:3]
    max_scale = (bbox_scale / 128).clamp_min(1e-5).view(1, 3)
    clamped = torch.minimum(reconstruction.get_scaling, max_scale)
    reconstruction.from_scaling(clamped)


def gaussians_to_splat(gaussians: List[Gaussian]) -> GaussianSplat:
    if not gaussians:
        raise ValueError("No gaussians to convert.")
    xyz = torch.cat([g.get_xyz for g in gaussians], dim=0)
    features_dc = torch.cat([g._features_dc for g in gaussians], dim=0)
    features_rest: Optional[torch.Tensor] = None
    opacity = torch.cat(
        [torch.logit(g.get_opacity.clamp(1e-5, 1.0 - 1e-5)) for g in gaussians],
        dim=0,
    )
    scaling = torch.cat(
        [torch.log(g.get_scaling.clamp_min(1e-6)) for g in gaussians],
        dim=0,
    )
    rotation = torch.cat([g.get_rotation for g in gaussians], dim=0)
    return GaussianSplat(
        xyz=xyz,
        features_dc=features_dc,
        opacity=opacity,
        scaling=scaling,
        rotation=rotation,
        features_rest=features_rest,
    )


def render_orbit(
    representation: GaussianSplat,
    scene_id: str,
    intrinsics: dict,
    num_frames: int,
    radius: float,
    height: float,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    scene_center = representation.xyz.detach().cpu().numpy().mean(axis=0)
    angle_step = 2 * np.pi / num_frames
    rendered_frames = []
    pipe_cfg = Namespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)

    for i in range(num_frames):
        theta = i * angle_step
        cam_x = scene_center[0] + radius * np.cos(theta)
        cam_y = scene_center[2] + radius * np.sin(theta)
        cam_z = scene_center[1] + height
        

        cam_position = np.array([cam_y, cam_x, cam_z])
        forward = scene_center - cam_position
        forward /= np.linalg.norm(forward)
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        R = np.stack([right, up, forward], axis=1)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R
        view[:3, 3] =  cam_position
        view = np.linalg.inv(view)

        # world_view_transform = torch.tensor(view, device="cuda", dtype=torch.float32)
        fovx = focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"])
        fovy = focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"])
        # projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        # full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
        
        cam = MiniCam(
            width=int(intrinsics["width"]),
            height=int(intrinsics["height"]),
            fovy=fovy,
            fovx=fovx,
            znear=0.01,
            zfar=100.0,
            R=view[:3, :3].T,
            T=view[:3, 3],
            
        )
        rendered = render(
            cam,
            representation,
            pipe=pipe_cfg,
            bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
        )["render"]

        frame_path = osp.join(output_dir, f"{scene_id}_frame_{i:03d}.png")
        # save_image(rendered, frame_path)
        rendered_frames.append(
            (rendered.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        )
        torch.cuda.empty_cache()

    video_path = osp.join(output_dir, f"{scene_id}_orbit.mp4")
    imageio.mimsave(video_path, rendered_frames, fps=30)
    print(f"Saved orbit video to {video_path}")


def _normalize_frame_id(frame_id: str) -> str:
    fid = str(frame_id)
    if fid.isdigit():
        return f"{int(fid):06d}"
    return fid


def _resolve_frame_id(frame_idx: Optional[str], data_root: str, scene_id: str) -> str:
    if frame_idx is None:
        raise ValueError("frame_idx is required for frame rendering.")
    fid = str(frame_idx)
    pose_path = osp.join(
        data_root, "scenes", scene_id, "sequence", f"frame-{fid}.pose.txt"
    )
    if osp.exists(pose_path):
        return fid
    fid_pad = _normalize_frame_id(fid)
    pose_path = osp.join(
        data_root, "scenes", scene_id, "sequence", f"frame-{fid_pad}.pose.txt"
    )
    if osp.exists(pose_path):
        return fid_pad
    return fid


def _render_pose_pair(
    representation: GaussianSplat,
    scene_id: str,
    intrinsics: dict,
    pose_c2w: np.ndarray,
    output_dir: str,
    frame_id: str,
    novel_yaw_deg: float,
    novel_shift: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pipe_cfg = Namespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    bg = torch.tensor((0.0, 0.0, 0.0), device="cuda")

    def _make_cam(pose: np.ndarray) -> MiniCam:
        return MiniCam(
            width=int(intrinsics["width"]),
            height=int(intrinsics["height"]),
            fovy=focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"]),
            fovx=focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"]),
            znear=0.01,
            zfar=100.0,
            R=pose[:3, :3].T,
            T=pose[:3, 3],
            K=intrinsics["intrinsic_mat"],
        )

    cam = _make_cam(pose_c2w)
    rendered = render(cam, representation, pipe=pipe_cfg, bg_color=bg)["render"]
    save_image(rendered, osp.join(output_dir, f"{scene_id}_frame_{frame_id}_pose.png"))

    yaw = np.deg2rad(novel_yaw_deg)
    rot_y = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    R_c2w = pose_c2w[:3, :3]
    t_c2w = pose_c2w[:3, 3]
    t_c2w = t_c2w + novel_shift * R_c2w[:, 0]
    pose_novel = pose_c2w.copy()
    pose_novel[:3, :3] = rot_y @ R_c2w
    pose_novel[:3, 3] = t_c2w

    cam_novel = _make_cam(pose_novel)
    rendered_novel = render(cam_novel, representation, pipe=pipe_cfg, bg_color=bg)[
        "render"
    ]
    save_image(
        rendered_novel, osp.join(output_dir, f"{scene_id}_frame_{frame_id}_novel.png")
    )


def parse_args():
    parser = argparse.ArgumentParser("Render orbit views for a trained scene-gs model.")
    parser.add_argument("--config", required=True, help="Path to config used for training.")
    parser.add_argument("--snapshot", required=True, help="Checkpoint produced by train_scene_gs.py.")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to visualize.")
    parser.add_argument("--scene_id", type=str, default=None, help="Explicit scene id to visualize.")
    parser.add_argument(
        "--frame_idx",
        "--frame_ids",
        type=str,
        default=None,
        help="Optional frame index filter when selecting by --scene_id.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--output", type=str, default="./orbit_vis", help="Where to store frames and video.")
    parser.add_argument("--num_frames", type=int, default=120, help="Number of orbit frames.")
    parser.add_argument("--radius", type=float, default=2.0, help="Orbit radius.")
    parser.add_argument("--height", type=float, default=1.5, help="Camera height relative to scene center.")
    parser.add_argument("--object_ids", type=int, nargs="+", default=None, help="Optional object ids to render.")
    parser.add_argument(
        "--render_frame",
        action="store_true",
        help="Render a specific frame pose and a nearby novel view.",
    )
    parser.add_argument(
        "--novel_yaw_deg",
        type=float,
        default=5.0,
        help="Yaw offset (deg) for the nearby novel view.",
    )
    parser.add_argument(
        "--novel_shift",
        type=float,
        default=0.05,
        help="Rightward shift (m) for the nearby novel view.",
    )
    return parser.parse_known_args()


def main():
    common.init_log()
    args, unknown = parse_args()
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    # The scene-level encoder expects raw Gaussian splats with 1024-dim features.
    # Matching the training setup requires loading splats from gs_annotations instead
    # of the compressed latent embeddings stored under gs_embeddings.
    cfg.data.preload_slat = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_object_dataset = getattr(cfg.train, "object_level", False)
    dataset_cls = Scan3RObjectDataset if use_object_dataset else Scan3RSceneBatchDataset
    
    dataset = dataset_cls(cfg, split=args.split)
    print(f"Loaded dataset with {len(dataset)} items across {len(getattr(dataset, 'scan_ids', []))} scans.")

    model = LatentAutoencoder(cfg.autoencoder, device=device)
    # model = LatentAutoencoderRgbSkip(cfg.autoencoder, device=device)
    state = torch.load(args.snapshot, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    sample_index = args.index
    sample = None
    if args.scene_id is not None:
        frames_seen: List[int] = []
        requested_frame = (
            _normalize_frame_id(args.frame_idx)
            if args.frame_idx is not None
            else None
        )
        for idx in range(len(dataset)):
            candidate = dataset[idx]
            if candidate["scan_id"] != args.scene_id:
                continue
            cand_frame = _normalize_frame_id(candidate["frame_idx"])
            frames_seen.append(cand_frame)
            if requested_frame is not None and cand_frame != requested_frame:
                continue
            sample = candidate
            sample_index = idx
            break
        if sample is None:
            if not frames_seen:
                raise ValueError(
                    f"Scene id {args.scene_id} not found in dataset of size {len(dataset)}."
                )
            raise ValueError(
                f"Frame {args.frame_idx} not found for scene {args.scene_id}. "
                f"Available frames: {sorted(set(frames_seen))[:10]}{'...' if len(frames_seen) > 10 else ''}"
            )
    else:
        if not (0 <= sample_index < len(dataset)):
            raise ValueError(f"Index {sample_index} is out of range for dataset of size {len(dataset)}.")
        sample = dataset[sample_index]
    print(f"Selected dataset index {sample_index} -> scene {sample['scan_id']} frame {sample['frame_idx']}.")
    data_dict = dataset.collate_fn([sample])
    data_dict = torch_util.to_cuda(data_dict) if torch.cuda.is_available() else data_dict

    with torch.no_grad():
        embedding = model.encode(data_dict)
        reconstruction = model.decode(embedding)

    scene_ids = data_dict["scene_graphs"]["scene_ids"]
    scene_id = scene_ids[0][0] if isinstance(scene_ids[0], (list, np.ndarray)) else scene_ids[0]
    translations = data_dict["scene_graphs"]["mean_obj_splat"]
    scales = data_dict["scene_graphs"]["scale_obj_splat"]

    selected_idx = list(range(len(reconstruction)))
    if args.object_ids is not None:
        obj_ids = data_dict["scene_graphs"]["obj_ids"]
        target = set(int(o) for o in args.object_ids)
        selected_idx = [i for i, oid in enumerate(obj_ids) if int(oid) in target]
        if not selected_idx:
            raise ValueError("None of the requested object_ids are present in this sample.")

    for i in selected_idx:
        apply_scene_alignment(reconstruction[i], translations[i], scales[i])
        clamp_gaussian_scale(reconstruction[i], scales[i])

    gaussians = [copy.deepcopy(reconstruction[i]) for i in selected_idx]
    representation = gaussians_to_splat(gaussians)

    intrinsics = dataset.image_intrinsics[scene_id]
    if args.render_frame:
        if args.scene_id is None or args.frame_idx is None:
            raise ValueError("--render_frame requires --scene_id and --frame_idx.")
        frame_id = _resolve_frame_id(args.frame_idx, cfg.data.root_dir, scene_id)
        pose = scan3r.load_pose(cfg.data.root_dir, scene_id, frame_id)
        pose_c2w = np.linalg.inv(pose)
        _render_pose_pair(
            representation,
            scene_id,
            intrinsics,
            pose_c2w,
            output_dir=args.output,
            frame_id=frame_id,
            novel_yaw_deg=args.novel_yaw_deg,
            novel_shift=args.novel_shift,
        )
    render_orbit(
        representation,
        scene_id,
        intrinsics,
        num_frames=args.num_frames,
        radius=args.radius,
        height=args.height,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
