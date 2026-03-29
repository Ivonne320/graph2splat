import argparse
import os
import os.path as osp
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from configs import Config, update_configs
from src.models.structure_model_with_bbox_head import StructureModel
from utils import scan3r


def parse_args() -> Tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser(
        description="Visualize structure completion results for a scene/frame."
    )
    parser.add_argument("--config", type=str, required=True, help="Config file path.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to structure model checkpoint."
    )
    parser.add_argument(
        "--scene_id", type=str, required=True, help="Scene ID to visualize."
    )
    parser.add_argument(
        "--frame_id",
        type=str,
        required=True,
        help="Frame ID used during preprocessing (e.g., 000000).",
    )
    parser.add_argument(
        "--structure_pack_root",
        type=str,
        default="/cluster/scratch/wangyih/3RScan",
        help="Root directory containing scene_level_structure packs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the visualization image.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Occupancy probability threshold for predicted voxels.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=50000,
        help="Maximum number of points to plot for each set.",
    )
    return parser.parse_known_args()


def load_model(cfg: Config, checkpoint: str, device: torch.device) -> StructureModel:
    model = StructureModel(cfg=cfg.autoencoder, device=device)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def scatter_voxel_mean(idx_t: torch.Tensor, feat_t: torch.Tensor, G: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = feat_t.device
    if idx_t.numel() == 0:
        C = feat_t.shape[-1] if feat_t.ndim == 2 else 64
        grid_feats = torch.zeros(1, C, G, G, G, device=device, dtype=feat_t.dtype)
        seed_occ = torch.zeros(1, 1, G, G, G, device=device, dtype=feat_t.dtype)
        return grid_feats, seed_occ

    M, C = feat_t.shape
    lin = (idx_t[:, 0] * G * G + idx_t[:, 1] * G + idx_t[:, 2]).long()

    Csum = torch.zeros(C, G * G * G, device=device, dtype=feat_t.dtype)
    cnt = torch.zeros(G * G * G, device=device, dtype=feat_t.dtype)

    Csum.index_add_(1, lin, feat_t.T)
    cnt.index_add_(0, lin, torch.ones(M, device=device, dtype=feat_t.dtype))

    mask = cnt > 0
    Csum[:, mask] /= cnt[mask]
    grid_feats = Csum.view(C, G, G, G).unsqueeze(0)

    seed_occ = torch.zeros(1, 1, G, G, G, device=device, dtype=feat_t.dtype)
    uniq = torch.unique(lin)
    seed_occ.view(1, 1, -1)[0, 0, uniq] = 1.0
    return grid_feats, seed_occ


def remap_seed_idx_with_bbox(
    seed_idx: torch.Tensor,
    mean_src: torch.Tensor,
    scale_src: torch.Tensor,
    mean_dst: torch.Tensor,
    scale_dst: torch.Tensor,
    G: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    if seed_idx.numel() == 0:
        return seed_idx

    c_seed = (seed_idx.float() + 0.5) / G - 0.5
    s_src = torch.clamp(scale_src.view(1, 1), min=eps)
    m_src = mean_src.view(1, 3)
    world_can = c_seed * (2.0 * s_src) + m_src

    s_dst = torch.clamp(scale_dst.view(1, 1), min=eps)
    m_dst = mean_dst.view(1, 3)
    centers_dst = (world_can - m_dst) / (2.0 * s_dst)
    idx_dst = torch.floor((centers_dst + 0.5) * G).long()
    return torch.clamp(idx_dst, 0, G - 1)


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(0)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def remap_idx_between_boxes(
    idx: np.ndarray,
    G: int,
    mu_src: np.ndarray,
    s_src: float,
    mu_dst: np.ndarray,
    s_dst: float,
) -> np.ndarray:
    if idx.size == 0:
        return idx
    centers_normed = (idx.astype(np.float32) + 0.5) / G - 0.5
    src_points = centers_normed * (2.0 * float(s_src)) + mu_src[None, :]
    dst_norm = (src_points - mu_dst[None, :]) / (2.0 * float(s_dst))
    idx_dst = np.floor((dst_norm + 0.5) * G).astype(np.int32)
    return np.clip(idx_dst, 0, G - 1)


def prepare_canonical_points(idx: np.ndarray, scale: float, mean: np.ndarray, G: int) -> np.ndarray:
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centers_normed = (idx.astype(np.float32) + 0.5) / G - 0.5
    centers_can = centers_normed * (2.0 * float(scale)) + mean[None, :]
    return centers_can.astype(np.float32)


def visualize(
    cfg: Config,
    model: StructureModel,
    args: argparse.Namespace,
) -> None:
    device = next(model.parameters()).device
    pack_path = osp.join(
        args.structure_pack_root,
        "files",
        "gs_annotations",
        args.scene_id,
        "scene_level_structure",
        f"student_pack_aligned_{args.frame_id}.npz",
    )
    if not osp.isfile(pack_path):
        raise FileNotFoundError(f"Structure pack not found: {pack_path}")
    pack = np.load(pack_path)
    G = int(pack["G"])

    seed_idx_np = pack["seed_idx"].astype(np.int32)
    feats_np = pack["feats"].astype(np.float32)
    mean_seed0_np = pack["seed_box_init_mean"].astype(np.float32)
    scale_seed0 = float(pack["seed_box_init_scale"])
    mean_gt_np = pack["mean_gt"].astype(np.float32)
    scale_gt = float(pack["scale_gt"])
    gt_idx_np = pack["vox_idx_gt_occ"].astype(np.int32)
    R_frame = pack["R_frame"].astype(np.float32)

    seed_idx_t = torch.from_numpy(seed_idx_np).long().to(device)
    feats_t = torch.from_numpy(feats_np).float().to(device)

    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            feats_comp = model.comp(feats_t)
        mean_seed0 = torch.from_numpy(mean_seed0_np).float().to(device).view(1, 3)
        scale_seed0_t = torch.tensor(scale_seed0, dtype=torch.float32, device=device).view(1, 1)

        mean_gt_t = torch.from_numpy(mean_gt_np).float().to(device).view(1, 3)
        scale_gt_t = torch.tensor(scale_gt, dtype=torch.float32, device=device).view(1, 1)

        idx_dst = remap_seed_idx_with_bbox(
            seed_idx_t,
            mean_seed0,
            scale_seed0_t,
            mean_gt_t,
            scale_gt_t,
            G,
        )
        idx_dst_np = idx_dst.detach().cpu().numpy()
        grid_feats, seed_occ = scatter_voxel_mean(idx_dst.int(), feats_comp.float(), G)
        x_in = torch.cat([seed_occ, grid_feats], dim=1)

        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            z, mu, _logvar, _feat = model.encoder(
                x_in, sample_posterior=False, return_raw=True, return_feat=True
            )
            logits = model.decoder(mu)

    probs = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
    pred_idx_np = np.stack(np.nonzero(probs > args.threshold), axis=1).astype(np.int32)

    seed_points_can = prepare_canonical_points(idx_dst_np, scale_gt, mean_gt_np, G)
    pred_points_can = prepare_canonical_points(pred_idx_np, scale_gt, mean_gt_np, G)
    gt_points_can = prepare_canonical_points(gt_idx_np, scale_gt, mean_gt_np, G)

    seed_points_can = downsample_points(seed_points_can, args.max_points)
    pred_points_can = downsample_points(pred_points_can, args.max_points)
    gt_points_can = downsample_points(gt_points_can, args.max_points)

    scenes_dir = osp.join(cfg.data.root_dir, "scenes")
    frame_path = osp.join(
        scenes_dir, args.scene_id, "sequence", f"frame-{args.frame_id}.color.jpg"
    )
    if not osp.isfile(frame_path):
        raise FileNotFoundError(f"Frame image not found: {frame_path}")
    rgb = plt.imread(frame_path)

    mesh = scan3r.load_ply_mesh(
        data_dir=scenes_dir,
        scan_id=args.scene_id,
        label_file_name="labels.instances.annotated.v2.ply",
    )
    gt_voxels_can = downsample_points(gt_points_can, args.max_points)

    all_xy = []
    for arr in (seed_points_can, pred_points_can, gt_points_can, gt_voxels_can):
        if arr.size > 0:
            all_xy.append(arr[:, :2])
    xy_limits = None
    if all_xy:
        stacked = np.concatenate(all_xy, axis=0)
        min_xy = stacked.min(axis=0)
        max_xy = stacked.max(axis=0)
        pad = (max_xy - min_xy) * 0.05
        xy_limits = (min_xy - pad, max_xy + pad)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(rgb)
    axes[0].axis("off")

    axes[1].scatter(
        seed_points_can[:, 0],
        seed_points_can[:, 1],
        c=seed_points_can[:, 2] if seed_points_can.size > 0 else None,
        s=2,
        cmap="viridis",
    )
    axes[1].axis("off")

    axes[2].scatter(
        pred_points_can[:, 0],
        pred_points_can[:, 1],
        c=pred_points_can[:, 2] if pred_points_can.size > 0 else None,
        s=2,
        cmap="magma",
    )
    axes[2].axis("off")

    axes[3].scatter(
        gt_voxels_can[:, 0],
        gt_voxels_can[:, 1],
        c=gt_voxels_can[:, 2] if gt_voxels_can.size > 0 else None,
        s=1,
        cmap="plasma",
        alpha=0.6,
    )
    axes[3].axis("off")

    for ax in axes[1:]:
        ax.set_aspect("equal", adjustable="box")
        if xy_limits is not None:
            ax.set_xlim(xy_limits[0][0], xy_limits[1][0])
            ax.set_ylim(xy_limits[0][1], xy_limits[1][1])

    fig.tight_layout()
    output_path = args.output or f"structure_completion_{args.scene_id}_{args.frame_id}.png"
    os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def main() -> None:
    args, unknown = parse_args()
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)
    visualize(cfg, model, args)


if __name__ == "__main__":
    main()
