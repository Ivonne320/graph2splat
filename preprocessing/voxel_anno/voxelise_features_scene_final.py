import itertools
import logging
from collections import Counter, defaultdict
import os
import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple
import copy

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import utils3d
import json
import time

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from configs import Config, update_configs
from utils import common, scan3r
from utils import visualisation as vis
from utils.canonicalize import align_gravity_with_plane, yaw_canonicalize_xy
# voxelize

_LOGGER = logging.getLogger(__name__)

@torch.no_grad()
def _get_dino_embedding(images: torch.Tensor):
    """
    images: (B,3,H,W) in [0,1]
    returns:
      emb:   (B, C, H_p, W_p)
      sizes: (H_in, W_in, H_p, W_p)
    """
    # preprocess with your transform
    imgs = images.reshape(-1, 3, images.shape[-2], images.shape[-1]).cpu()
    inp  = transform(imgs).cuda()         # (B,3,H_in,W_in)

    model.eval()
    if hasattr(model, "forward_features"):
        out = model.forward_features(inp)
    else:
        out = model(inp, is_training=True)    # dict with patch tokens

    # 1) take patch tokens directly (B, N, C)
    if "x_norm_patchtokens" not in out:
        # fallback if needed
        reg = getattr(model, "num_register_tokens", 0)
        tok = out["x_prenorm"][:, 1 + reg : ]         # (B, N, C)
    else:
        tok = out["x_norm_patchtokens"]               # (B, N, C)

    # 2) compute patch grid from actual input + patch size
    H_in, W_in = int(inp.shape[-2]), int(inp.shape[-1])
    p = getattr(model, "patch_size", None)
    if p is None and hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
        p = model.patch_embed.patch_size
    p = int(p[0] if isinstance(p, (tuple, list)) else p)
    H_p, W_p = H_in // p, W_in // p
    assert tok.shape[1] == H_p * W_p, f"N={tok.shape[1]} != {H_p*W_p} (H_in={H_in}, W_in={W_in}, p={p})"

    # 3) (B,N,C) -> (B,C,H_p,W_p)
    emb = tok.permute(0, 2, 1).contiguous().view(tok.size(0), tok.size(2), H_p, W_p)
    return emb, (H_in, W_in, H_p, W_p)

def unproject_frame_to_world(frame_id, extrinsics, intrinsics_rgb, intrinsics_depth, root_dir, scan_id):
    """
    Returns world_points (N,3) and an optional valid mask for one frame.
    Handles depth->RGB resize and intrinsics scaling.
    Assumes `extrinsics[frame_id]` is CAMERA->WORLD (C2W). If yours is W2C, invert it here.
    """
    rgb_path   = f"{root_dir}/scenes/{scan_id}/sequence/frame-{frame_id}.color.jpg"
    depth_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{frame_id}.depth.pgm"

    # Load
    with Image.open(rgb_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)  # (Hrgb,Wrgb,3)
    depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0  # meters

    H_d, W_d = depth.shape
    H_r, W_r = rgb.shape[:2]

    # Nearest resize depth to RGB resolution
    depth_rgb = np.array(Image.fromarray(depth).resize((W_r, H_r), Image.NEAREST), dtype=np.float32)

    # Scale depth intrinsics to RGB size
    Kd = intrinsics_depth.copy()
    Kd[0,0] *= W_r / W_d  # fx
    Kd[0,2] *= W_r / W_d  # cx
    Kd[1,1] *= H_r / H_d  # fy
    Kd[1,2] *= H_r / H_d  # cy
    fx, fy, cx, cy = Kd[0,0], Kd[1,1], Kd[0,2], Kd[1,2]

    # Pixel grid
    u, v = np.meshgrid(np.arange(W_r, dtype=np.float32),
                       np.arange(H_r, dtype=np.float32), indexing="xy")
    Z = depth_rgb.reshape(-1)
    u = u.reshape(-1)
    v = v.reshape(-1)

    # Validity (tune near/far)
    z_near, z_far = 0.05, 10.0
    valid = np.isfinite(Z) & (Z > z_near) & (Z < z_far)
    u = u[valid]; v = v[valid]; Z = Z[valid]

    # Back-project to camera
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    cam_pts = np.stack([X, Y, Z], axis=1)  # (N,3)

    # Camera->World for this frame
    T_cw = extrinsics[frame_id].astype(np.float32)   # if your extrinsics are W2C, do: T_cw = np.linalg.inv(extrinsics[frame_id])
    R_cw, t_cw = T_cw[:3,:3], T_cw[:3,3]
    world_pts = cam_pts @ R_cw.T + t_cw[None,:]
    return world_pts

def normalize_points_with_padding(points: np.ndarray, G: int,
                                  pad_ratio: float = None,
                                  pad_voxels: int = None,
                                  isotropic: bool = False):
    """
    Normalize to [-0.5, 0.5]^3 but *shrink* visible content so it sits inside
    an inner cube, leaving empty margins for completion.

    Args:
        points: (N,3) world coords
        G: grid size (e.g., 64)
        pad_ratio: multiplicative padding on scale (e.g., 1.25 means 25% extra margin).
        pad_voxels: exact empty-voxel margin per face (e.g., 6 -> leave ~6 voxels empty on each side).
        isotropic: if True, use a single scalar scale (max over axes). If False (default), per-axis.

    Returns:
        mean: (3,), scale: (3,)  or scalar if isotropic=True
        pts_normed: (N,3) in [-0.5, 0.5] (most will be in a tighter inner cube).
    """
    assert (pad_ratio is None) ^ (pad_voxels is None), \
        "Provide exactly one of pad_ratio or pad_voxels."

    mean = points.mean(axis=0).astype(np.float32)
    pts  = (points - mean).astype(np.float32)

    # base scale (so that without padding, points would just touch +/-0.5)
    if isotropic:
        base = np.max(np.abs(pts))
        base = 1.0 if base == 0 else base
        scale = np.array([base, base, base], dtype=np.float32)
    else:
        base = np.max(np.abs(pts), axis=0)
        base[base == 0] = 1.0
        scale = base.astype(np.float32)

    if pad_ratio is not None:
        # simply enlarge scale by a factor >1 -> content shrinks in normalized cube
        scale = scale * float(pad_ratio)
    else:
        # exact voxel margin M on each side -> inner half-size = 0.5 - M/G
        # want max_norm = 0.5 - m, with m = M/G
        # original max_norm (without padding) would be 0.5; to get smaller inner half-size h:
        # scale' = scale * (0.5 / h) = scale / (1 - 2*m)
        m = float(pad_voxels) / float(G)
        factor = 1.0 / max(1e-6, (1.0 - 2.0 * m))  # safe guard
        scale = scale * factor

    # normalize
    pts_normed = pts / (2.0 * scale[None, :])
    pts_normed = np.clip(pts_normed, -0.5 + 1e-6, 0.5 - 1e-6)
    return mean, scale, pts_normed

def _prep_image_for_dino(np_rgb_uint8: np.ndarray) -> torch.Tensor:
    # (H,W,3) uint8 -> (1,3,H,W) float in [0,1]
    t = torch.from_numpy(np_rgb_uint8.copy()).permute(2,0,1).float() / 255.0
    return t.unsqueeze(0).cuda()

def _project_to_image(
    voxel: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    grid_size: tuple[int] = (64, 64, 64),
):
    voxel_size = 1.0 / grid_size[0]
    voxel = voxel.float() * voxel_size
    assert voxel.min() >= 0.0 and voxel.max() <= 1.0

    voxel = voxel * 2.0 - 1.0
    assert voxel.min() >= -1.0 and voxel.max() <= 1.0
    # voxel = voxel * scale + mean
    voxel = voxel * scale[None, :] + mean[None, :]
    uv = utils3d.torch.project_cv(
        voxel.float(), extrinsics.float(), intrinsics.float()
    )[0]
    return uv

def remap_seed_idx_between_norms(seed_idx_src, G, mean_src, scale_src, mean_dst, scale_dst):
    """
    seed_idx_src: (M,3) int32 indices on the source (PCD) grid
    mean_src, scale_src: (3,) world-space params for source grid
    mean_dst, scale_dst: (3,) world-space params for destination (GT) grid
    Returns: seed_idx_dst (M,3) int32 on the destination grid
    """
    seed_idx_src = np.asarray(seed_idx_src, dtype=np.int32)
    # idx -> normalized (src)
    x_norm = (seed_idx_src.astype(np.float32) + 0.5) / G - 0.5  # [-0.5,0.5]
    # normalized (src) -> world
    x_world = x_norm * (2.0 * scale_src[None, :]) + mean_src[None, :]
    # world -> normalized (dst)
    x_norm_dst = (x_world - mean_dst[None, :]) / (2.0 * scale_dst[None, :])
    # normalized (dst) -> idx
    i_dst = np.floor((x_norm_dst + 0.5) * G).astype(np.int32)
    i_dst = np.clip(i_dst, 0, G - 1)
    return i_dst

def voxelize_mesh_simple_dense(scene_mesh, G=64, n_rand=2, k_dilate=3, device="cuda"):
    """
    Minimal voxelizer:
      - take vertices
      - add edge midpoints + face centroids
      - add a few random barycentric samples per face
      - bin to voxel indices
      - optional small 3D dilation for thickness
    All coordinates are assumed normalized to [-0.5, 0.5].
    """
    V = np.asarray(scene_mesh.vertices, dtype=np.float32)
    Fidx = np.asarray(scene_mesh.triangles, dtype=np.int32)
    if Fidx.size == 0 or V.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    v0 = V[Fidx[:, 0]]
    v1 = V[Fidx[:, 1]]
    v2 = V[Fidx[:, 2]]

    # Base points: vertices, edge midpoints, face centroids
    mids01 = (v0 + v1) * 0.5
    mids12 = (v1 + v2) * 0.5
    mids20 = (v2 + v0) * 0.5
    cents  = (v0 + v1 + v2) / 3.0

    pts_list = [V, mids01, mids12, mids20, cents]

    # A few random barycentric samples per face (very cheap)
    if n_rand > 0:
        u = np.random.rand(Fidx.shape[0], n_rand, 1).astype(np.float32)
        v = np.random.rand(Fidx.shape[0], n_rand, 1).astype(np.float32)
        swap = (u + v > 1).astype(np.float32)
        u = u * (1 - swap) + (1 - u) * swap
        v = v * (1 - swap) + (1 - v) * swap
        w = 1.0 - u - v
        # shape: (F, n_rand, 3)
        tri = (u * v0[:, None, :] + v * v1[:, None, :] + w * v2[:, None, :]).reshape(-1, 3)
        pts_list.append(tri)

    pts = np.concatenate(pts_list, axis=0)  # (N,3)

    # Quantize to voxel indices
    eps = 1e-6
    idx = np.floor((pts + 0.5 - eps) * G).astype(np.int32)
    idx = np.clip(idx, 0, G - 1)
    idx = np.unique(idx, axis=0)  # drop duplicates

    # Optional: small morphological dilation on GPU (fast)
    if k_dilate > 0 and idx.shape[0] > 0:
        occ = torch.zeros((1, 1, G, G, G), device=device, dtype=torch.uint8)
        occ[0, 0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        pad = k_dilate // 2
        occ = F.max_pool3d(occ.float(), kernel_size=k_dilate, stride=1, padding=pad)
        xyz = (occ > 0.5).nonzero(as_tuple=False)  # (N,5) with batch/channels dims
        if xyz.numel() > 0:
            idx = torch.stack([xyz[:, 2], xyz[:, 3], xyz[:, 4]], dim=1).detach().cpu().numpy()

    return idx.astype(np.int64)

def _dilate_voxels(voxel_grid: o3d.geometry.VoxelGrid) -> np.ndarray:
    voxel_grid = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    # densify voxel grid
    dilated_voxels = set()
    directions = [d for d in itertools.product([-1, 0, 1], repeat=3) if d != (0, 0, 0)]
    for v in voxel_grid:
        dilated_voxels.add(tuple(v))
        for d in directions:
            neighbor = tuple(v + np.array(d))
            if all(0 <= n < 64 for n in neighbor):
                dilated_voxels.add(neighbor)
    voxel_grid = np.array(list(set(dilated_voxels)))
    return voxel_grid


def _dilate_voxels_from_idx(voxel_idx: np.ndarray, grid_size: int = 64) -> np.ndarray:
    """
    Dilate occupied voxels by 1 in all 26 directions (Chebyshev radius=1).
    voxel_idx: (N,3) int32 array of occupied voxel indices
    Returns (M,3) unique dilated voxel indices
    """
    dilated_voxels = set()
    directions = [np.array(d, dtype=np.int32) for d in itertools.product([-1,0,1], repeat=3) if d != (0,0,0)]
    for v in voxel_idx:
        v = v.astype(np.int32)
        dilated_voxels.add(tuple(v))
        for d in directions:
            neighbor = v + d
            if np.all((0 <= neighbor) & (neighbor < grid_size)):
                dilated_voxels.add(tuple(neighbor))
    return np.array(list(dilated_voxels), dtype=np.int32)

# def _dilate_voxels_idx(voxel_grid: o3d.geometry.VoxelGrid) -> np.ndarray:
#     # voxel_grid = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
#     # densify voxel grid
#     dilated_voxels = set()
#     directions = [d for d in itertools.product([-1, 0, 1], repeat=3) if d != (0, 0, 0)]
#     for v in voxel_grid:
#         dilated_voxels.add(tuple(v))
#         for d in directions:
#             neighbor = tuple(v + np.array(d))
#             if all(0 <= n < 64 for n in neighbor):
#                 dilated_voxels.add(neighbor)
#     voxel_grid = np.array(list(set(dilated_voxels)))
#     return voxel_grid
OFFSETS = np.array([o for o in itertools.product([-1,0,1], repeat=3) if o != (0,0,0)],
                   dtype=np.int16)
def _dilate_voxels_idx(vox, G=64):
    vox = np.asarray(vox, dtype=np.int16)          # (K,3)
    nbrs = (vox[:, None, :] + OFFSETS[None, :, :]).reshape(-1, 3)  # (K*26,3)
    valid = (nbrs >= 0).all(1) & (nbrs < G).all(1)
    both = np.vstack([vox, nbrs[valid]])
    out = np.unique(both, axis=0).astype(np.int32)
    return out
def _project_and_sample_dino(
    voxel_world: torch.Tensor,           # (M,3) world coords, torch float
    T_wc: np.ndarray,                    # (4,4) W->C for this view
    K: np.ndarray,                       # (3,3) intrinsics for this view's image
    img_hw: tuple,                       # (H,W) of the RGB used for DINO
    dino_in_hw: tuple,                   # (H,W) input size fed to DINO
    dino_tokens: torch.Tensor,           # (1,1024,n,n)
) -> tuple:
    """
    Returns: idx_keep (np.int64[M_kept]), normalized grid coords (np.float32[M_kept,2])
    """
    H_rgb, W_rgb = img_hw
    # project
    uv = utils3d.torch.project_cv(
        voxel_world.float(),
        torch.from_numpy(T_wc).float()[None, ...],
        torch.from_numpy(K).float()[None, ...]
    )[0]  # (M,2)
    uv = uv.squeeze(0)
    u, v = uv[:,0], uv[:,1]
    inb_img = (u >= 0) & (u < W_rgb) & (v >= 0) & (v < H_rgb)

    if inb_img.sum() == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

    uv = uv[inb_img]
    idx_all = np.arange(voxel_world.shape[0])
    idx_img = idx_all[inb_img.cpu().numpy()]

    # map to DINO input space (e.g., 518 for DINOv3 ViT-L/16)
    H_in, W_in = dino_in_hw
    sx = float(W_in) / float(W_rgb)
    sy = float(H_in) / float(H_rgb)
    u_in = uv[:,0] * sx
    v_in = uv[:,1] * sy

    H_p, W_p = dino_tokens.shape[-2], dino_tokens.shape[-1]
    gx = 2.0 * (u_in * (W_p / float(W_in)) + 0.5) / float(W_p) - 1.0
    gy = 2.0 * (v_in * (H_p / float(H_in)) + 0.5) / float(H_p) - 1.0

    inb_grid = (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)
    if inb_grid.sum() == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

    gx = gx[inb_grid]; gy = gy[inb_grid]
    idx_keep = idx_img[inb_grid.cpu().numpy()]        # indices into original M

    grid = torch.stack([gx, gy], dim=-1)
    grid_np = grid.detach().cpu().numpy().astype(np.float32)
    return idx_keep.astype(np.int64), grid_np

def _normalize_segmented_mesh(segmented_mesh: o3d.geometry.TriangleMesh):
    vertices = np.asarray(segmented_mesh.vertices)
    mean = vertices.mean(axis=0)
    vertices -= mean
    # scale = np.max(np.abs(vertices), axis=0)
    scale = np.max(np.abs(vertices))
    # scale[scale == 0] = 1.0
    vertices *= 1.0 / (2 * scale)
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    segmented_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    scale = np.array([scale, scale, scale])
    return mean, scale

@torch.no_grad()
def voxelise_features(
    obj_data: Dict[str, str],
    scan_id: str,
    mode: str = "gs_annotations",
    grid_size: int = 128,
) -> None:
    """
    Voxelise features for scan using Option 1:
    - Voxelize mesh once in mesh-canonical.
    - For each frame, rotate centers by ΔR and re-index (no re-voxelization).
    - Seeds are indexed in the same fixed canonical box (mu=0, isotropic s).
    """
    G = int(grid_size)
    scenes_dir = osp.join(root_dir, "scenes")
    frame_idxs = scan3r.load_frame_idxs(data_dir=scenes_dir, scan_id=scan_id)
    if len(frame_idxs) > 100:
        frame_idxs = frame_idxs[:100]
        # frame_idxs = ['000024']

    # --- calib (shared) ---
    extrinsics_all = scan3r.load_frame_poses(data_dir=root_dir, scan_id=scan_id, frame_idxs=frame_idxs)
    intrinsics = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id)
    K_rgb = intrinsics["intrinsic_mat"]
    K_depth = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id, type='depth')["intrinsic_mat"]

    # --- scene mesh & segmentation (once) ---
    mesh = scan3r.load_ply_mesh(
        data_dir=scenes_dir, scan_id=scan_id, label_file_name="labels.instances.annotated.v2.ply"
    )
    annos = scan3r.load_ply_data(
        data_dir=scenes_dir, scan_id=scan_id, label_file_name="labels.instances.annotated.v2.ply"
    )["vertex"]["objectId"]
    object_ids = [int(obj["id"]) for obj in obj_data["objects"]]
    object_vertex_mask = np.isin(annos, object_ids)
    selected_vertices = np.where(object_vertex_mask)[0]
    faces = np.asarray(mesh.triangles); vertices = np.asarray(mesh.vertices)
    face_mask = np.all(np.isin(faces, selected_vertices), axis=1)
    selected_faces = faces[face_mask]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_vertices)}
    remapped_faces = np.vectorize(index_map.get)(selected_faces)

    scene_mesh = o3d.geometry.TriangleMesh()
    scene_mesh.vertices = o3d.utility.Vector3dVector(vertices[selected_vertices])
    scene_mesh.triangles = o3d.utility.Vector3iVector(remapped_faces)
    vertex_obj_ids = annos[selected_vertices].astype(np.int32)
    # ---------- OUTPUT DIR ----------
    scene_output_dir = osp.join(
        args.model_dir,
        "files",
        mode,
        scan_id,
        f"scene_level_structure_no_dilation_{G}",
    )
    os.makedirs(scene_output_dir, exist_ok=True)

    # Pre-compute the canonical GT voxels once per scene so they can be reused by
    # every frame pack. This keeps each pack small and avoids re-voxelizing.
    scene_meta_path = osp.join(scene_output_dir, "scene_occ_meta.npz")
    if args.override or not osp.isfile(scene_meta_path):
        scene_mesh_canonical = copy.deepcopy(scene_mesh)
        mean_gt, scale_gt = _normalize_segmented_mesh(scene_mesh_canonical)
        pcd = scene_mesh_canonical.sample_points_uniformly(number_of_points=400000)
        # voxel_grid_canonical = voxelize_mesh_simple_dense(
        #     scene_mesh_canonical,
        #     G=G,
        #     n_rand=0,
        #     k_dilate=2,
        #     device="cuda" if torch.cuda.is_available() else "cpu",
        # )
        voxel_grid_canonical = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd,
            voxel_size=1/G,
            min_bound=(-0.5, -0.5, -0.5),
            max_bound=(0.5, 0.5, 0.5)
        )
        voxel_grid_canonical = np.array([voxel.grid_index for voxel in voxel_grid_canonical.get_voxels()])
        vox_idx_gt_occ = voxel_grid_canonical.astype(np.int32)

        # Compute per-voxel object IDs by voting from mesh vertices in canonical space.
        # vertices of scene_mesh_canonical are already normalized to [-0.5, 0.5].
        vertices_can = np.asarray(scene_mesh_canonical.vertices, dtype=np.float32)
        vert_idx_can = np.clip(
            np.floor((vertices_can + 0.5 - 1e-6) * G).astype(np.int32), 0, G - 1
        )
        lin_vert = (
            vert_idx_can[:, 0] * G * G
            + vert_idx_can[:, 1] * G
            + vert_idx_can[:, 2]
        )
        lin_vox = (
            vox_idx_gt_occ[:, 0] * G * G
            + vox_idx_gt_occ[:, 1] * G
            + vox_idx_gt_occ[:, 2]
        )
        lin_to_voxpos = np.full(G * G * G, -1, dtype=np.int32)
        lin_to_voxpos[lin_vox] = np.arange(len(lin_vox), dtype=np.int32)
        vert_voxpos = lin_to_voxpos[lin_vert]
        valid = vert_voxpos >= 0
        vox_obj_vote: defaultdict = defaultdict(Counter)
        for vp, oid in zip(vert_voxpos[valid].tolist(), vertex_obj_ids[valid].tolist()):
            vox_obj_vote[vp][oid] += 1
        vox_idx_gt_obj_ids = np.zeros(len(vox_idx_gt_occ), dtype=np.int32)
        for vp, counter in vox_obj_vote.items():
            vox_idx_gt_obj_ids[vp] = max(counter, key=counter.get)

        if not args.dry_run:
            np.savez(
                scene_meta_path,
                G=np.int32(G),
                mean_gt=mean_gt.astype(np.float32),
                scale_gt=scale_gt.astype(np.float32),
                vox_idx_gt_occ=vox_idx_gt_occ,
                vox_idx_gt_obj_ids=vox_idx_gt_obj_ids,
            )
            _LOGGER.info(
                f"[{scan_id}] Saved canonical occupancy meta → {scene_meta_path}"
            )
    else:
        meta_npz = np.load(scene_meta_path, allow_pickle=False)
        mean_gt = meta_npz["mean_gt"].astype(np.float32)
        scale_gt = meta_npz["scale_gt"].astype(np.float32)
        vox_idx_gt_occ = meta_npz["vox_idx_gt_occ"].astype(np.int32)
        vox_idx_gt_obj_ids = meta_npz["vox_idx_gt_obj_ids"].astype(np.int32)

    # ---------- (B) PER-FRAME LOOP: ROTATE CENTERS & RE-INDEX; BUILD PACK ----------
    masks_all = scan3r.load_masks(data_dir=root_dir, scan_id=scan_id)
    object_ids_arr = np.array(object_ids, dtype=np.int32)
    for ref_fid in frame_idxs:
        # 1) seed lifting
        out_name = f"student_pack_aligned_{ref_fid}.npz"
        out_path = osp.join(scene_output_dir, out_name)
        if (
            osp.exists(out_path) and not args.override
        ):
            _LOGGER.info(f"Skipping {ref_fid} ")
            continue
        t0 = time.time()
        scene_mesh_preserve = copy.deepcopy(scene_mesh)
        Wp_world = unproject_frame_to_world(ref_fid, extrinsics_all, K_rgb, K_depth, root_dir, scan_id)  # (Ns,3)
        if Wp_world.shape[0] == 0:
            _LOGGER.warning(f"[{scan_id}:{ref_fid}] no lifted points; skipping.")
            continue
        _LOGGER.info(f"Seed lifting took {time.time()-t0:.2f}s"); t0 = time.time()
        # 2) per-frame canonical rotation from seed
        Rg_frame = align_gravity_with_plane(Wp_world)
        Rz_frame = yaw_canonicalize_xy((Rg_frame @ Wp_world.T).T)
        # R_frame = Rz_frame @ Rg_frame
        _LOGGER.info(f"Frame canonicalization took {time.time()-t0:.2f}s"); t0 = time.time()

        # ------not doing canonicalization
        R_frame = np.eye(3, dtype=np.float32)
        
        V_world = np.asarray(scene_mesh_preserve.vertices, dtype=np.float32)
        V_can = (R_frame @ V_world.T).T
        scene_mesh_preserve.vertices = o3d.utility.Vector3dVector(V_can)
        mean = mean_gt
        scale = scale_gt
        _LOGGER.info(f"Dilation of mesh took {time.time()-t0:.2f}s"); t0 = time.time()
        # 3) re-index the pre-voxelized mesh by ΔR = R_frame @ R_mesh.T
        Wp = (R_frame @ Wp_world.T).T
        mean_seed = Wp.mean(axis=0).astype(np.float32)                      # record only (not used for indexing)
        ctr = (Wp - mean_seed[None, :]).astype(np.float32)
        base = float(np.max(np.abs(ctr)))
        scale_seed = max(base, 1e-6)

        # x_norm = (Wp - mean[None, :]) / (2.0 * scale)
        x_norm = (Wp - mean_seed[None, :]) / (2.0 * scale_seed)
        seed_idx = np.floor((x_norm + 0.5) * G).astype(np.int32)
        seed_idx = np.clip(seed_idx, 0, G - 1)
        seed_idx = np.unique(seed_idx, axis = 0)
        # seed_idx = _dilate_voxels_from_idx(seed_idx, grid_size=G)
        # seed_idx = _dilate_voxels_idx(seed_idx)
        _LOGGER.info(f"Dilation of seed took {time.time()-t0:.2f}s"); t0 = time.time()
        # _LOGGER.info(f"Dilation took {time.time()-t0:.2f}s"); t0 = time.time()
        centers_normed = (seed_idx.astype(np.float32) + 0.5) / G - 0.5
        # voxel_world = centers_normed * (2.0 * scale) + mean[None, :]
        voxel_world = centers_normed * (2.0 * scale_seed) + mean_seed[None, :]
        voxel_world  = (R_frame.T @ voxel_world.T).T
        # project & sample from ref image
        rgb_ref_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{ref_fid}.color.jpg"
        with Image.open(rgb_ref_path) as img:
            rgb_ref = np.asarray(img.convert("RGB"), dtype=np.uint8)

        frame_tokens, _sizes = _get_dino_embedding(_prep_image_for_dino(rgb_ref))
        _LOGGER.info(f"Dino embedding took {time.time()-t0:.2f}s"); t0 = time.time()
        dino_tokens_path = osp.join(scene_output_dir, f"dino_tokens_{ref_fid}.npy")
        if args.override or not osp.exists(dino_tokens_path):
            np.save(
                dino_tokens_path,
                frame_tokens.squeeze(0).detach().cpu().numpy().astype(np.float16),
            )
        T_wc_ref = np.linalg.inv(extrinsics_all[ref_fid]).astype(np.float32)
        idx_keep, sample_grid = _project_and_sample_dino(
            voxel_world=torch.from_numpy(voxel_world).float(),
            T_wc=T_wc_ref,
            K=K_rgb,
            img_hw=rgb_ref.shape[:2],
            dino_in_hw=_sizes[:2],
            dino_tokens=frame_tokens,
        )
        _LOGGER.info(f"Project and sample took {time.time()-t0:.2f}s"); t0 = time.time()
        seed_idx_pcd = seed_idx[idx_keep]           # (M_keep,3)
        if args.visualize:
            payload = torch.concatenate(
                [torch.tensor(seed_idx_pcd, dtype=torch.float32),
                torch.tensor(sample_grid, dtype=torch.float32)], dim=1
            )  # (N, 1027)
            vis.save_voxel_as_ply(
                payload.cpu().numpy(),
                f"vis/{scan_id}_debug.ply",
                show_color=True,
            )

        # Determine which scene objects are visible in this frame from the 2D mask.
        # Fall back to all scene objects if the mask is missing for this frame.
        frame_mask = masks_all.get(ref_fid)
        if frame_mask is not None:
            visible_obj_ids = np.intersect1d(
                np.unique(frame_mask).astype(np.int32), object_ids_arr
            ).astype(np.int32)
        else:
            _LOGGER.warning(f"[{scan_id}:{ref_fid}] No 2D mask found; using all object IDs.")
            visible_obj_ids = object_ids_arr.copy()

        # 6) PACK: fixed canonical box (mu=0, scale=s_iso) + per-frame rotation/scene meta
        pack = {
            "G": np.int32(G),
            "R_frame": R_frame.astype(np.float32),     # current frame canonical
            "mean_gt": mean.astype(np.float32),
            "scale_gt": scale.astype(np.float32),
            "seed_idx": seed_idx_pcd.astype(np.int32),
            "sample_grid": sample_grid.astype(np.float32),
            "seed_box_init_mean": mean_seed.astype(np.float32),       # for logging
            "seed_box_init_scale": np.float32(scale_seed),            # for logging
            "frame_id_used": np.array(ref_fid),
            "pad_meta": np.array([-1, 1], dtype=np.float32),
            "visible_obj_ids": visible_obj_ids,
        }

        out_name = f"student_pack_aligned_{ref_fid}.npz"
        out_path = osp.join(scene_output_dir, out_name)
        if not args.dry_run:
            np.savez(out_path, **pack)
            _LOGGER.info(f"[{scan_id}:{ref_fid}] Saved structure pack → {out_path}")

    # # Also save the fixed canonical box (useful for downstream)
    # if not args.dry_run:
    #     mean_scale_path = osp.join(scene_output_dir, "mean_scale_dense.npz")
    #     np.savez(mean_scale_path,
    #              mean=np.zeros(3, dtype=np.float32),
    #              scale=np.array([s_iso, s_iso, s_iso], dtype=np.float32),
    #              mu_scene=mu_scene.astype(np.float32),
    #              R_mesh=R_mesh.astype(np.float32))
    #     _LOGGER.info(f"[{scan_id}] Saved canonical box/meta → {mean_scale_path}")

def process_data(
    cfg: Config, mode: str = "gs_annotations", split: str = "train"
) -> np.ndarray:
    """
    Process scans to get featured voxel representation.

    Args:
        cfg: Configuration object.
        mode (str, optional): Mode to run subscan generation on. Defaults to "gs_annotations".
        split (str, optional): Split to run subscan generation on. Defaults to "train".

    Returns:
        np.ndarray: processed subscan IDs.
    """

    scan_type = cfg.autoencoder.encoder.scan_type
    encoder_cfg = getattr(cfg.autoencoder, "encoder", None)
    grid_size = getattr(encoder_cfg, "resolution", 128)
    resplit = "resplit_" if cfg.data.resplit else ""
    scan_ids_filename = (
        f"{split}_{resplit}scans.txt"
        if scan_type == "scan"
        else f"{split}_scans_subscenes.txt"
    )
    objects_info_file = osp.join(root_dir, "files", "objects.json")
    all_obj_info = common.load_json(objects_info_file)

    subscan_ids_generated = np.genfromtxt(
        osp.join(root_dir, "files", scan_ids_filename), dtype=str
    )
    subscan_ids_processed = []

    subRescan_ids_generated = {}
    scans_dir = cfg.data.root_dir
    scans_files_dir = osp.join(scans_dir, "files")

    all_scan_data = common.load_json(osp.join(scans_files_dir, "3RScan.json"))

    for scan_data in all_scan_data:
        ref_scan_id = scan_data["reference"]
        if ref_scan_id in subscan_ids_generated:
            rescan_ids = [scan["reference"] for scan in scan_data["scans"]]
            subRescan_ids_generated[ref_scan_id] = [ref_scan_id] + rescan_ids

    subscan_ids_generated = subRescan_ids_generated
    all_subscan_ids = [
        subscan_id
        for scan_id in subscan_ids_generated
        for subscan_id in subscan_ids_generated[scan_id]
    ]
    all_subscan_ids = all_subscan_ids[200:][::2]
    # all_subscan_ids = ['bf9a3df1-45a5-2e80-8198-0652e415e289','1dd720a1-2ba0-22d9-8b6e-bb00c888a414','10b17967-3938-2467-88c5-a299519f9ad7','1d234014-e280-2b1a-8eab-fe44989693aa','1d2f8518-d757-207c-8d4a-b2f43254c68f',
    #                    'ea31825e-0a4c-2749-91f9-1cc45973a0f6','20c993a1-698f-29c5-8716-5a937fdd879a','42384916-60a7-271e-9c1f-6722abc6495d','20c993af-698f-29c5-84b2-972451f94cfb']

    for subscan_id in tqdm(all_subscan_ids):
        obj_data = next(
            obj_data
            for obj_data in all_obj_info["scans"]
            if obj_data["scan"] == subscan_id
        )

        voxelise_features(
            mode=mode,
            obj_data=obj_data,
            scan_id=subscan_id,
            grid_size=grid_size,
        )

        subscan_ids_processed.append(subscan_id)

    subscan_ids = np.array(subscan_ids_processed)
    return subscan_ids


def parse_args() -> Tuple[Namespace, list]:
    """
    Parse command line arguments.

    Returns:
        Tuple[argparse.Namespace, list]: Parsed arguments and unknown arguments.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
    )
    parser.add_argument(
        "--split",
        dest="split",
        default="train",
        type=str,
    )
    parser.add_argument("--model_dir", type=str, default="")
    # parser.add_argument("--model", type=str, default="dinov3_vitl16")
    parser.add_argument("--model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_dir", type=str, default="vis")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--override", action="store_true")
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == "__main__":
    common.init_log(level=logging.INFO)
    _LOGGER.info("**** Starting feature voxelisation for 3RScan ****")
    args, unknown = parse_args()
    os.makedirs(args.vis_dir, exist_ok=True)
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    root_dir = cfg.data.root_dir

    # model = torch.hub.load("/cluster/home/wangyih/.cache/torch/hub/facebookresearch_dinov3_main/", args.model, source='local', pretrained=False)
    # ckpt_path = "/cluster/home/wangyih/.cache/torch/hub/facebookresearch_dinov3_main/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    # ckpt = torch.load(ckpt_path, map_location="cpu")
    # state_dict = ckpt.get("model", ckpt)
    # model.load_state_dict(state_dict, strict=False)

    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model.eval().cuda()
    transform = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    scan_ids = process_data(cfg, mode="gs_annotations", split=args.split)
