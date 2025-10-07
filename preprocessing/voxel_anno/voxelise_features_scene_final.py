import itertools
import logging
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

def _dilate_voxels_idx(voxel_grid: o3d.geometry.VoxelGrid) -> np.ndarray:
    # voxel_grid = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
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

def _project_and_sample_dino(
    voxel_world: torch.Tensor,           # (M,3) world coords, torch float
    T_wc: np.ndarray,                    # (4,4) W->C for this view
    K: np.ndarray,                       # (3,3) intrinsics for this view's image
    img_hw: tuple,                       # (H,W) of the RGB used for DINO
    dino_tokens: torch.Tensor,           # (1,1024,n,n)
) -> tuple:
    """
    Returns: idx_keep (np.int64[M_kept]), tokens (np.float16[M_kept,1024])
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
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 1024), dtype=np.float16)

    uv = uv[inb_img]
    idx_all = np.arange(voxel_world.shape[0])
    idx_img = idx_all[inb_img.cpu().numpy()]

    # map to 518 feature space used by your backbone
    sx = 518.0 / float(W_rgb)
    sy = 518.0 / float(H_rgb)
    u518 = uv[:,0] * sx
    v518 = uv[:,1] * sy

    gx = 2.0 * (u518 + 0.5) / 518.0 - 1.0
    gy = 2.0 * (v518 + 0.5) / 518.0 - 1.0

    inb_grid = (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)
    if inb_grid.sum() == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 1024), dtype=np.float16)

    gx = gx[inb_grid]; gy = gy[inb_grid]
    idx_keep = idx_img[inb_grid.cpu().numpy()]        # indices into original M

    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2).to(dino_tokens.device)
    feat = F.grid_sample(
        dino_tokens.float(),   # (1,1024,n,n)
        grid.float(),          # (1,M_kept,1,2)
        mode="bilinear",
        align_corners=False
    ).squeeze(-1).permute(0,2,1).contiguous()         # (1,M_kept,1024)

    tokens = feat[0].detach().cpu().numpy().astype(np.float16)
    return idx_keep.astype(np.int64), tokens

def _normalize_segmented_mesh(segmented_mesh: o3d.geometry.TriangleMesh):
    vertices = np.asarray(segmented_mesh.vertices)
    mean = vertices.mean(axis=0)
    vertices -= mean
    scale = np.max(np.abs(vertices))
    # scale = np.max(np.abs(vertices), axis=0) 
    # scale[scale == 0] = 1.0
    vertices *= 1.0 / (2 * scale)
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    segmented_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mean, scale

# @torch.no_grad()
# def voxelise_features(
#     obj_data: Dict[str, str],
#     scan_id: str,
#     mode: str = "gs_annotations",
# ) -> None:
#     """
#     Voxelise features for scan.

#     Args:
#         obj_data (Dict[str, str]): Object data.
#         scan_id (str): Scan ID.
#         mode (str, optional): Mode to run subscan generation on. Defaults to "gs_annotations".
#     """

#     scenes_dir = osp.join(root_dir, "scenes")
#     frame_idxs = scan3r.load_frame_idxs(data_dir=scenes_dir, scan_id=scan_id)
#     if len(frame_idxs)> 100:
#         frame_idxs = frame_idxs[:100]

#     extrinsics = scan3r.load_frame_poses(
#         data_dir=root_dir, scan_id=scan_id, frame_idxs=frame_idxs
#     )
#     intrinsics = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id)
#     K_rgb = intrinsics["intrinsic_mat"]
#     K_depth = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id, type='depth')["intrinsic_mat"]
    
#     mask = scan3r.load_masks(data_dir=root_dir, scan_id=scan_id)
        
#     mesh = scan3r.load_ply_mesh(
#         data_dir=scenes_dir,
#         scan_id=scan_id,
#         label_file_name="labels.instances.annotated.v2.ply",
#     )
#     annos = scan3r.load_ply_data(
#         data_dir=scenes_dir,
#         scan_id=scan_id,
#         label_file_name="labels.instances.annotated.v2.ply",
#     )["vertex"]["objectId"]
#     object_ids = [int(obj["id"]) for obj in obj_data["objects"]]
#     print("object_ids: ", object_ids)
    
#     scene_output_dir = osp.join(args.model_dir, "files", mode, scan_id, "scene_level_structure")
#     voxel_path = osp.join(scene_output_dir, "voxel_output_dense.npz")
#     mean_scale_path=osp.join(scene_output_dir, "mean_scale_dense.npz")
#     if (
#             osp.exists(mean_scale_path)
#             and osp.exists(voxel_path)
#             and "arr_0" in np.load(voxel_path)
#             and not args.override
#         ):
#             _LOGGER.info(f"Skipping {scan_id} ")
#             return
#     try:
        
#         # STEP 1: Segment the mesh

#         object_vertex_mask = np.isin(annos, object_ids) 
#         selected_vertices = np.where(object_vertex_mask)[0]
#         faces = np.asarray(mesh.triangles)
#         vertices = np.asarray(mesh.vertices)
#         face_mask = np.all(np.isin(faces, selected_vertices), axis=1)
#         selected_faces = faces[face_mask]
#         index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_vertices)}
#         remapped_faces = np.vectorize(index_map.get)(selected_faces)
#         # Create the mesh with only object geometry
#         scene_mesh = o3d.geometry.TriangleMesh()
#         scene_mesh.vertices = o3d.utility.Vector3dVector(vertices[selected_vertices])
#         scene_mesh.triangles = o3d.utility.Vector3iVector(remapped_faces)
#         for ref_fid in frame_idxs:
#             Wp_world = unproject_frame_to_world(ref_fid, extrinsics, K_rgb, K_depth, root_dir, scan_id)  # (Ns,3)
#             # Canonicalization from seed
#             # V_world = np.asarray(scene_mesh.vertices, dtype=np.float32)
#             Rg = align_gravity_with_plane(Wp_world)
#             Rz = yaw_canonicalize_xy((Rg @ Wp_world.T).T)
#             R_canonicalize = Rz @ Rg
#             V_world = np.asarray(scene_mesh.vertices, dtype=np.float32)
#             V_can = (R_canonicalize @ V_world.T).T
#             scene_mesh.vertices = o3d.utility.Vector3dVector(V_can)
            
#             # STEP 2: Normalize to unit cube (-0.5, 0.5)
            
#             mean, scale = _normalize_segmented_mesh(scene_mesh)
#             # STEP 3: Voxelise the mesh
#             voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
#                 scene_mesh,
#                 1 / 64,
#                 min_bound=(-0.5, -0.5, -0.5),
#                 max_bound=(0.5, 0.5, 0.5),
#             )
#             voxel_grid = _dilate_voxels(voxel_grid)

#             # STEP 4: Save mean and scale (Scene composition)
#             if not args.dry_run:
#                 os.makedirs(os.path.dirname(mean_scale_path), exist_ok=True)
#                 np.savez(mean_scale_path, mean=mean, scale=scale)
#                 _LOGGER.info(f"Saved mean and scale to {mean_scale_path}")

#             # STEP 5: Render the object
#             pose_camera_to_world = [
#                 np.linalg.inv(extrinsics[frame_idx]) for frame_idx in extrinsics
#             ]

#             masks = [mask[frame_id] for frame_id in frame_idxs]
#             for i, frame_id in enumerate(frame_idxs[:3]):
#                 print(f"Mask {i} (frame {frame_id}) unique labels:", np.unique(masks[i]))
#             # masks = [np.where(mask > 0, 1, 0) for mask in masks]
#             masks =  [np.isin(mask, object_ids).astype(np.uint8) for mask in masks]
#             G = 64
#             vox_idx_allowed = voxel_grid.astype(np.int32)
#             vox_idx_gt_occ = vox_idx_allowed.copy() 
#             ref_fid = frame_idxs[0]
#             K_rgb = intrinsics['intrinsic_mat']
#             K_depth = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id, type = 'depth')["intrinsic_mat"]
#             T_wc_ref = np.linalg.inv(extrinsics[ref_fid]).astype(np.float32) 
#             Wp = (R_canonicalize @ Wp_world.T).T
            
#             # Prepare init seed space
#             mean_seed = Wp.mean(axis=0).astype(np.float32)
#             ctr = (Wp - mean_seed[None, :]).astype(np.float32)
#             base = float(np.max(np.abs(ctr)))
#             scale_seed = max(base, 1e-6)
            
#             # x_norm = (Wp - mean[None, :]) / (2.0 * scale[None,:])
#             x_norm = (Wp - mean[None, :]) / (2.0 * scale)
#             seed_idx = np.floor((x_norm + 0.5) * G).astype(np.int32)
#             seed_idx = np.clip(seed_idx, 0, G - 1)
#             seed_idx = _dilate_voxels_from_idx(seed_idx, grid_size=G)
#             centers_normed = (seed_idx.astype(np.float32) + 0.5) / G - 0.5
#             voxel_world = centers_normed * (2.0 * scale) + mean[None, :]
#             voxel_world  = (R_canonicalize.T @ voxel_world.T).T
            
#             rgb_ref_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{ref_fid}.color.jpg"
#             with Image.open(rgb_ref_path) as img:
#                 rgb_ref = np.asarray(img.convert("RGB"), dtype=np.uint8)
#             frame_tokens, (H_in,W_in,H_p,W_p) = _get_dino_embedding(_prep_image_for_dino(rgb_ref))   # (1,1024,Hp,Wp)
#             idx_keep, tokens = _project_and_sample_dino(
#                 voxel_world=torch.from_numpy(voxel_world).float(),  # (Ms0,3) world
#                 T_wc=T_wc_ref, K=K_rgb, img_hw=rgb_ref.shape[:2],
#                 dino_tokens=frame_tokens
#             )
#             seed_idx_pcd = seed_idx[idx_keep]
#             feats    = tokens.astype(np.float32)              # (Ms,1024) float

#             pack = {
#                 "G": np.int32(G),
#                 "R_canonicalize": R_canonicalize,
#                 "mean": mean.astype(np.float32),     # canonical for TRAIN now = PCD
#                 "scale": scale.astype(np.float32),
#                 "vox_idx_gt_occ": vox_idx_gt_occ.astype(np.int32),  # target on PCD grid
#                 "seed_idx": seed_idx_pcd.astype(np.int32),              # conditioning on PCD grid
#                 "feats": feats.astype(np.float32),                      # 1:1 with seeds
#                 "seed_box_init_mean": mean_seed.astype(np.float32),
#                 "seed_box_init_scale": np.float32(scale_seed),
#                 "frame_id_used": np.array(ref_fid),
#                 "pad_meta": np.array([-1, 1], dtype=np.float32),
#                 }

#             # Filename convention consistent with the rest of your code:
#             out_name = f"student_pack_aligned_{ref_fid}.npz"
#             out_path = osp.join(scene_output_dir, out_name)
#             if not args.dry_run:
#                 os.makedirs(scene_output_dir, exist_ok=True)
#                 np.savez(out_path, **pack)
#                 _LOGGER.info(f"Saved structure pack → {out_path}")
#     except (FileNotFoundError, RuntimeError, ValueError) as e:
#         _LOGGER.exception(f"Error processing {scan_id} : {e}")
@torch.no_grad()
def voxelise_features(
    obj_data: Dict[str, str],
    scan_id: str,
    mode: str = "gs_annotations",
) -> None:
    """
    Voxelise features for scan using Option 1:
    - Voxelize mesh once in mesh-canonical.
    - For each frame, rotate centers by ΔR and re-index (no re-voxelization).
    - Seeds are indexed in the same fixed canonical box (mu=0, isotropic s).
    """
    G = 64
    scenes_dir = osp.join(root_dir, "scenes")
    frame_idxs = scan3r.load_frame_idxs(data_dir=scenes_dir, scan_id=scan_id)
    if len(frame_idxs) > 150:
        frame_idxs = frame_idxs[:150]
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
    # ---------- OUTPUT DIR ----------
    scene_output_dir = osp.join(args.model_dir, "files", mode, scan_id, "scene_level_structure_uncan")
    os.makedirs(scene_output_dir, exist_ok=True)

    # ---------- (B) PER-FRAME LOOP: ROTATE CENTERS & RE-INDEX; BUILD PACK ----------
    for ref_fid in frame_idxs:
        # 1) seed lifting
        scene_mesh_preserve = copy.deepcopy(scene_mesh)
        Wp_world = unproject_frame_to_world(ref_fid, extrinsics_all, K_rgb, K_depth, root_dir, scan_id)  # (Ns,3)
        if Wp_world.shape[0] == 0:
            _LOGGER.warning(f"[{scan_id}:{ref_fid}] no lifted points; skipping.")
            continue

        # 2) per-frame canonical rotation from seed
        Rg_frame = align_gravity_with_plane(Wp_world)
        Rz_frame = yaw_canonicalize_xy((Rg_frame @ Wp_world.T).T)
        R_frame = Rz_frame @ Rg_frame

        # ------not doing canonicalization
        R_frame = np.eye(3, dtype=np.float32)
        
        V_world = np.asarray(scene_mesh_preserve.vertices, dtype=np.float32)
        V_can = (R_frame @ V_world.T).T
        scene_mesh_preserve.vertices = o3d.utility.Vector3dVector(V_can)
        mean, scale = _normalize_segmented_mesh(scene_mesh_preserve)
        V_norm = np.asarray(scene_mesh_preserve.vertices, dtype=np.float32)
        idx = np.floor((V_norm + 0.5) * G).astype(np.int32)
        idx = np.clip(idx, 0, G-1)
        idx = np.unique(idx, axis=0) 
        voxel_grid = idx
        voxel_grid = _dilate_voxels_idx(voxel_grid)
        vox_idx_allowed = voxel_grid.astype(np.int32)
        vox_idx_gt_occ = vox_idx_allowed.copy() 
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
        seed_idx = _dilate_voxels_idx(seed_idx)
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
        T_wc_ref = np.linalg.inv(extrinsics_all[ref_fid]).astype(np.float32)
        idx_keep, tokens = _project_and_sample_dino(
            voxel_world=torch.from_numpy(voxel_world).float(),
            T_wc=T_wc_ref, K=K_rgb, img_hw=rgb_ref.shape[:2], dino_tokens=frame_tokens
        )
        seed_idx_pcd = seed_idx[idx_keep]           # (M_keep,3)
        feats = tokens.astype(np.float32)           # (M_keep,1024)
        if args.visualize:
            payload = torch.concatenate(
                [torch.tensor(seed_idx_pcd, dtype=torch.float32),
                torch.tensor(feats, dtype=torch.float32)], dim=1
            )  # (N, 1027)
            vis.save_voxel_as_ply(
                payload.cpu().numpy(),
                f"vis/{scan_id}_debug.ply",
                show_color=True,
            )
        
        # 6) PACK: fixed canonical box (mu=0, scale=s_iso) + per-frame rotation/scene meta
        pack = {
            "G": np.int32(G),
            "R_frame": R_frame.astype(np.float32),     # current frame canonical
            "mean_gt": mean.astype(np.float32),
            "scale_gt": scale.astype(np.float32),
            "vox_idx_gt_occ": vox_idx_gt_occ.astype(np.int32),
            "seed_idx": seed_idx_pcd.astype(np.int32),
            "feats": feats.astype(np.float32),
            "seed_box_init_mean": mean_seed.astype(np.float32),       # for logging
            "seed_box_init_scale": np.float32(scale_seed),            # for logging
            "frame_id_used": np.array(ref_fid),
            "pad_meta": np.array([-1, 1], dtype=np.float32),
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
    # all_subscan_ids = all_subscan_ids[0:2]
    # all_subscan_ids = ['fcf66d9e-622d-291c-84c2-bb23dfe31327']

    # all_subscan_ids = ["fcf66d9e-622d-291c-84c2-bb23dfe31327", "fcf66d8a-622d-291c-8429-0e1109c6bb26", "e44d238c-52a2-2879-89d9-a29ba04436e0"]
    all_subscan_ids = ["fcf66d88-622d-291c-871f-699b2d063630"]


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
    parser.add_argument("--model", type=str, default="dinov3_vitl16")
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

    model = torch.hub.load("/home/yihan/.cache/torch/hub/facebookresearch_dinov3_main", args.model, source='local', pretrained=False)
    ckpt_path = "/home/yihan/.cache/torch/hub/facebookresearch_dinov3_main/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval().cuda()
    transform = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    scan_ids = process_data(cfg, mode="gs_annotations", split=args.split)
