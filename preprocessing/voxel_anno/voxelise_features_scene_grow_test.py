import itertools
import logging
import os, glob, re
import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple

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
from utils.pcd_alignment import *
import cv2
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
# voxelize

_LOGGER = logging.getLogger(__name__)

def _get_dino_embedding(images: torch.Tensor) -> torch.Tensor:
    images = images.reshape(-1, 3, images.shape[-2], images.shape[-1]).cpu()
    inputs = transform(images).cuda()
    outputs = model(inputs, is_training=True)

    n_patch = 518 // 14
    bs = images.shape[0]
    patch_embeddings = (
        outputs["x_prenorm"][:, model.num_register_tokens + 1 :]
        .permute(0, 2, 1)
        .reshape(bs, 1024, n_patch, n_patch)
    )
    return patch_embeddings

def dilate_voxel_indices(vox_idx: np.ndarray, grid_size=64) -> np.ndarray:
    """
    Dilate a set of voxel indices (N,3) by 1 in 6/18/26-neighborhood.

    Args:
        vox_idx (np.ndarray): (N,3) integer voxel indices.
        grid_size (int): Grid dimension (default=64).

    Returns:
        np.ndarray: Dilated voxel indices (M,3), unique.
    """
    vox_set = set(map(tuple, vox_idx))
    dilated = set()
    # all 26 directions (3x3x3 minus center)
    directions = [d for d in itertools.product([-1,0,1], repeat=3) if d != (0,0,0)]
    for v in vox_set:
        dilated.add(v)
        for d in directions:
            n = (v[0]+d[0], v[1]+d[1], v[2]+d[2])
            if all(0 <= n[i] < grid_size for i in range(3)):
                dilated.add(n)
    return np.array(list(dilated), dtype=np.int32)

def _save_featured_voxel(
    voxel: torch.Tensor, output_file: str = "voxel_output_dense.npz"
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file, voxel.cpu().numpy())
    _LOGGER.info(f"Voxel saved to {output_file}")


def _normalize_points(points: np.ndarray):
    """
    points: (N,3) in world coords
    Returns mean (3,), scale (3,) and normalized points in [-0.5,0.5]^3 (clipped)
    """
    mean = points.mean(axis=0)
    pts = points - mean
    scale = np.max(np.abs(pts), axis=0)
    # scale = np.max(np.abs(pts))
    scale[scale == 0] = 1.0
    pts = pts / (2.0 * scale)
    pts = np.clip(pts, -0.5 + 1e-6, 0.5 - 1e-6)
    return mean, scale, pts

def _voxelize_points_normed(points_normed: np.ndarray, grid_size=64):
    """
    points_normed in [-0.5,0.5]. Returns unique voxel centers (M,3) in normalized coords.
    """
    # uv = (points_normed + 0.5).clip(0, 1 - 1e-8)
    # vox_idx = np.floor(uv * grid_size).astype(np.int32)
    # centers = (vox_idx.astype(np.float32) + 0.5) / grid_size - 0.5
    # return vox_idx, centers
    idx = np.floor((points_normed + 0.5) * grid_size).astype(np.int32)
    idx = np.clip(idx, 0, grid_size - 1)
    # unique occupied voxels
    idx = np.unique(idx, axis=0)
    # center coords back to normalized space
    centers = (idx.astype(np.float32) + 0.5) / grid_size - 0.5
    return idx, centers  # (M,3)

def to_homog_4x4(E):
    E = np.asarray(E)
    if E.shape == (4, 4):
        return E
    if E.shape == (3, 4):
        H = np.eye(4, dtype=E.dtype)
        H[:3, :4] = E
        return H

def _prep_image_for_dino(np_rgb_uint8: np.ndarray) -> torch.Tensor:
    # (H,W,3) uint8 -> (1,3,H,W) float in [0,1]
    t = torch.from_numpy(np_rgb_uint8.copy()).permute(2,0,1).float() / 255.0
    return t.unsqueeze(0).cuda()

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

def _l2_normalize(x: np.ndarray, eps=1e-6):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _lin(ijk, G):
    return (ijk[:,0]*G*G + ijk[:,1]*G + ijk[:,2]).astype(np.int64)

def make_idx_to_world_from_mean_scale(mean, scale, G):
    """
    Inverse of your normalization:
      pts_norm = (X - mean) / (2*scale) in [-0.5,0.5]^3  (per-axis scale)
      idx in [0, G-1] -> center at norm = (idx+0.5)/G - 0.5
      world = mean + norm * (2*scale)
    """
    mean  = np.asarray(mean,  dtype=np.float32)           # (3,)
    scale = np.asarray(scale, dtype=np.float32)           # (3,) per-axis
    Gf    = float(G)

    def idx_to_world(ix, iy, iz):
        norm = (np.array([ix, iy, iz], dtype=np.float32) + 0.5) / Gf - 0.5
        return mean + norm * (2.0 * scale)

    return idx_to_world


def filter_gen_by_ref_similarity(
    idx_keep_ref, tokens_ref, idx_keep_gen, tokens_gen,
    vox_idx, G, R_vox=1, tau=0.80, use_max_neighbor=True,
    mean=None,          # (3,)
    scale=None,         # (3,) per-axis
    world_points=None,  # (Nw,3) float32, lifted from REF view (in world coords)
    r_world_multiplier=3.0
):
    """
    Keep only GEN voxels whose features are similar enough to a REF voxel within R_vox
    AND (optionally) that have at least one neighbor in world_points.

    If world_points is provided, a GEN voxel is dropped when it has no nearby world point.
    World neighbor radius defaults to ~ R_vox * voxel_size * sqrt(3) (cube diagonal),
    scaled by r_world_multiplier.
    """
    
    if len(idx_keep_gen) == 0:
        return idx_keep_gen, tokens_gen

    # -----------------------------
    # 1) Similarity gate (voxel index space)
    # -----------------------------
    ref_coords = vox_idx[idx_keep_ref]                      # (Nr,3) int
    ref_feats  = tokens_ref.astype(np.float32, copy=False)  # (Nr,D)
    ref_norm   = ref_feats / (np.linalg.norm(ref_feats, axis=1, keepdims=True) + 1e-8)

    tree_ref = cKDTree(ref_coords.astype(np.float32))
    r_euclid = np.sqrt(3.0) * float(R_vox) + 1e-6           # upper bound for Chebyshev R

    gen_coords = vox_idx[idx_keep_gen]                      # (Ng,3)
    nb_ref     = tree_ref.query_ball_point(gen_coords.astype(np.float32), r=r_euclid)

    ref_pos = {int(i): p for p, i in enumerate(idx_keep_ref)}  # exact-match shortcut

    gen_feats = tokens_gen.astype(np.float32, copy=False)
    gen_norm  = gen_feats / (np.linalg.norm(gen_feats, axis=1, keepdims=True) + 1e-8)

    keep_sim = np.zeros(len(idx_keep_gen), dtype=bool)
    count = 0

    for g_idx, (v_id, nb_list) in enumerate(zip(idx_keep_gen.tolist(), nb_ref)):
        if v_id in ref_pos:
            rpos = ref_pos[v_id]
            sim  = float(np.dot(gen_norm[g_idx], ref_norm[rpos]))
            keep_sim[g_idx] = (sim >= tau)
            continue

        if len(nb_list) == 0:
            keep_sim[g_idx] = False
            count + 1
            continue

        if use_max_neighbor:
            sims = np.dot(ref_norm[nb_list], gen_norm[g_idx])
            sim  = float(np.max(sims))
        else:
            mu = ref_norm[nb_list].mean(axis=0)
            mu /= (np.linalg.norm(mu) + 1e-8)
            sim = float(np.dot(mu, gen_norm[g_idx]))

        keep_sim[g_idx] = (sim >= tau)
    print('no nb_list: ', count)

    # -----------------------------
    # 2) World-points gate (world coords), using your mean/scale/G
    # -----------------------------
    if world_points is not None:
        assert mean is not None and scale is not None and G is not None, \
            "Provide mean, scale (per-axis), and G to map vox_idx -> world."

        idx_to_world = make_idx_to_world_from_mean_scale(mean, scale, G)

        # Map GEN voxel centers -> world coords
        gen_world = np.asarray([idx_to_world(*uvw) for uvw in gen_coords], dtype=np.float32)

        # Per-axis voxel size in world units: Δ = (2*scale)/G
        voxel_size_axis = (2.0 * np.asarray(scale, dtype=np.float32)) / float(G)

        # Radius equivalent to R_vox in world space: L2 norm of per-axis steps
        # over an R_vox cube (diagonal in anisotropic world units)
        r_world = np.linalg.norm(voxel_size_axis * float(R_vox)) * float(r_world_multiplier) + 1e-6

        tree_wp = cKDTree(world_points.astype(np.float32))
        nb_wp   = tree_wp.query_ball_point(gen_world, r=r_world)
        keep_wp = np.fromiter((len(n) > 0 for n in nb_wp), dtype=bool, count=len(nb_wp))
        print("sim kept:", int(keep_sim.sum()))
        print("world kept:", int(keep_wp.sum()))
        print("both kept:", int((keep_sim & keep_wp).sum()))

        keep = keep_sim & keep_wp
    else:
        keep = keep_sim

    idx_filt = idx_keep_gen[keep]
    tok_filt = tokens_gen[keep]
    return idx_filt, tok_filt


def save_union_average(idx_keep_ref, tokens_ref, idx_keep_gen, tokens_gen, vox_idx, voxel_path):
    # Collect available views
    views = []
    if len(idx_keep_ref):
        views.append((idx_keep_ref.astype(np.int64), tokens_ref.astype(np.float32)))  # keep float32 for averaging
    if len(idx_keep_gen):
        views.append((idx_keep_gen.astype(np.int64), tokens_gen.astype(np.float32)))

    if not views:
        print("[WARN] No visible voxels in any view; not saving.")
        return

    # Union of voxel indices across views
    union_keys = sorted(set(int(i) for idx, _ in views for i in idx))
    key_to_row = {k: r for r, k in enumerate(union_keys)}
    N = len(union_keys)
    D = views[0][1].shape[1]  # feature dim (e.g., 1024)

    # Sum & count aggregation
    sum_feats = np.zeros((N, D), dtype=np.float32)
    counts    = np.zeros((N, 1), dtype=np.float32)

    for idx, tok in views:
        pos = {int(i): p for p, i in enumerate(idx)}
        for k, p in pos.items():
            r = key_to_row[k]
            sum_feats[r] += tok[p]
            counts[r, 0] += 1.0

    # Simple mean over available views
    feats_union = (sum_feats / np.clip(counts, 1.0, None)).astype(np.float32)

    # Map union voxel ids back to (x,y,z)
    coords_union = vox_idx[np.asarray(union_keys, dtype=np.int64)].astype(np.float32)

    # Pack and save
    payload = np.concatenate([coords_union, feats_union], axis=1)  # [N, 3+1024]
    vis.save_voxel_as_ply(
            payload,
            f"vis/scene_level_voxel_grow.ply",
            show_color=True,
        )
    _save_featured_voxel(torch.from_numpy(payload), output_file=voxel_path)
    


def filter_ref_drop_if_removed_in_gen(idx_keep_ref, tokens_ref,
                                      idx_keep_gen, idx_keep_gen_filt):
    """
    Drop REF voxels if their voxel-id was filtered out in GEN.

    Args:
      idx_keep_ref      : (Nr,) int array of voxel ids (REF)
      tokens_ref        : (Nr,D) float array
      idx_keep_gen      : (Ng,) int array of voxel ids before GEN filtering
      idx_keep_gen_filt : (Ng',) int array of voxel ids after GEN filtering

    Returns:
      idx_ref_filt, tokens_ref_filt
    """
    # GEN indices that got dropped
    dropped_gen_ids = set(idx_keep_gen.tolist()) - set(idx_keep_gen_filt.tolist())

    # Build mask for REF
    ref_mask = np.fromiter((i not in dropped_gen_ids for i in idx_keep_ref),
                           dtype=bool, count=len(idx_keep_ref))
    print(ref_mask)
    print(idx_keep_ref)

    return idx_keep_ref[ref_mask], tokens_ref[ref_mask]

@torch.no_grad()
def voxelise_features(
    obj_data: Dict[str, str],
    scan_id: str,
    mode: str = "gs_annotations",
) -> None:
    """
    Voxelise features for scan.

    Args:
        obj_data (Dict[str, str]): Object data.
        scan_id (str): Scan ID.
        mode (str, optional): Mode to run subscan generation on. Defaults to "gs_annotations".
    """
    G = 64
    scenes_dir = osp.join(root_dir, "scenes")
    frame_idxs = scan3r.load_frame_idxs(data_dir=scenes_dir, scan_id=scan_id)
    # frame_idxs, heldout_idxs = scan3r.load_frame_idxs_held_out(data_dir = scenes_dir, scan_id = scan_id, heldout_ratio=0.2)
    frame_id = frame_idxs[0]  # or any frame you want
    print('frame_id:',frame_id)
    rgb_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{frame_id}.color.jpg"
    depth_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{frame_id}.depth.pgm" 
    extrinsics = scan3r.load_frame_poses(
        data_dir=root_dir, scan_id=scan_id, frame_idxs=frame_idxs
    )
    
    intrinsics = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id)
    intrinsics = intrinsics["intrinsic_mat"]
    intrinsics_d = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id, type = 'depth')
    intrinsics_d = intrinsics_d["intrinsic_mat"]
   
    with Image.open(rgb_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)  # (H,W,3)
        
    depth = np.array(Image.open(depth_path))  # int32, mm
    depth = depth.astype(np.float32) / 1000.0
    resized_depth = Image.fromarray(depth).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
    
    
    fx, fy = intrinsics_d[0,0], intrinsics_d[1,1]
    cx, cy = intrinsics_d[0,2], intrinsics_d[1,2]
    H, W = depth.shape
    fx *= rgb.shape[1] / W
    cx *= rgb.shape[1] / W
    fy *= rgb.shape[0] / H
    cy *= rgb.shape[0] / H
    # build pixel grid
    H_rgb, W_rgb = rgb.shape[:2]
    # u, v = np.meshgrid(np.arange(W_rgb, dtype=np.float32), np.arange(H_rgb, dtype=np.float32), indexing="xy")
    u, v = np.meshgrid(np.arange(W_rgb, dtype=np.float32), np.arange(H_rgb, dtype=np.float32), indexing="xy")
    Z = np.array(resized_depth)
    # Z = depth
    u = u.reshape(-1)
    v = v.reshape(-1)
    Z = Z.reshape(-1)
    # valid = Z > 0
    # u = u[valid]; v = v[valid]; Z = Z[valid]
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    cam_points = np.stack([X, Y, Z], axis=1)  # (N,3) camera coords

    # camera->world pose for this frame
    T_cw = (extrinsics[frame_id]).astype(np.float32)  # 4x4
    # T_wc = (extrinsics[frame_id]).astype(np.float32)
    # R_wc = T_wc[:3, :3]; t_wc = T_wc[:3, 3]
    R_cw = T_cw[:3, :3]; t_cw = T_cw[:3, 3]

    world_points = (cam_points @ R_cw.T) + t_cw[None, :]  # (N,3)

    vggt_cam = np.load("/home/yihan/vggt/cameras.npz")
    
    E_ref_pred = vggt_cam["extrinsic"][0][0]   # [4,4] W2C
    E_gen_pred = vggt_cam["extrinsic"][0][1]
    E_ref_pred = to_homog_4x4(E_ref_pred)
    E_gen_pred = to_homog_4x4(E_gen_pred)
    
    
    pcd_vggt  = o3d.io.read_point_cloud("/home/yihan/vggt/vggt_unprojection.ply")
    P = np.asarray(pcd_vggt.points).astype(np.float32)   # (N,3)
    C = np.asarray(pcd_vggt.colors) if pcd_vggt.has_colors() else None
 
    P_gt = (P @ R_cw.T) + t_cw[None, :]

    # Save
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(P_gt)
    
    pcd_gt.colors = pcd_vggt.colors
    o3d.io.write_point_cloud("vis/cloud_gt_world_vggt.ply", pcd_gt)
    
    scene_output_dir = osp.join(args.model_dir, "files", mode, scan_id, "scene_level")
    voxel_path = osp.join(scene_output_dir, "voxel_output_dense.npz")
    mean_scale_path=osp.join(scene_output_dir, "mean_scale_dense.npz")
    # mean, scale, pts_normed = _normalize_points(world_points)
    mean, scale, pts_normed = _normalize_points(P_gt)
    print('mean: ', mean)
    print('scale: ', scale)
    print('len pts_normed: ', len(pts_normed))
    vox_idx, voxel_centers_normed = _voxelize_points_normed(pts_normed, grid_size=64)  # (M,3) 
    print('len vox_idx: ', len(vox_idx))
    print('len voxel_centers_normed: ', len(voxel_centers_normed))
    # save mean/scale for later composition
    if not args.dry_run:
        os.makedirs(os.path.dirname(mean_scale_path), exist_ok=True)
        np.savez(mean_scale_path, mean=mean, scale=scale)
        _LOGGER.info(f"Saved mean and scale to {mean_scale_path}") 
        
    # Prepare a single input image tensor for DINO
    # ---- DINO input (avoid non-writable warning) ----
    rgb_safe = rgb.copy()  # make array writable
    rgb_tensor = torch.from_numpy(rgb_safe).permute(2, 0, 1).float() / 255.0  # (3,H,W)

    # Run DINO (produces (1, 1024, n, n) at 518x518)
    patch_embeddings = _get_dino_embedding(rgb_tensor.unsqueeze(0).cuda())
    n = patch_embeddings.shape[-1]  # ~37 for ViT-L/14 @ 518

    # ---- Project voxel centers to RGB pixels ----
    T_wc =np.linalg.inv(extrinsics[frame_id]).astype(np.float32)          # world->camera
    R_wc, t_wc = T_wc[:3, :3], T_wc[:3, 3]

    voxel = torch.from_numpy(voxel_centers_normed).float()  # (M,3) in [-0.5,0.5]
    # de-normalize voxel centers back to world coords
    
    voxel_world = voxel * torch.from_numpy(scale).float()[None, :] * 2  \
                + torch.from_numpy(mean).float()[None, :]
                
    # both view sampling           
    rgb_ref = rgb
    ref_tokens_518 = _get_dino_embedding(_prep_image_for_dino(rgb_ref))
    gen_img_pil = Image.open('/home/yihan/vggt/input_images_20250821_172604_219866/images/generated.jpeg').convert("RGB")
    rgb_gen = np.asarray(gen_img_pil, dtype=np.uint8)  # (H,W,3)
    print('rgb_gen shape: ', rgb_gen.shape)
    gen_tokens_518 = _get_dino_embedding(_prep_image_for_dino(rgb_gen))
    # 3) World->Cam for REF (you already have this)
    T_wc_ref = np.linalg.inv(extrinsics[frame_id]).astype(np.float32) 
    Rel =  (E_gen_pred) @np.linalg.inv(E_ref_pred)
    extrinsics_gen = Rel @ T_wc_ref  
    # extrinsics_gen =  E_gen_pred @ extrinsics[frame_id]
    T_wc_gen = (extrinsics_gen).astype(np.float32) 
    K_ref = intrinsics
    # K_gen = intrinsics 
    H_ref, W_ref = rgb_ref.shape[:2]    # e.g., (968, 1296)
    H_gen, W_gen = rgb_gen.shape[:2]    # (768, 1344)
    K_gen = intrinsics.copy()
    sx = W_gen / float(W_ref)
    sy = H_gen / float(H_ref)
    K_gen[0,0] *= sx
    K_gen[0,2] *= sx
    K_gen[1,1] *= sy
    K_gen[1,2] *= sy 
    # 6) Sample REF
    idx_keep_ref, tokens_ref = _project_and_sample_dino(
        voxel_world=torch.from_numpy(voxel_world.numpy() if isinstance(voxel_world, torch.Tensor) else voxel_world),
        T_wc=T_wc_ref,
        K=K_ref,
        img_hw=rgb_ref.shape[:2],
        dino_tokens=ref_tokens_518,
    )
    coords_ref = vox_idx[idx_keep_ref] 
    payload_ref = np.concatenate(
    [coords_ref.astype(np.float32), tokens_ref.astype(np.float32)], axis=1
    )  # shape (Nr, 3 + D)
    vis.save_voxel_as_ply(
    payload_ref,
    "vis/scene_level_voxel_ref.ply",
    show_color=True,  # assumes the first 3 feature dims are RGB or can be mapped to colors
    )
    
    print('len idx_keep_ref: ',len(idx_keep_ref))

    # 7) Sample GEN
    idx_keep_gen, tokens_gen = _project_and_sample_dino(
        voxel_world=torch.from_numpy(voxel_world.numpy() if isinstance(voxel_world, torch.Tensor) else voxel_world),
        T_wc=T_wc_gen,
        K=K_gen,
        img_hw=rgb_gen.shape[:2],
        dino_tokens=gen_tokens_518,
    )
    
    coords_gen = vox_idx[idx_keep_gen] 
    payload_gen = np.concatenate(
    [coords_gen.astype(np.float32), tokens_gen.astype(np.float32)], axis=1
    )  # shape (Nr, 3 + D)
    vis.save_voxel_as_ply(
    payload_gen,
    "vis/scene_level_voxel_gen.ply",
    show_color=True,  # assumes the first 3 feature dims are RGB or can be mapped to colors
    )
    set_ref = set(idx_keep_ref.tolist())
    set_gen = set(idx_keep_gen.tolist())
    idx_both = np.array(sorted(list(set_ref & set_gen)), dtype=np.int64)
 
    G = 64
 
    feat_dim = 1024
    idx_keep_gen_filt, tokens_gen_filt = filter_gen_by_ref_similarity(
    idx_keep_ref=idx_keep_ref,
    tokens_ref=tokens_ref,
    idx_keep_gen=idx_keep_gen,
    tokens_gen=tokens_gen,
    vox_idx=vox_idx,   # (M,3) int indices from _voxelize_points_normed
    G=G,
    R_vox=8,           # neighbor radius in voxels (try 1, or 2 if sparse)
    tau=0.7,          # cosine threshold; try 0.75–0.90 depending on how strict you want to be
    use_max_neighbor=True,
    world_points = None,
    mean = mean,
    scale = scale
    )
    # Filter REF by what survived in GEN:
    idx_ref_filt, tokens_ref_filt = filter_ref_drop_if_removed_in_gen(
    idx_keep_ref, tokens_ref,
    idx_keep_gen, idx_keep_gen_filt
)

    # save_union_average(idx_keep_ref, tokens_ref, idx_keep_gen, tokens_gen, vox_idx, voxel_path)
    save_union_average(
    # idx_keep_ref, tokens_ref,
    idx_ref_filt, tokens_ref_filt,
    idx_keep_gen_filt, tokens_gen_filt,
    vox_idx, voxel_path
    )


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

    # all_subscan_ids = [
    #     subscan_id
    #     for scan_id in subscan_ids_generated
    #     for subscan_id in subscan_ids_generated[scan_id]
    # ]
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

    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model.eval().cuda()
    transform = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    scan_ids = process_data(cfg, mode="gs_annotations", split=args.split)
