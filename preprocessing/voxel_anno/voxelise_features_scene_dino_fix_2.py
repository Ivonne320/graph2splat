import itertools
import logging
import os
import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple, Optional

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import utils3d
import json
import time 

from PIL import Image
from tqdm import tqdm

from configs import Config, update_configs
from utils import common, scan3r
from utils import visualisation as vis
# voxelize

_LOGGER = logging.getLogger(__name__)

@torch.no_grad()
def _get_dino_embedding(
    images: torch.Tensor, batch_size: int = 8, image_size: int = 512
) -> torch.Tensor:
    """
    images: (N,3,H,W) in [0,1]
    Returns (N,C,Hp,Wp) patch embeddings from DINOv3.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected images shape (N,3,H,W), got {images.shape}")
    if images.device.type != "cuda":
        images = images.cuda(non_blocking=True)

    imgs = F.interpolate(
        images, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    patches = []
    model.eval()
    for start in range(0, imgs.shape[0], batch_size):
        chunk = imgs[start : start + batch_size]
        outputs = model.forward_features(chunk)
        if "x_norm_patchtokens" in outputs:
            tok = outputs["x_norm_patchtokens"]
        elif "x_prenorm" in outputs:
            reg = getattr(model, "num_register_tokens", 0)
            tok = outputs["x_prenorm"][:, 1 + reg :, :]
        else:
            raise KeyError("DINOv3 features missing x_norm_patchtokens/x_prenorm")
        n_patch = int(tok.shape[1] ** 0.5)
        if n_patch * n_patch != tok.shape[1]:
            raise ValueError(f"Non-square patch grid: {tok.shape[1]} tokens")
        tok = tok.transpose(1, 2).reshape(chunk.size(0), tok.shape[2], n_patch, n_patch)
        patches.append(tok.detach().float())
    return torch.cat(patches, dim=0)


# @torch.no_grad()
# def _get_dino_embedding(images: torch.Tensor, bs: int = 16) -> torch.Tensor:
#     """
#     images: (N,3,H,W) in [0,1], on GPU or CPU. Returns (N,1024,Hp,Wp) with Hp=Wp=518//14=37
#     """
#     # Keep on GPU, NHWC-friendly
#     if images.device.type != "cuda":
#         images = images.cuda(non_blocking=True)
#     # Resize on GPU (faster than torchvision CPU resize)
#     imgs = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)

#     # Normalize on GPU
#     mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1,3,1,1)
#     std  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1,3,1,1)
#     imgs = (imgs - mean) / std

#     outs = []
#     n = imgs.shape[0]
#     model.eval()
#     # with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
#     for i in range(0, n, bs):
#         chunk = imgs[i:i+bs]
#         out = model(chunk, is_training=True)  # don’t force training path
#         x = out["x_prenorm"]  # (B, tokens, C)
#         # strip [cls] and register tokens
#         x = x[:, model.num_register_tokens + 1 :, :]  # (B, P*P, C)
#         B, L, C = x.shape
#         P = 518 // 14  # 37
#         x = x.transpose(1, 2).reshape(B, C, P, P).contiguous()  # (B,1024,37,37)
#         outs.append(x)
#     return torch.cat(outs, dim=0).float()  # back to fp32 if you like


# @torch.no_grad()
# def _get_dino_embedding(images: torch.Tensor):
#     """
#     images: (B,3,H,W) in [0,1]
#     returns:
#       emb:   (B, C, H_p, W_p)
#       sizes: (H_in, W_in, H_p, W_p)
#     """
#     # preprocess with your transform
#     imgs = images.reshape(-1, 3, images.shape[-2], images.shape[-1]).cpu()
#     inp  = transform(imgs).cuda()         # (B,3,H_in,W_in)

#     model.eval()
#     out = model(inp, is_training=True)    # dict with patch tokens

#     # 1) take patch tokens directly (B, N, C)
#     if "x_norm_patchtokens" not in out:
#         # fallback if needed
#         reg = getattr(model, "num_register_tokens", 0)
#         tok = out["x_prenorm"][:, 1 + reg : ]         # (B, N, C)
#     else:
#         tok = out["x_norm_patchtokens"]               # (B, N, C)

#     # 2) compute patch grid from actual input + patch size
#     H_in, W_in = int(inp.shape[-2]), int(inp.shape[-1])
#     p = getattr(model, "patch_size", None)
#     if p is None and hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
#         p = model.patch_embed.patch_size
#     p = int(p[0] if isinstance(p, (tuple, list)) else p)
#     H_p, W_p = H_in // p, W_in // p
#     assert tok.shape[1] == H_p * W_p, f"N={tok.shape[1]} != {H_p*W_p} (H_in={H_in}, W_in={W_in}, p={p})"

#     # 3) (B,N,C) -> (B,C,H_p,W_p)
#     emb = tok.permute(0, 2, 1).contiguous().view(tok.size(0), tok.size(2), H_p, W_p)
#     return emb, (H_in, W_in, H_p, W_p)

def _save_featured_voxel(
    voxel: torch.Tensor, output_file: str = "voxel_output_dense.npz"
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file, voxel.cpu().numpy())
    _LOGGER.info(f"Voxel saved to {output_file}")


def _project_to_image(
    voxel: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    grid_size: tuple[int] = (128, 128, 128),
):
    voxel_size = 1.0 / grid_size[0]
    # voxel = voxel.float() * voxel_size
    voxel = ((voxel.float() + 0.5) * voxel_size) - 0.5
    # assert voxel.min() >= 0.0 and voxel.max() <= 1.0

    # voxel = voxel * 2.0 - 1.0
    # assert voxel.min() >= -1.0 and voxel.max() <= 1.0
    # voxel = voxel * scale + mean
    # voxel = voxel * scale[None, :] + mean[None, :]
    voxel = voxel * (2.0 * scale[None, :]) + mean[None, :]
    uv = utils3d.torch.project_cv(
        voxel.float(), extrinsics.float(), intrinsics.float()
    )[0]
    return uv


def _upright_angle_from_pose(pose: np.ndarray) -> float:
    """
    Estimate the in-plane rotation (deg) to align world +Z with image up.
    Assumes pose is camera-to-world; uses OpenCV camera coords (x right, y down).
    """
    pose = np.asarray(pose).reshape(4, 4)
    r_cw = pose[:3, :3]
    r_wc = r_cw.T
    up_w = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up_cam = r_wc @ up_w
    norm_xy = np.linalg.norm(up_cam[:2])
    if norm_xy < 1e-6:
        return 0.0
    angle = np.degrees(np.arctan2(up_cam[0], -up_cam[1]))
    return float(angle)


def _rotate_images_batch(images: torch.Tensor, angles_deg: list[float]) -> torch.Tensor:
    """
    Rotate a batch of images by per-frame angles (CCW in image coords).
    images: (N,3,H,W)
    """
    if images.numel() == 0 or len(angles_deg) == 0:
        return images
    device = images.device
    angles = torch.tensor(angles_deg, device=device, dtype=images.dtype) * (np.pi / 180.0)
    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)
    theta = torch.zeros((images.size(0), 2, 3), device=device, dtype=images.dtype)
    theta[:, 0, 0] = cos_t
    theta[:, 0, 1] = sin_t
    theta[:, 1, 0] = -sin_t
    theta[:, 1, 1] = cos_t
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(
        images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def _rotate_uvs_batch(
    uv: torch.Tensor, angles_deg: list[float], width: float, height: float
) -> torch.Tensor:
    """
    Rotate pixel coordinates (x,y) by per-frame angles (CCW in image coords).
    uv: (Nimg, Npts, 2)
    """
    if uv.numel() == 0 or len(angles_deg) == 0:
        return uv
    device = uv.device
    angles = torch.tensor(angles_deg, device=device, dtype=uv.dtype) * (np.pi / 180.0)
    cos_t = torch.cos(angles)[:, None]
    sin_t = torch.sin(angles)[:, None]
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    x = uv[..., 0] - cx
    y = uv[..., 1] - cy
    x_rot = cos_t * x + sin_t * y
    y_rot = -sin_t * x + cos_t * y
    return torch.stack([x_rot + cx, y_rot + cy], dim=-1)


def _segment_mesh(
    mesh: o3d.geometry.TriangleMesh, annos: np.ndarray, obj_id: int, scan_id: str
):
    faces = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    vertex_mask = annos == obj_id
    selected_vertices = np.where(vertex_mask)[0]
    index_map = {
        old_idx: dense_idx for dense_idx, old_idx in enumerate(selected_vertices)
    }

    # Filter faces that only contain selected vertices
    face_mask = np.all(np.isin(faces, selected_vertices), axis=1)
    selected_faces = faces[face_mask]
    reindexed_faces = np.vectorize(index_map.get)(selected_faces)

    # Create the segmented mesh
    segmented_mesh = o3d.geometry.TriangleMesh()
    segmented_mesh.vertices = o3d.utility.Vector3dVector(vertices[selected_vertices])
    segmented_mesh.triangles = o3d.utility.Vector3iVector(reindexed_faces)
    if args.visualize:
        o3d.io.write_triangle_mesh(
            f"vis/{scan_id}_{obj_id}_no_scale_segmented_mesh.ply", segmented_mesh
        )
    return segmented_mesh


# def _dilate_voxels(voxel_grid: o3d.geometry.VoxelGrid) -> np.ndarray:
#     voxel_grid = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
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
def _dilate_voxels(voxel_grid: o3d.geometry.VoxelGrid, G: int = 64, k: int = 3) -> np.ndarray:
    """
    Return (N,3) integer grid indices after dilation with a k^3 cube.
    """
    idx = np.array([v.grid_index for v in voxel_grid.get_voxels()], dtype=np.int64)
    occ = torch.zeros((1,1,G,G,G), device="cuda", dtype=torch.uint8)
    occ[0,0, idx[:,0], idx[:,1], idx[:,2]] = 1

    # 3D max-pool == morphological dilation
    # pad='same' with replicate padding
    pad = k//2
    occ = F.max_pool3d(occ.float(), kernel_size=k, stride=1, padding=pad)
    occ = (occ > 0.5).squeeze().nonzero(as_tuple=False).detach().cpu().numpy()
    return occ  # (N,3) in [0,G-1]


def _normalize_segmented_mesh(segmented_mesh: o3d.geometry.TriangleMesh):
    vertices = np.asarray(segmented_mesh.vertices)
    mean = vertices.mean(axis=0)
    vertices -= mean
    # scale = np.max(np.abs(vertices))
    scale = np.max(np.abs(vertices), axis=0) 
    scale[scale == 0] = 1.0
    vertices *= 1.0 / (2 * scale)
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    segmented_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mean, scale

def _gather_patch_tokens(patch_embeddings: torch.Tensor,
                         projection_normed: torch.Tensor,
                         H_rgb: int, W_rgb: int,
                         patch: int = 14) -> torch.Tensor:
    """
    patch_embeddings: (Nimg, 1024, P, P) with P=37
    projection_normed: (Nimg, Npts, 2) in [-1,1] (x:W, y:H)
    Returns: (Nimg, Npts, 1024) by nearest patch gather (no bilinear)
    """
    Nimg, C, P, _ = patch_embeddings.shape
    # Map [-1,1] back to pixel coords
    xy = (projection_normed + 1) * 0.5 * torch.tensor([W_rgb, H_rgb], device=projection_normed.device)
    # Convert to patch index
    ij = (xy / patch).long()  # (Nimg, Npts, 2) -> (x->j, y->i)
    j = ij[..., 0].clamp(0, P-1)
    i = ij[..., 1].clamp(0, P-1)
    # Linearize and gather
    lin = i * P + j  # (Nimg, Npts)
    feat = patch_embeddings.permute(0,2,3,1).reshape(Nimg, P*P, C)  # (Nimg, P^2, C)
    gathered = torch.gather(feat, 1, lin.unsqueeze(-1).expand(-1,-1,C))  # (Nimg, Npts, C)
    return gathered

def _average_patchtokens(
    patchtokens: torch.Tensor,
    projection_px: torch.Tensor,
    image_size: tuple[int, int],
    use_valid_mask: bool = True,
) -> torch.Tensor:
    """
    patchtokens: (Nimg, Npts, C)
    projection_px: (Nimg, Npts, 2) in pixel coords (x, y)
    image_size: (W, H)
    """
    if not use_valid_mask:
        return patchtokens.mean(dim=0)

    W, H = image_size
    # uv = projection_px.to(device=patchtokens.device, dtype=patchtokens.dtype)
    uv = projection_px
    valid = (
        (uv[..., 0] >= 0) & (uv[..., 0] < W) &
        (uv[..., 1] >= 0) & (uv[..., 1] < H)
    )

    if not valid.any():
        return patchtokens.mean(dim=0)

    valid = valid.unsqueeze(-1).to(dtype=patchtokens.dtype)
    patchtokens= patchtokens * (valid)                      # in-place mask, no extra allocation
    num = patchtokens.sum(dim=0)
    denom = valid.sum(dim=0).clamp_min(1.0)
    return (num / denom).float()


def _average_patchtokens_chunked(
    patch_embeddings: torch.Tensor,
    projection_normed: torch.Tensor,
    projection_px: torch.Tensor,
    image_size: tuple[int, int],
    use_valid_mask: bool = True,
    chunk_size: Optional[int] = None,
    max_chunk_mb: int = 512,
) -> torch.Tensor:
    """
    Sample and average patch tokens without materialising the full (Nimg,Npts,C) tensor.

    Args:
        patch_embeddings: (Nimg, C, Hp, Wp) DINO features on GPU.
        projection_normed: (Nimg, Npts, 2) grid in [-1,1] on CPU.
        projection_px: (Nimg, Npts, 2) grid in pixel coords for masking.
        image_size: (W, H) original image size.
        use_valid_mask: whether to mask projections outside the image bounds.
        chunk_size: optional fixed number of frames per chunk. If None, it is
            derived from `max_chunk_mb` so the intermediate tensor stays small.
        max_chunk_mb: soft cap for the temporary chunk tensor size.

    Returns:
        torch.Tensor: (Npts, C) averaged features on CPU.
    """

    if patch_embeddings.ndim != 4:
        raise ValueError(
            f"Expected patch embeddings of shape (N,C,H,W), got {patch_embeddings.shape}"
        )
    if projection_normed.ndim != 3:
        raise ValueError(
            f"Expected projection grid of shape (N,Npts,2), got {projection_normed.shape}"
        )

    Nimg, C, _, _ = patch_embeddings.shape
    Npts = projection_normed.shape[1]
    if Nimg == 0 or Npts == 0:
        return torch.zeros((Npts, C), dtype=torch.float32)

    if chunk_size is None:
        bytes_per_frame = Npts * C * patch_embeddings.element_size()
        max_chunk_bytes = max_chunk_mb * 1024 * 1024
        chunk_frames = max_chunk_bytes // max(bytes_per_frame, 1)
        chunk_size = max(1, min(Nimg, int(chunk_frames)))
    chunk_size = max(1, min(chunk_size, Nimg))

    W, H = image_size
    dtype = torch.float32
    sum_valid = torch.zeros((Npts, C), dtype=dtype)
    sum_counts = torch.zeros((Npts, 1), dtype=dtype)
    sum_all = torch.zeros((Npts, C), dtype=dtype)
    any_valid = not use_valid_mask

    for start in range(0, Nimg, chunk_size):
        end = min(start + chunk_size, Nimg)
        emb_chunk = patch_embeddings[start:end]
        grid_chunk = projection_normed[start:end].to(emb_chunk.device, non_blocking=True)
        grid_chunk = grid_chunk.unsqueeze(1)
        tokens = (
            F.grid_sample(
                emb_chunk,
                grid_chunk,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(2)
            .permute(0, 2, 1)
            .cpu()
            .to(dtype)
        )  # (chunk, Npts, C)

        sum_all += tokens.sum(dim=0)

        if use_valid_mask:
            px = projection_px[start:end]
            valid = (
                (px[..., 0] >= 0)
                & (px[..., 0] < W)
                & (px[..., 1] >= 0)
                & (px[..., 1] < H)
            )
            if valid.any():
                any_valid = True
            valid = valid.unsqueeze(-1).to(dtype)
        else:
            valid = torch.ones(tokens.shape[:2] + (1,), dtype=dtype)

        masked = tokens * valid
        sum_valid += masked.sum(dim=0)
        sum_counts += valid.sum(dim=0)

    if use_valid_mask and not any_valid:
        return (sum_all / float(Nimg)).float()

    denom = sum_counts.clamp_min(1.0)
    return (sum_valid / denom).float()

def _grid_sample_safe(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    grid_sample wrapper that retries without cuDNN if it hits CUDNN_STATUS_NOT_SUPPORTED.
    """
    input = input.contiguous()
    grid = grid.contiguous()
    try:
        return F.grid_sample(input, grid, mode=mode, align_corners=align_corners)
    except RuntimeError as exc:
        if "CUDNN_STATUS_NOT_SUPPORTED" not in str(exc):
            raise
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            return F.grid_sample(input, grid, mode=mode, align_corners=align_corners)
        finally:
            torch.backends.cudnn.enabled = prev
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

def _pixel_to_grid_norm(
    proj_px: torch.Tensor,
    in_size: tuple[int, int],
    feat_size: tuple[int, int],
) -> torch.Tensor:
    """
    Convert pixel coordinates in the DINO input space (W_in,H_in) into
    normalized grid coords for sampling feature maps of size (Wp,Hp).

    Args:
        proj_px: (Nimg, Npts, 2) pixel coords (x,y) in DINO input space.
        in_size: (W_in, H_in) image size fed into DINO.
        feat_size: (Wp, Hp) spatial size of patch feature map.

    Returns:
        grid: (Nimg, Npts, 2) in [-1,1] for F.grid_sample on (Hp,Wp).
    """
    if proj_px.ndim != 3 or proj_px.size(-1) != 2:
        raise ValueError(f"Expected proj_px (Nimg,Npts,2), got {proj_px.shape}")
    W_in, H_in = in_size
    Wp, Hp = feat_size
    # pixel -> feature-grid pixel coordinates
    u = proj_px[..., 0] * (float(Wp) / float(W_in))
    v = proj_px[..., 1] * (float(Hp) / float(H_in))
    grid = proj_px.new_empty(proj_px.shape)
    # normalize to [-1,1] in feature map coordinates
    grid[..., 0] = ((u + 0.5) / float(Wp)) * 2.0 - 1.0
    grid[..., 1] = ((v + 0.5) / float(Hp)) * 2.0 - 1.0
    return grid

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

    scenes_dir = osp.join(root_dir, "scenes")
    frame_idxs_raw = scan3r.load_frame_idxs(data_dir=scenes_dir, scan_id=scan_id)
    # frame_idxs, heldout_idxs = scan3r.load_frame_idxs_held_out(data_dir = scenes_dir, scan_id = scan_id, heldout_ratio=0.2)

    intrinsics = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id)
    mask = scan3r.load_masks(data_dir=root_dir, scan_id=scan_id)
    rendered =  []
    frame_idxs = []
    for frame_id in frame_idxs_raw:
        path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{frame_id}.color.jpg"
        score = scan3r._laplacian_focus_score(path, 256)
        # if score < 80:
        #     continue
        frame_idxs.append(frame_id)
        with Image.open(path) as img:
            arr = np.array(img)  # Load full data while file is open
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        rendered.append(tensor)
    print(f"len of raw frames: {len(frame_idxs)}")
    return
    if len(frame_idxs) == 0:
        _LOGGER.warning(f"No frames passed focus filtering for {scan_id}, skipping.")
        return
    extrinsics = scan3r.load_frame_poses(
        data_dir=root_dir, scan_id=scan_id, frame_idxs=frame_idxs
    )
    frame_angles = []
    for frame_id in frame_idxs:
        pose = extrinsics.get(frame_id)
        frame_angles.append(_upright_angle_from_pose(pose) if pose is not None else 0.0)
    mesh = scan3r.load_ply_mesh(
        data_dir=scenes_dir,
        scan_id=scan_id,
        label_file_name="labels.instances.annotated.v2.ply",
    )
    annos = scan3r.load_ply_data(
        data_dir=scenes_dir,
        scan_id=scan_id,
        label_file_name="labels.instances.annotated.v2.ply",
    )["vertex"]["objectId"]
    object_ids = [int(obj["id"]) for obj in obj_data["objects"]]
    print("object_ids: ", object_ids)
    
    scene_output_dir = osp.join(args.model_dir, "files", mode, scan_id, "scene_level_dinov3_128_reso_fixed_2")
    voxel_path = osp.join(scene_output_dir, "voxel_output_dense.npz")
    mean_scale_path=osp.join(scene_output_dir, "mean_scale_dense.npz")
    # if (
    #         osp.exists(mean_scale_path)
    #         and osp.exists(voxel_path)
    #         and "arr_0" in np.load(voxel_path)
    #         and not args.override
    #     ):
    #         _LOGGER.info(f"Skipping {scan_id} ")
    #         return
    try:
        
        # STEP 1: Segment the mesh
        # segmented_mesh = _segment_mesh(mesh, annos, obj_id, scan_id)
        # object_vertex_mask = annos >= 0
        object_vertex_mask = np.isin(annos, object_ids)
        selected_vertices = np.where(object_vertex_mask)[0]
        faces = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        face_mask = np.all(np.isin(faces, selected_vertices), axis=1)
        selected_faces = faces[face_mask]
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_vertices)}
        remapped_faces = np.vectorize(index_map.get)(selected_faces)
        # Create the mesh with only object geometry
        scene_mesh = o3d.geometry.TriangleMesh()
        scene_mesh.vertices = o3d.utility.Vector3dVector(vertices[selected_vertices])
        scene_mesh.triangles = o3d.utility.Vector3iVector(remapped_faces)
        
        # STEP 2: Normalize to unit cube (-0.5, 0.5)
        t0 = time.time()
        mean, scale = _normalize_segmented_mesh(scene_mesh)
        G = 128
        # Fast surface sampling voxelizer: denser than vertices, much faster than full mesh voxelization.
        voxel_grid = voxelize_mesh_simple_dense(
            scene_mesh,
            G=G,
            n_rand=0,  # increase for more surface coverage
            k_dilate=0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # voxel_grid = _dilate_voxels_idx(voxel_grid)
        
        _LOGGER.info(f"Step 2 normalization took {time.time()-t0:.2f}s"); t0 = time.time()
        # STEP 3: Voxelise the mesh
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        #     scene_mesh,
        #     1 / 64,
        #     min_bound=(-0.5, -0.5, -0.5),
        #     max_bound=(0.5, 0.5, 0.5),
        # )
        # voxel_grid = np.array([v.grid_index for v in voxel_grid.get_voxels()], dtype=np.int64)
        _LOGGER.info(f"Step 3 voxelization took {time.time()-t0:.2f}s"); t0 = time.time()

        # STEP 4: Save mean and scale (Scene composition)
        if not args.dry_run:
            os.makedirs(os.path.dirname(mean_scale_path), exist_ok=True)
            np.savez(mean_scale_path, mean=mean, scale=scale)
            _LOGGER.info(f"Saved mean and scale to {mean_scale_path}")

        # if (
        #     os.path.exists(voxel_path) and "arr_0" in np.load(voxel_path)
        # ) and not args.override:
        #     _LOGGER.info(f"Skipping {scan_id}")
        #     return

        # STEP 5: Render the object
        pose_camera_to_world = [
            np.linalg.inv(extrinsics[frame_idx]) for frame_idx in extrinsics
        ]

        masks = [mask[frame_id] for frame_id in frame_idxs]
        for i, frame_id in enumerate(frame_idxs[:3]):
            print(f"Mask {i} (frame {frame_id}) unique labels:", np.unique(masks[i]))
        # masks = [np.where(mask > 0, 1, 0) for mask in masks]
        masks =  [np.isin(mask, object_ids).astype(np.uint8) for mask in masks]
        rendered_obj = [
            image * mask[None, :, :] for image, mask in zip(rendered, masks)
        ]
        # remove empty images
        idx_empty = [i for i, r in enumerate(rendered_obj) if r.sum() == 0]
        rendered_obj = [
            r for i, r in enumerate(rendered_obj) if i not in idx_empty
        ][:150]
        _LOGGER.info(f"Length of rendered_obj: {len(rendered_obj)}")
        frame_angles = [a for i, a in enumerate(frame_angles) if i not in idx_empty][:150]
        rendered_obj = torch.stack(rendered_obj).float()
        if args.upright_images:
            rendered_obj = _rotate_images_batch(rendered_obj, frame_angles)
        pose_camera_to_world = [
            pose
            for i, pose in enumerate(pose_camera_to_world)
            if i not in idx_empty
        ][:150]

        # STEP 6: Project the voxel to the image
        projection = _project_to_image(
            torch.Tensor(voxel_grid),
            torch.Tensor(mean),
            # torch.Tensor([scale]),
            torch.Tensor(scale),
            torch.from_numpy(np.stack(pose_camera_to_world)),
            torch.from_numpy(intrinsics["intrinsic_mat"]),
            grid_size=(G, G, G),
        )  # Shape: (Nimages, Npoints, 2)
        _LOGGER.info(f"Step 6 projection took {time.time()-t0:.2f}s"); t0 = time.time()

        # STEP 7: Normalize the projection to [-1, 1]
        projection_px = projection
        if args.upright_images:
            projection_px = _rotate_uvs_batch(
                projection_px,
                frame_angles,
                intrinsics["width"],
                intrinsics["height"],
            )
       
        H_in, W_in = 512, 512
        W_orig = int(intrinsics["width"]); H_orig = int(intrinsics["height"])
        proj_px_in = projection_px.clone()
        proj_px_in[..., 0] = proj_px_in[..., 0] * (W_in / W_orig)
        proj_px_in[..., 1] = proj_px_in[..., 1] * (H_in / H_orig)
        proj_px_in = projection_px.clone()
        proj_px_in[..., 0] = proj_px_in[..., 0] * (float(W_in) / float(W_orig))
        proj_px_in[..., 1] = proj_px_in[..., 1] * (float(H_in) / float(H_orig))
        
        
         
       
        # STEP 8: Get the DINO embeddings
        patch_embeddings = _get_dino_embedding(
            rendered_obj
        )  # Shape: (Nimages, 1024, 64, 64)
        # patch_embeddings, (H_in, W_in, H_p, W_p) = _get_dino_embedding(rendered_obj.unsqueeze(0).cuda())
        _LOGGER.info(f"Step 8 get dino embedding took {time.time()-t0:.2f}s"); t0 = time.time()
        Hp = int(patch_embeddings.shape[-2])
        Wp = int(patch_embeddings.shape[-1])
        projection = _pixel_to_grid_norm(
            proj_px_in,
            in_size=(W_in, H_in),
            feat_size=(Wp, Hp),
        )
        # STEP 9: Match the embeddings to the projection
        patchtokens = _average_patchtokens_chunked(
            patch_embeddings,
            projection,
            projection_px,
            (intrinsics["width"], intrinsics["height"]),
            use_valid_mask=True,
        ).cpu().numpy().astype(np.float16)
        # patchtokens = (
        #     F.grid_sample(
        #         patch_embeddings,
        #         projection.cuda().unsqueeze(1),
        #         mode="bilinear",
        #         align_corners=False,
        #     )
        #     .squeeze(2)
        #     .permute(0, 2, 1)
        #     .cpu()
        #     .numpy()
        # )  # Shape: (Nimages, Npoints, 1024)

        # H_rgb = int(intrinsics["height"])
        # W_rgb = int(intrinsics["width"])
        # projection = projection.cuda(non_blocking=True)
        # patchtokens = _gather_patch_tokens(patch_embeddings, projection, H_rgb, W_rgb, patch=14)
        # _LOGGER.info(f"Step 9 gather patch tokens took {time.time()-t0:.2f}s"); t0 = time.time()
        # patchtokens = patchtokens.mean(dim=0).half().cpu().numpy()  # (Npts,1024)
        # projection_px = projection_px.to(patchtokens.device)
        # patchtokens = np.mean(patchtokens, axis=0).astype(
        #     np.float16
        # )  # Shape: (Npoints, 1024)
        assert patchtokens.shape[0] == voxel_grid.shape[0]
        assert patchtokens.shape[1] == 1024
        assert voxel_grid.shape[1] == 3
        voxel_grid = torch.concatenate(
            [torch.Tensor(voxel_grid), torch.Tensor(patchtokens)], dim=1
        )
        if args.visualize:
            vis.save_voxel_as_ply(
                voxel_grid.cpu().numpy(),
                f"vis/{scan_id}_scene_level_voxel_new.ply",
                show_color=True,
            )
        assert voxel_grid.shape[-1] == 1027
        if not args.dry_run:
            _save_featured_voxel(
                voxel_grid,
                output_file=voxel_path,
            )
        print("num_voxels:", voxel_grid.shape[0])
        # print("final array shape:", voxel.shape, "dtype:", voxel.dtype)
        # # store held-out indxes
        # held_out_dir = osp.join(scene_output_dir, "heldout_frame_indices.json")
        # with open(held_out_dir, "w") as f:
        #     json.dump(heldout_idxs,f)
            
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        _LOGGER.exception(f"Error processing {scan_id} : {e}")


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
    all_subscan_ids = all_subscan_ids[:10]
    out_txt = osp.join(root_dir, 'files', f"reproj_{split}_processed_{mode}.txt")
    with open(out_txt, "w") as f:
        for sid in all_subscan_ids:
            f.write(f"{sid}\n")
    # all_subscan_ids = ["77361fca-d054-2a22-8974-547ca1fbb90f"]
    # all_subscan_ids = ['fcf66d9e-622d-291c-84c2-bb23dfe31327','02b33df9-be2b-2d54-9062-1253be3ce186','02b33dfd-be2b-2d54-91d2-55454852009e','02b33e01-be2b-2d54-93fb-4145a709cec5',
    #                          'fcf66d8a-622d-291c-8429-0e1109c6bb26','fcf66d88-622d-291c-871f-699b2d063630','02b33e03-be2b-2d54-9129-5d28efdd68fa', '0958220d-e2c2-2de1-9710-c37018da1883',
    #                          '0958220b-e2c2-2de1-96bc-739f09c1e8f8', '09582205-e2c2-2de1-9475-1cdac7639e60','09582207-e2c2-2de1-972c-225d968c2ab4', '09582209-e2c2-2de1-9610-08baed932919',
    #                          '09582212-e2c2-2de1-9700-fa44b14fbded','0958221b-e2c2-2de1-96b1-6233099811a0','09582214-e2c2-2de1-956a-64d8da4ba7cc','09582216-e2c2-2de1-97de-efcab1ef9c43',
    #                          '09582219-e2c2-2de1-9534-519142703037','09582225-e2c2-2de1-9564-f6681ef5e511', '0958222a-e2c2-2de1-9474-35e601b3682a','0958222d-e2c2-2de1-9732-e2fb990692ef',
    #                          '09582223-e2c2-2de1-94b6-750684b4f80a', '09582228-e2c2-2de1-953d-f6f1ee4b3699','09582244-e2c2-2de1-956c-357092d949d1', 'dcb6a329-5526-23f1-9d81-7718f682269c']
    
    # all_subscan_ids = all_subscan_ids[1036:]
    

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
    # parser.add_argument("--model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_dir", type=str, default="vis")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument(
        "--use_valid_mask",
        action="store_true",
        help="Mask out projections that fall outside the image before averaging DINO features.",
    )
    parser.add_argument(
        "--upright_images",
        action="store_true",
        help="Rotate images (and projections) to align world +Z with image up.",
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == "__main__":
    common.init_log(level=logging.INFO)
    _LOGGER.info("**** Starting feature voxelisation for 3RScan ****")
    args, unknown = parse_args()
    os.makedirs(args.vis_dir, exist_ok=True)
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    root_dir = cfg.data.root_dir

    model = torch.hub.load("/cluster/home/wangyih/.cache/torch/hub/facebookresearch_dinov3_main", args.model, source='local', pretrained=False)
    ckpt_path = "/cluster/home/wangyih/.cache/torch/hub/facebookresearch_dinov3_main/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    # torch.backends.cudnn.benchmark = True
    # torch.set_float32_matmul_precision('high') 
    # model = torch.hub.load("facebookresearch/dinov2", args.model)
    model.eval().cuda()
    scan_ids = process_data(cfg, mode="gs_annotations", split=args.split)
