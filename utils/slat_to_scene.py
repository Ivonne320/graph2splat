import torch
from src.modules.sparse.basic import SparseTensor
from typing import Union, List
from src.representations.gaussian.gaussian_model import Gaussian
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from utils.visualisation import save_voxel_as_ply
import itertools
import numpy as np
import os
import open3d as o3d

def revoxelize_to_fixed_scene_slat(
    splat,
    means,
    scales,
    resolution=64,
):
    """
    Revoxelize per-object SLATs into a fixed-size scene-level SparseConvTensor of shape (64, 64, 64).

    Args:
        splat: SparseTensor with coords [N, 4] and feats [N, D]
        means: Tensor [B, 3] - object means
        scales: Tensor [B, 3] - object scales
        resolution: int - target grid resolution (default 64)

    Returns:
        SparseConvTensor with spatial_shape = [64, 64, 64]
    """
    coords = splat.coords[:, 1:].float()  # Drop batch dim, shape [N, 3]
    feats = splat.feats                   # Shape [N, D]
    batch_ids = splat.coords[:, 0]        # Object indices per voxel

    # Step 1: Normalize voxel centers to [-1, 1]
    coords_normalized = ((coords + 0.5) / resolution) * 2 - 1

    # Step 2: Convert to world space
    scales_per_voxel = scales[batch_ids]
    means_per_voxel = means[batch_ids]
    coords_world = coords_normalized * scales_per_voxel + means_per_voxel  # Shape [N, 3]

    # Step 3: Compute scene bounding box
    bbox_min = means.min(dim=0).values - scales.max(dim=0).values
    bbox_max = means.max(dim=0).values + scales.max(dim=0).values
    bbox_extent = bbox_max - bbox_min

    # Step 4: Compute voxel_size so that entire bbox fits in fixed resolution
    voxel_size = (bbox_extent / resolution).max().item()
    scene_origin = bbox_min

    # Step 5: Revoxelize world coords into scene grid
    voxel_coords = torch.floor((coords_world - scene_origin) / voxel_size).int()
    voxel_coords = voxel_coords.clamp(min=0, max=resolution - 1)

    # Step 6: Reconstruct SparseConvTensor
    batch_coords = torch.zeros((voxel_coords.shape[0], 4), dtype=torch.int32, device=coords.device)
    batch_coords[:, 0] = 0  # batch=0
    batch_coords[:, 1:] = voxel_coords
    return SparseTensor(
        feats=feats,
        coords=batch_coords,
        shape=torch.Size([1, *([resolution] * 3)])
    ), voxel_size, scene_origin
    
def revoxelize_to_fixed_scene_slat_with_aggregation(
    splat,
    means,
    scales,
    resolution=64,
):
    """
    Revoxelize per-object SLATs into a fixed-size scene-level SparseConvTensor with feature aggregation.

    Args:
        splat: SparseTensor with coords [N, 4] and feats [N, D]
        means: Tensor [B, 3] - object means
        scales: Tensor [B, 3] - object scales
        resolution: int - target grid resolution (default 64)

    Returns:
        (scene_tensor, voxel_size, scene_origin)
    """
    coords = splat.coords[:, 1:].float()
    feats = splat.feats
    batch_ids = splat.coords[:, 0]

    # Step 1: normalize voxel centers to [-1, 1]
    coords_normalized = ((coords + 0.5) / resolution) * 2 - 1


    # Step 2: backproject to world coordinates
    scales_per_voxel = scales[batch_ids]
    means_per_voxel = means[batch_ids]
    coords_world = coords_normalized * scales_per_voxel + means_per_voxel

    # bbox_min = (means - scales/2).min(dim=0).values
    # bbox_max = (means + scales/2).max(dim=0).values
    bbox_min = (means - scales).min(dim=0).values
    bbox_max = (means + scales).max(dim=0).values
    bbox_extent = bbox_max - bbox_min

    # Step 4: determine voxel size and origin
    voxel_size = (bbox_extent / resolution).max().item()
    
    scene_origin = (bbox_max + bbox_min) / 2 
   
    # Step 5: revoxelize to scene grid
    voxel_coords = torch.round((coords_world - scene_origin) / voxel_size + (resolution) / 2).int()
   
    mask = (voxel_coords >= 0) & (voxel_coords < resolution)
    mask = mask.all(dim=1)

    voxel_coords = voxel_coords[mask]
    feats = feats[mask]

    # Step 6: aggregate features (average features at same voxel)
    key_tuples = [tuple(k.tolist()) for k in voxel_coords]
    feat_dict = defaultdict(list)
    for k, f in zip(key_tuples, feats):
        feat_dict[k].append(f)
    
    unique_coords = []
    averaged_feats = []
    for k, v in feat_dict.items():
        unique_coords.append(k)
        averaged_feats.append(torch.stack(v).mean(dim=0))

    coords_tensor = torch.tensor(unique_coords, dtype=torch.int32, device=feats.device)
    feats_tensor = torch.stack(averaged_feats).to(feats.device)

    batch_coords = torch.zeros((coords_tensor.shape[0], 4), dtype=torch.int32, device=feats.device)
    batch_coords[:, 0] = 0
    batch_coords[:, 1:] = coords_tensor

    scene_tensor = SparseTensor(
        feats=feats_tensor,
        coords=batch_coords,
        shape=torch.Size([1, resolution, resolution, resolution])
    )
    print("Target bbox min:", bbox_min)
    print("Target bbox max:", bbox_max)

    return scene_tensor, voxel_size, scene_origin, bbox_extent,bbox_min
    # return scene_tensor, voxel_size, scene_origin
    

def revoxelize_scene_via_normalized_coords(
    splat: SparseTensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    resolution: int = 64,
    eps: float = 1e-6,
):
    """
    Revoxelize per-object SLATs into a fixed-size scene-level SparseConvTensor using
    normalized scene-space coordinates with anisotropic scaling to prevent skew.

    Args:
        splat: SparseTensor with coords [N, 4] and feats [N, D]
        means: Tensor [B, 3] - object centers (world-space means)
        scales: Tensor [B, 3] - object half-scales (world-space half-extents)
        resolution: int - target grid resolution (default 64)
        eps: float - small epsilon to avoid boundary issues

    Returns:
        scene_tensor: SparseTensor of shape [1, R, R, R]
        scene_mean: Tensor [3] - world-space center of the normalized scene
        scene_scales: Tensor [3] - per-axis half-scale of the scene
    """
    # Extract raw voxel coords and features
    coords = splat.coords[:, 1:].float()      # [N, 3]
    feats = splat.feats                       # [N, D]
    batch_ids = splat.coords[:, 0].long()     # [N]

    # Step 1: object-space to world-space
    coords_norm = ((coords + 0.5) / resolution) * 2.0 - 1.0
    scales_per = scales[batch_ids]
    means_per  = means[batch_ids]
    coords_world = coords_norm * scales_per + means_per

    # Step 2: compute global bbox from object means & scales
    # Use object-space bbox to avoid bias from sparse world points
    bbox_min = (means - scales).min(dim=0).values
    bbox_max = (means + scales).max(dim=0).values
    scene_mean = (bbox_min + bbox_max) / 2.0
    scene_scales = (bbox_max - bbox_min) / 2.0
    scene_scales = scene_scales.clamp(min=eps)

    # Step 3: normalize world coords to [-0.5,0.5] using object-based bbox
    coords_scene = (coords_world - scene_mean) / (2.0 * scene_scales)
    coords_scene = coords_scene.clamp(-0.5 + eps, 0.5 - eps)

    # Step 4: map to integer grid [0, resolution)
    # use round instead of floor to avoid half-voxel bias
    voxel_coords = torch.clamp(
        torch.round((coords_scene + 0.5) * resolution),
        min=0, max=resolution-1
    ).int()

    # Step 5: mask out-of-bounds
    mask = (voxel_coords >= 0) & (voxel_coords < resolution)
    mask = mask.all(dim=1)
    filtered = voxel_coords[mask]
    feats_filtered = feats[mask]

    # Step 6: aggregate features by voxel index
    agg = defaultdict(list)
    for idx, f in zip(filtered.tolist(), feats_filtered):
        agg[tuple(idx)].append(f)

    coords_list, feats_list = [], []
    for idx, flist in agg.items():
        coords_list.append(idx)
        feats_list.append(torch.stack(flist).mean(dim=0))

    coords_tensor = torch.tensor(coords_list, dtype=torch.int32, device=feats.device)
    feats_tensor = torch.stack(feats_list).to(feats.device)

    # Step 7: build SparseTensor
    batch_coords = torch.zeros((coords_tensor.size(0), 4), dtype=torch.int32, device=feats.device)
    batch_coords[:, 0] = 0
    batch_coords[:, 1:] = coords_tensor
    scene_tensor = SparseTensor(
        feats=feats_tensor,
        coords=batch_coords,
        shape=torch.Size([1, resolution, resolution, resolution])
    )

    return scene_tensor, scene_mean, scene_scales




def add_coordinate_frame_to_scene():
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    return axis

def visualize_slat_alignment(slat_tensor, obj_idx=0, out_dir="vis_slat"):
    """
    Visualize a SLAT as 3D voxels using Open3D.

    Args:
        slat_tensor: torchsparse.SparseTensor or any object with .coords and .feats
        obj_idx: int, index for naming
        out_dir: output folder for saving .ply
    """
    os.makedirs(out_dir, exist_ok=True)

    coords = slat_tensor.coords.cpu().numpy()[:, 1:]  # Drop batch index
    feats = slat_tensor.feats.cpu().numpy()

    # Normalize voxel coordinates and features
    coords_centered = coords - coords.min(axis=0)  # Align all objects to origin
    featured_voxel = np.concatenate([coords_centered, feats], axis=1)

    save_voxel_as_ply(
        featured_voxel=featured_voxel,
        filename=os.path.join(out_dir, f"object_{obj_idx:02d}.ply"),
        show_color=True,
    )
    # pcd = o3d.io.read_point_cloud(os.path.join(out_dir, f"object_{obj_idx:02d}.ply"))
    # axis = add_coordinate_frame_to_scene()
    # o3d.visualization.draw_geometries([pcd, axis])