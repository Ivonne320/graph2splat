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

def _normalize_points(points: np.ndarray):
    """
    points: (N,3) in world coords
    Returns mean (3,), scale (3,) and normalized points in [-0.5,0.5]^3 (clipped)
    """
    mean = points.mean(axis=0)
    pts = points - mean
    scale = np.max(np.abs(pts), axis=0)
    scale[scale == 0] = 1.0
    pts = pts / (2.0 * scale)
    pts = np.clip(pts, -0.5 + 1e-6, 0.5 - 1e-6)
    return mean, scale, pts

def _voxelize_points_normed(points_normed: np.ndarray, grid_size=64):
    """
    points_normed in [-0.5,0.5]. Returns unique voxel centers (M,3) in normalized coords.
    """
    idx = np.floor((points_normed + 0.5) * grid_size).astype(np.int32)
    idx = np.clip(idx, 0, grid_size - 1)
    # unique occupied voxels
    idx = np.unique(idx, axis=0)
    # center coords back to normalized space
    centers = (idx.astype(np.float32) + 0.5) / grid_size - 0.5
    return idx, centers  # (M,3)


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
    frame_idxs = scan3r.load_frame_idxs(data_dir=scenes_dir, scan_id=scan_id)
    # frame_idxs, heldout_idxs = scan3r.load_frame_idxs_held_out(data_dir = scenes_dir, scan_id = scan_id, heldout_ratio=0.2)
    frame_id = frame_idxs[0]  # or any frame you want
    rgb_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{frame_id}.color.jpg"
    depth_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{frame_id}.depth.pgm" 
    extrinsics = scan3r.load_frame_poses(
        data_dir=root_dir, scan_id=scan_id, frame_idxs=frame_idxs
    )
    
    intrinsics = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id)
    intrinsics = intrinsics["intrinsic_mat"]
    intrinsics_d = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id, type = 'depth')
    intrinsics_d = intrinsics_d["intrinsic_mat"]
    # mask = scan3r.load_masks(data_dir=root_dir, scan_id=scan_id)
    # load RGB (for DINO)
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
    u, v = np.meshgrid(np.arange(W_rgb, dtype=np.float32), np.arange(H_rgb, dtype=np.float32), indexing="xy")
    Z = np.array(resized_depth)
    # valid = Z > 0  # remove invalid depth
    # u = u[valid]; v = v[valid]; Z = Z[valid]
    u = u.reshape(-1)
    v = v.reshape(-1)
    Z = Z.reshape(-1)
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    cam_points = np.stack([X, Y, Z], axis=-1)  # (N,3) camera coords

    # camera->world pose for this frame
    T_cw = (extrinsics[frame_id]).astype(np.float32)  # 4x4
    R_cw = T_cw[:3, :3]; t_cw = T_cw[:3, 3]

    world_points = (cam_points @ R_cw.T) + t_cw[None, :]  # (N,3)
    
    u_valid = u.astype(np.int32)
    v_valid = v.astype(np.int32)
    colors = rgb[v_valid, u_valid, :].astype(np.float32) / 255.0  # (N,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    out_path = f"vis/{scan_id}_lifted_points.ply"
    o3d.io.write_point_cloud(out_path, pcd)
    # normalize and voxelize
    scene_output_dir = osp.join(args.model_dir, "files", mode, scan_id, "scene_level")
    voxel_path = osp.join(scene_output_dir, "voxel_output_dense.npz")
    mean_scale_path=osp.join(scene_output_dir, "mean_scale_dense.npz")
    mean, scale, pts_normed = _normalize_points(world_points)
    vox_idx, voxel_centers_normed = _voxelize_points_normed(pts_normed, grid_size=64)  # (M,3) 
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

    # project to RGB (intrinsics are RGB K @ full RGB size)
    uv = utils3d.torch.project_cv(
        voxel_world.float(),
        torch.from_numpy(T_wc).float()[None, ...],
        torch.from_numpy(intrinsics).float()[None, ...]
    )[0]  # (M,2), pixels in RGB frame
    uv = uv.squeeze(0) 
    M = voxel_centers_normed.shape[0]
    feat_dim = 1024
    idx_all = np.arange(M)
    
    # Allocate accumulators across ALL views (GT + generated)
    feat_sum = np.zeros((M, feat_dim), dtype=np.float64)
    feat_cnt = np.zeros((M,), dtype=np.float64)
    
    
    # ---- Sampling single gt view dino feature first -----
    
    # ---- Keep only pixels inside the RGB image ----
    H_rgb, W_rgb = rgb.shape[:2]
    u = uv[:, 0]; v = uv[:, 1]
    inb_img = (u >= 0) & (u < W_rgb) & (v >= 0) & (v < H_rgb)
    if inb_img.sum() == 0:
        raise RuntimeError("All projected voxels are out of image bounds.")

    uv = uv[inb_img]  # (M_in, 2)
    # keep the same subset for the voxel coordinates (numpy side)
    idx_img = idx_all[inb_img.cpu().numpy()]      # ★ indices after image mask
    voxel_centers_img = voxel_centers_normed[idx_img]  # (M_img, 3)

    # ---- Map to 518×518 then to DINO's patch grid (n×n) ----
    sx = 518.0 / float(W_rgb)
    sy = 518.0 / float(H_rgb)
    # uv_resized = uv.clone()
    # uv_resized[:, 0] = uv_resized[:, 0] * sx
    # uv_resized[:, 1] = uv_resized[:, 1] * sy

    # # 518px -> [0, n-1] patch coords
    # u_patch = uv_resized[:, 0] / 518.0 * (n - 1)
    # v_patch = uv_resized[:, 1] / 518.0 * (n - 1)

    # # build grid in [-1, 1] for input of size (n, n)
    # gx = (u_patch / (n - 1)) * 2.0 - 1.0
    # gy = (v_patch / (n - 1)) * 2.0 - 1.0

    # # Keep only grid points inside [-1,1] to avoid out-of-bounds zeros
    # inb_grid = (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)
    # gx = gx[inb_grid]; gy = gy[inb_grid]
    # idx_keep = idx_img[inb_grid.cpu().numpy()]    # ★ final kept indices in original M

    # grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2).to(patch_embeddings.device)
    u518 = uv[:, 0] * sx
    v518 = uv[:, 1] * sy

    # convert to grid_sample coords for an n×n feature map, align_corners=False
    gx = 2.0 * (u518 + 0.5) / 518.0 - 1.0
    gy = 2.0 * (v518 + 0.5) / 518.0 - 1.0

    inb_grid = (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)
    gx = gx[inb_grid]; gy = gy[inb_grid]
    idx_keep = idx_img[inb_grid.cpu().numpy()]
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2).to(patch_embeddings.device)

    # ---- Sample DINO features at those points ----
    feat = F.grid_sample(
        patch_embeddings.float(),   # (1, 1024, n, n)
        grid.float(),               # (1, M_kept, 1, 2)
        mode="bilinear",
        align_corners=False
    ).squeeze(-1).permute(0, 2, 1).contiguous()            # (1, M_kept, 1024)

    patchtokens = feat[0].detach().cpu().numpy().astype(np.float16)  # (M_kept, 1024)
    vox_idx_keep = vox_idx[idx_keep] 
    voxel_with_feats = np.concatenate(
        [vox_idx_keep.astype(np.float32), patchtokens], axis=1
    )
    if args.visualize:
        vis.save_voxel_as_ply(
            voxel_with_feats,
            f"vis/{scan_id}_scene_level_voxel.ply",
            show_color=True,
        )

    if not args.dry_run:
        _save_featured_voxel(torch.from_numpy(voxel_with_feats), output_file=voxel_path)

def _natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(s))]

def load_generated_with_gt_ref(
    root_dir,                 # ".../spiral" (folder that has per_view_generations/ and poses/)
    C2W_ref,                  # your GT camera-to-world of the reference frame (T_cw_ref)
    K_ref,                    # 3x3 intrinsics of the reference (at size W_ref x H_ref)
    ref_size,                 # (W_ref, H_ref) of the reference image that K_ref is defined for
):
    img_dir  = os.path.join(root_dir, "per_view_generations")
    pose_dir = os.path.join(root_dir, "poses")

    img_files  = sorted(glob.glob(os.path.join(img_dir, "*.png")), key=_natkey)
    pose_files = sorted(glob.glob(os.path.join(pose_dir, "[0-9][0-9][0-9][0-9].txt")), key=_natkey)

    # align by common indices if counts differ
    if len(img_files) != len(pose_files):
        to_key = lambda p: os.path.splitext(os.path.basename(p))[0]
        img_map  = {to_key(p): p for p in img_files}
        pose_map = {to_key(p): p for p in pose_files}
        keys = sorted(set(img_map) & set(pose_map), key=lambda k: int(k))
        img_files  = [img_map[k]  for k in keys]
        pose_files = [pose_map[k] for k in keys]

    W_ref, H_ref = ref_size
    fx_ref, fy_ref = K_ref[0,0], K_ref[1,1]
    cx_ref, cy_ref = K_ref[0,2], K_ref[1,2]

    gen_rgb_list, gen_T_wc_list, gen_K_list = [], [], []

    for img_path, pose_path in zip(img_files, pose_files):
        # --- load image ---
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        rgb = np.array(im, dtype=np.uint8)

        # --- scale intrinsics from ref size to this image size ---
        sx, sy = W / float(W_ref), H / float(H_ref)
        K = np.array([[fx_ref * sx, 0.0,        cx_ref * sx],
                      [0.0,         fy_ref * sy, cy_ref * sy],
                      [0.0,         0.0,         1.0      ]], dtype=np.float32)

        # --- relative pose on disk: C2W_rel = inv(C2W_ref) @ C2W_view ---
        C2W_rel = np.loadtxt(pose_path).astype(np.float32).reshape(4,4)

        # absolute C2W for this generated view
        C2W_view = C2W_ref @ C2W_rel

        # world->camera for downstream projection
        T_wc = np.linalg.inv(C2W_view).astype(np.float32)

        gen_rgb_list.append(rgb)
        gen_T_wc_list.append(T_wc)
        gen_K_list.append(K_ref)

    return gen_rgb_list, gen_T_wc_list, gen_K_list

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
