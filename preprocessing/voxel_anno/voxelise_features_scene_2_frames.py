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

# def _get_dino_embedding(images: torch.Tensor) -> torch.Tensor:
#     images = images.reshape(-1, 3, images.shape[-2], images.shape[-1]).cpu()
#     inputs = transform(images).cuda()
#     outputs = model(inputs, is_training=True)

#     n_patch = 518 // 14
#     bs = images.shape[0]
#     patch_embeddings = (
#         outputs["x_prenorm"][:, model.num_register_tokens + 1 :]
#         .permute(0, 2, 1)
#         .reshape(bs, 1024, n_patch, n_patch)
#     )
#     return patch_embeddings
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
            f"vis/scene_level_voxel_2_frame_inference.ply",
            show_color=True,
        )
    _save_featured_voxel(torch.from_numpy(payload), output_file=voxel_path)
    

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

def voxel_downsample_np(points, voxel_size):
    """Simple numpy voxel downsample (hash by floor(points/voxel))."""
    if points.size == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    # unique keys, keep the first occurrence index
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]

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
    fid0, fid1 = frame_idxs[0], frame_idxs[1]
    # print('frame_id:',frame_id)
    rgb_1_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{fid0}.color.jpg"
    rgb_2_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{fid1}.color.jpg"
    # depth_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{frame_id}.depth.pgm" 
    extrinsics = scan3r.load_frame_poses(
        data_dir=root_dir, scan_id=scan_id, frame_idxs=frame_idxs
    )
    
    intrinsics = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id)
    intrinsics = intrinsics["intrinsic_mat"]
    intrinsics_d = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id, type = 'depth')
    intrinsics_d = intrinsics_d["intrinsic_mat"]
    K_rgb  = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id)["intrinsic_mat"]
    K_depth= scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id, type='depth')["intrinsic_mat"]
    
    Wp0 = unproject_frame_to_world(fid0, extrinsics, K_rgb, K_depth, root_dir, scan_id)
    Wp1 = unproject_frame_to_world(fid1, extrinsics, K_rgb, K_depth, root_dir, scan_id)
    
    # Joint cloud
    world_points_joint = np.vstack([Wp0, Wp1])

    # world_points = (cam_points @ R_cw.T) + t_cw[None, :]  # (N,3)
    world_points = world_points_joint
    # o3d.io.write_point_cloud("vis/cloud_gt_2_frame.ply", world_points_joint)
    
    scene_output_dir = osp.join(args.model_dir, "files", mode, scan_id, "scene_level")
    voxel_path = osp.join(scene_output_dir, "voxel_output_dense.npz")
    mean_scale_path=osp.join(scene_output_dir, "mean_scale_dense.npz")
    mean, scale, pts_normed = _normalize_points(world_points)
   
    print('mean: ', mean)
    print('scale: ', scale)
    print('len pts_normed: ', len(pts_normed))
    vox_idx, voxel_centers_normed = _voxelize_points_normed(pts_normed, grid_size=64)  # (M,3) 
    vox_idx_dilated = _dilate_voxels_from_idx(vox_idx, grid_size=64)
    voxel_centers_normed = (vox_idx_dilated.astype(np.float32) + 0.5) / 64 - 0.5
    vox_idx = vox_idx_dilated
    print('len vox_idx: ', len(vox_idx))
    print('len voxel_centers_normed: ', len(voxel_centers_normed))
    # save mean/scale for later composition
    if not args.dry_run:
        os.makedirs(os.path.dirname(mean_scale_path), exist_ok=True)
        np.savez(mean_scale_path, mean=mean, scale=scale)
        _LOGGER.info(f"Saved mean and scale to {mean_scale_path}") 
        
    T_wc =np.linalg.inv(extrinsics[frame_id]).astype(np.float32)          # world->camera
    R_wc, t_wc = T_wc[:3, :3], T_wc[:3, 3]

    voxel = torch.from_numpy(voxel_centers_normed).float()  # (M,3) in [-0.5,0.5]
    # de-normalize voxel centers back to world coords
    
    voxel_world = voxel * torch.from_numpy(scale).float()[None, :] * 2  \
                + torch.from_numpy(mean).float()[None, :]
    
    with Image.open(rgb_1_path) as img:
        rgb_1 = np.asarray(img.convert("RGB"), dtype=np.uint8)
    with Image.open(rgb_2_path) as img:
        rgb_2 = np.asarray(img.convert("RGB"), dtype=np.uint8)
                
    # both view sampling           
  
    # frame_1_tokens_518 = _get_dino_embedding(_prep_image_for_dino(rgb_1))
    frame_1_tokens_518 , (H_in, W_in, H_p, W_p)= _get_dino_embedding(_prep_image_for_dino(rgb_1))
    # frame_2_tokens_518 = _get_dino_embedding(_prep_image_for_dino(rgb_2))
    frame_2_tokens_518, (H_in, W_in, H_p, W_p) = _get_dino_embedding(_prep_image_for_dino(rgb_2))
    
    # 3) World->Cam for REF (you already have this)
    T_wc_1 = np.linalg.inv(extrinsics[fid0]).astype(np.float32) 
    T_wc_2 = np.linalg.inv(extrinsics[fid1]).astype(np.float32) 
    K_ref = intrinsics
    # 6) Sample REF
    idx_keep_1, tokens_1 = _project_and_sample_dino(
        voxel_world=torch.from_numpy(voxel_world.numpy() if isinstance(voxel_world, torch.Tensor) else voxel_world),
        T_wc=T_wc_1,
        K=K_ref,
        img_hw=rgb_1.shape[:2],
        dino_tokens=frame_1_tokens_518,
    )

    # 7) Sample GEN
    idx_keep_2, tokens_2 = _project_and_sample_dino(
        voxel_world=torch.from_numpy(voxel_world.numpy() if isinstance(voxel_world, torch.Tensor) else voxel_world),
        T_wc=T_wc_2,
        K=K_ref,
        img_hw=rgb_2.shape[:2],
        dino_tokens=frame_2_tokens_518,
    )
    save_union_average(idx_keep_1, tokens_1, idx_keep_2, tokens_2, vox_idx, voxel_path)



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
    # parser.add_argument("--model", type=str, default="dinov2_vitl14_reg")
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

    # model = torch.hub.load("facebookresearch/dinov2", args.model)
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
