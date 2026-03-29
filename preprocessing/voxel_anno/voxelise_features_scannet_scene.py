import itertools
import logging
import os
import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import Tuple

import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
import torch.nn.functional as F
import utils3d
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm

from configs import Config, update_configs
from utils import common, scannet
from utils import visualisation as vis

_LOGGER = logging.getLogger(__name__)

device = o3c.Device("CUDA", 0) if torch.cuda.is_available() else o3c.Device("CPU:0")
point_dtype = o3c.float32
dist_th = 0.05


@torch.no_grad()
def _get_dino_embedding(
    images: torch.Tensor, batch_size: int = 8, image_size: int = 518
) -> torch.Tensor:
    """
    images: (N,3,H,W) in [0,1]
    Returns (N,C,Hp,Wp) patch embeddings from DINOv2.
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
            raise KeyError("DINOv2 features missing x_norm_patchtokens/x_prenorm")
        n_patch = int(tok.shape[1] ** 0.5)
        if n_patch * n_patch != tok.shape[1]:
            raise ValueError(f"Non-square patch grid: {tok.shape[1]} tokens")
        tok = tok.transpose(1, 2).reshape(chunk.size(0), tok.shape[2], n_patch, n_patch)
        patches.append(tok.detach().float())
    return torch.cat(patches, dim=0)


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
    voxel = ((voxel.float() + 0.5) * voxel_size) - 0.5
    voxel = voxel * (2.0 * scale[None, :]) + mean[None, :]
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
    scale = np.max(np.abs(vertices), axis=0)
    scale[scale == 0] = 1.0
    vertices *= 1.0 / (2 * scale)
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    segmented_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mean, scale


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
    uv = projection_px.to(device=patchtokens.device)
    valid = (
        (uv[..., 0] >= 0)
        & (uv[..., 0] < W)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < H)
    )
    valid = valid.unsqueeze(-1).to(dtype=patchtokens.dtype)
    weighted = patchtokens * valid
    denom = valid.sum(dim=0).clamp_min(1.0)
    return weighted.sum(dim=0) / denom


def _grid_sample_safe(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
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


def voxelize_mesh_simple_dense(scene_mesh, G=64, n_rand=2, k_dilate=2, device="cuda"):
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

    mids01 = (v0 + v1) * 0.5
    mids12 = (v1 + v2) * 0.5
    mids20 = (v2 + v0) * 0.5
    cents = (v0 + v1 + v2) / 3.0

    pts_list = [V, mids01, mids12, mids20, cents]

    if n_rand > 0:
        u = np.random.rand(Fidx.shape[0], n_rand, 1).astype(np.float32)
        v = np.random.rand(Fidx.shape[0], n_rand, 1).astype(np.float32)
        swap = (u + v > 1).astype(np.float32)
        u = u * (1 - swap) + (1 - u) * swap
        v = v * (1 - swap) + (1 - v) * swap
        w = 1.0 - u - v
        tri = (u * v0[:, None, :] + v * v1[:, None, :] + w * v2[:, None, :]).reshape(
            -1, 3
        )
        pts_list.append(tri)

    pts = np.concatenate(pts_list, axis=0)

    eps = 1e-6
    idx = np.floor((pts + 0.5 - eps) * G).astype(np.int32)
    idx = np.clip(idx, 0, G - 1)
    idx = np.unique(idx, axis=0)

    if k_dilate > 0 and idx.shape[0] > 0:
        occ = torch.zeros((1, 1, G, G, G), device=device, dtype=torch.uint8)
        occ[0, 0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        pad = k_dilate // 2
        occ = F.max_pool3d(occ.float(), kernel_size=k_dilate, stride=1, padding=pad)
        xyz = (occ > 0.5).nonzero(as_tuple=False)
        if xyz.numel() > 0:
            idx = (
                torch.stack([xyz[:, 2], xyz[:, 3], xyz[:, 4]], dim=1)
                .detach()
                .cpu()
                .numpy()
            )

    return idx.astype(np.int64)


def load_mesh_annotations(scan_id: str) -> dict:
    mesh_path = scannet.load_mesh_path(data_split_dir=root_dir, scan_id=scan_id)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh_vertices = np.asarray(mesh.vertices)
    # load pred pcls
    sgfusion_pcl_file = osp.join(
        root_dir, "scene_graph_fusion", scan_id, "inseg_filtered.ply"
    )
    if osp.exists(sgfusion_pcl_file) is False:
        print(f"File not found {sgfusion_pcl_file}")
        return
    sgfusion_data = PlyData.read(sgfusion_pcl_file)["vertex"]
    sgfusion_points = np.stack(
        [sgfusion_data["x"], sgfusion_data["y"], sgfusion_data["z"]], axis=1
    )
    sgfusion_labels = np.asarray(sgfusion_data["label"])

    # transfer ply labels to mesh by open3d knn search
    ## generate kdtree for sgfusion points
    sgfusion_points_tensor = o3c.Tensor(
        sgfusion_points, dtype=point_dtype, device=device
    )
    kdtree_sgfusion = o3c.nns.NearestNeighborSearch(sgfusion_points_tensor)
    kdtree_sgfusion.knn_index()
    ## knn search
    mesh_vertices_tensor = o3c.Tensor(mesh_vertices, dtype=point_dtype, device=device)
    [idx, dist] = kdtree_sgfusion.knn_search(mesh_vertices_tensor, 1)
    dist_arr = (dist.cpu().numpy()).reshape(-1)
    idx_arr = (idx.cpu().numpy()).reshape(-1)
    valid_idx = dist_arr < dist_th**2
    ## get mesh labels
    mesh_obj_labels = np.zeros(mesh_vertices.shape[0], dtype=np.int32)
    mesh_obj_labels[valid_idx] = sgfusion_labels[idx_arr[valid_idx]]
    assert mesh_obj_labels.shape[0] == mesh_vertices.shape[0]
    return mesh, mesh_obj_labels


@torch.no_grad()
def voxelise_features(
    obj_data: list[str],
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

    frame_idxs = scannet.load_frame_idxs(
        data_split_dir=root_dir, scan_id=scan_id, skip=cfg.data.img.img_step
    )
    extrinsics = scannet.load_frame_poses(
        data_split_dir=root_dir, scan_id=scan_id, skip=cfg.data.img.img_step
    )
    intrinsics = scannet.load_frame_intrinsics(data_split_dir=root_dir, scan_id=scan_id)
    mask = scannet.load_masks(data_dir=root_dir, scan_id=scan_id)

    frame_idxs = [fid for fid in frame_idxs if fid in extrinsics]
    rendered = []
    for frame_id in frame_idxs:
        with Image.open(f"{root_dir}/scenes/{scan_id}/data/color/{frame_id}.jpg") as img:
            arr = np.array(img)
        rendered.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)

    mesh, annos = load_mesh_annotations(scan_id)
    object_ids = [int(obj_id) for obj_id in obj_data]

    scene_output_dir = osp.join(
        args.model_dir,
        "files",
        mode,
        scan_id,
        "scene_level_no_dilation_128_dinov2",
    )
    voxel_path = osp.join(scene_output_dir, "voxel_output_dense.npz")
    mean_scale_path = osp.join(scene_output_dir, "mean_scale_dense.npz")
    feat_dim = getattr(model, "embed_dim", None)
    if (
        osp.exists(mean_scale_path)
        and osp.exists(voxel_path)
        and not args.override
    ):
        try:
            voxel_npz = np.load(voxel_path)
            if "arr_0" in voxel_npz:
                existing = voxel_npz["arr_0"]
                existing_dim = (
                    existing.shape[1] - 3
                    if existing.ndim == 2 and existing.shape[1] > 3
                    else None
                )
                if (
                    feat_dim is not None
                    and existing_dim is not None
                    and existing_dim != feat_dim
                ):
                    _LOGGER.warning(
                        "Existing voxel feat dim %s != model dim %s for %s; regenerating.",
                        existing_dim,
                        feat_dim,
                        scan_id,
                    )
                else:
                    _LOGGER.info(f"Skipping {scan_id}")
                    return
        except (OSError, KeyError, ValueError) as exc:
            _LOGGER.warning(
                "Failed to read existing voxel file %s (%s); regenerating.",
                voxel_path,
                exc,
            )

    try:
        object_vertex_mask = np.isin(annos, object_ids)
        selected_vertices = np.where(object_vertex_mask)[0]
        if selected_vertices.size == 0:
            _LOGGER.warning(f"No object vertices found for {scan_id}, skipping.")
            return
        faces = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        face_mask = np.all(np.isin(faces, selected_vertices), axis=1)
        selected_faces = faces[face_mask]
        index_map = {
            old_idx: new_idx for new_idx, old_idx in enumerate(selected_vertices)
        }
        remapped_faces = np.vectorize(index_map.get)(selected_faces)

        scene_mesh = o3d.geometry.TriangleMesh()
        scene_mesh.vertices = o3d.utility.Vector3dVector(vertices[selected_vertices])
        scene_mesh.triangles = o3d.utility.Vector3iVector(remapped_faces)

        mean, scale = _normalize_segmented_mesh(scene_mesh)
        G = 128
        voxel_grid = voxelize_mesh_simple_dense(
            scene_mesh,
            G=G,
            n_rand=0,
            k_dilate=0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        if not args.dry_run:
            os.makedirs(os.path.dirname(mean_scale_path), exist_ok=True)
            np.savez(mean_scale_path, mean=mean, scale=scale)
            _LOGGER.info(f"Saved mean and scale to {mean_scale_path}")

        pose_camera_to_world = [np.linalg.inv(extrinsics[fid]) for fid in frame_idxs]
        masks = [mask[frame_id] for frame_id in frame_idxs]
        masks = [np.isin(m, object_ids).astype(np.uint8) for m in masks]
        rendered_obj = [
            image * mask[None, :, :] for image, mask in zip(rendered, masks)
        ]
        idx_empty = [i for i, r in enumerate(rendered_obj) if r.sum() == 0]
        rendered_obj = [r for i, r in enumerate(rendered_obj) if i not in idx_empty][
            :300
        ]
        pose_camera_to_world = [
            pose
            for i, pose in enumerate(pose_camera_to_world)
            if i not in idx_empty
        ][:300]

        if len(rendered_obj) == 0:
            _LOGGER.warning(f"No rendered frames for {scan_id}, skipping.")
            return
        rendered_obj = torch.stack(rendered_obj).float()

        projection = _project_to_image(
            torch.Tensor(voxel_grid),
            torch.Tensor(mean),
            torch.Tensor(scale),
            torch.from_numpy(np.stack(pose_camera_to_world)),
            torch.from_numpy(intrinsics["intrinsic_mat"]),
            grid_size=(G, G, G),
        )
        projection_px = projection
        projection = (
            projection_px
            / torch.Tensor([intrinsics["width"], intrinsics["height"]]).float()
        ) * 2.0 - 1.0

        patch_embeddings = _get_dino_embedding(rendered_obj)
        patchtokens = (
            _grid_sample_safe(
                patch_embeddings.float(),
                projection.cuda().unsqueeze(1).float(),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(2)
            .permute(0, 2, 1)
        )

        patchtokens = _average_patchtokens(
            patchtokens,
            projection_px,
            (intrinsics["width"], intrinsics["height"]),
            use_valid_mask=args.use_valid_mask,
        ).cpu().numpy().astype(np.float16)

        assert patchtokens.shape[0] == voxel_grid.shape[0]
        assert patchtokens.shape[1] == patch_embeddings.shape[1]
        assert voxel_grid.shape[1] == 3

        voxel_grid = torch.concatenate(
            [torch.Tensor(voxel_grid), torch.Tensor(patchtokens)], dim=1
        )
        if args.visualize:
            vis.save_voxel_as_ply(
                voxel_grid.cpu().numpy(),
                f"vis/{scan_id}_scene_level_voxel.ply",
                show_color=True,
            )
        if not args.dry_run:
            _save_featured_voxel(
                voxel_grid,
                output_file=voxel_path,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        _LOGGER.exception(f"Error processing {scan_id}: {e}")


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

    subscan_ids_generated = np.genfromtxt(
        osp.join(cfg.data.root_dir, "files", "scannet_test_split.txt"),
        dtype=str,
    )
    all_subscan_ids = subscan_ids_generated
    all_subscan_ids = all_subscan_ids[:10]
    subscan_ids_processed = []
    for subscan_id in tqdm(all_subscan_ids):
        if not osp.exists(
            osp.join(
                cfg.data.root_dir,
                "scene_graph_fusion",
                subscan_id,
                "{}.pkl".format(subscan_id),
            )
        ):
            _LOGGER.warning(f"Skipping {subscan_id} - no scene graph annotations")
            continue
        scene_graph_dict = common.load_pkl_data(
            osp.join(
                cfg.data.root_dir,
                "scene_graph_fusion",
                subscan_id,
                "{}.pkl".format(subscan_id),
            )
        )
        obj_data = scene_graph_dict["objects_id"]
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
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Force reload DINO weights from torch hub cache (use if embed_dim mismatches).",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_dir", type=str, default="vis")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument(
        "--use_valid_mask",
        action="store_true",
        help="Mask out projections that fall outside the image before averaging DINO features.",
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == "__main__":
    common.init_log(level=logging.INFO)
    _LOGGER.info("**** Starting feature voxelisation for ScanNet ****")
    args, unknown = parse_args()
    os.makedirs(args.vis_dir, exist_ok=True)
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    root_dir = cfg.data.root_dir

    model = torch.hub.load(
        "facebookresearch/dinov2", args.model, force_reload=args.force_reload
    )
    model.eval().cuda()
    model_dim = getattr(model, "embed_dim", None)
    expected_dims = {
        "dinov2_vits14_reg": 384,
        "dinov2_vitb14_reg": 768,
        "dinov2_vitl14_reg": 1024,
        "dinov2_vitg14_reg": 1536,
    }
    expected_dim = expected_dims.get(args.model)
    if expected_dim is not None and model_dim is not None and model_dim != expected_dim:
        raise ValueError(
            f"Model {args.model} embed_dim={model_dim} (expected {expected_dim}). "
            "Your torch hub cache may be stale; rerun with --force_reload or clear "
            "~/.cache/torch/hub/facebookresearch_dinov2_main."
        )
    scan_ids = process_data(cfg, mode="gs_annotations_scannet", split=args.split)
