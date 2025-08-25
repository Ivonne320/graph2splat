import itertools
import logging
import os
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

    extrinsics = scan3r.load_frame_poses(
        data_dir=root_dir, scan_id=scan_id, frame_idxs=frame_idxs
    )
    intrinsics = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id)
    mask = scan3r.load_masks(data_dir=root_dir, scan_id=scan_id)
    rendered =  []
    for frame_id in frame_idxs:
        path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{frame_id}.color.jpg"
        with Image.open(path) as img:
            arr = np.array(img)  # Load full data while file is open
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        rendered.append(tensor)
        
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
    
    scene_output_dir = osp.join(args.model_dir, "files", mode, scan_id, "scene_level")
    voxel_path = osp.join(scene_output_dir, "voxel_output_dense.npz")
    mean_scale_path=osp.join(scene_output_dir, "mean_scale_dense.npz")
    if (
            osp.exists(mean_scale_path)
            and osp.exists(voxel_path)
            and "arr_0" in np.load(voxel_path)
            and not args.override
        ):
            _LOGGER.info(f"Skipping {scan_id} ")
            return
    try:
        
        # STEP 1: Segment the mesh
        # segmented_mesh = _segment_mesh(mesh, annos, obj_id, scan_id)
        object_vertex_mask = annos > 0
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
        
        mean, scale = _normalize_segmented_mesh(scene_mesh)
        # STEP 3: Voxelise the mesh
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            scene_mesh,
            1 / 64,
            min_bound=(-0.5, -0.5, -0.5),
            max_bound=(0.5, 0.5, 0.5),
        )
        voxel_grid = _dilate_voxels(voxel_grid)

        # STEP 4: Save mean and scale (Scene composition)
        if not args.dry_run:
            os.makedirs(os.path.dirname(mean_scale_path), exist_ok=True)
            np.savez(mean_scale_path, mean=mean, scale=scale)
            _LOGGER.info(f"Saved mean and scale to {mean_scale_path}")

        if (
            os.path.exists(voxel_path) and "arr_0" in np.load(voxel_path)
        ) and not args.override:
            _LOGGER.info(f"Skipping {scan_id}")
            return

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
        rendered_obj = torch.stack(rendered_obj).float()
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
        )  # Shape: (Nimages, Npoints, 2)

        # STEP 7: Normalize the projection to [-1, 1]
        projection = (
            projection
            / torch.Tensor([intrinsics["width"], intrinsics["height"]]).float()
        ) * 2.0 - 1.0

        # STEP 8: Get the DINO embeddings
        patch_embeddings = _get_dino_embedding(
            rendered_obj
        )  # Shape: (Nimages, 1024, 64, 64)
        patch_embeddings, (H_in, W_in, H_p, W_p) = _get_dino_embedding(rendered_obj.unsqueeze(0).cuda())

        # STEP 9: Match the embeddings to the projection
        patchtokens = (
            F.grid_sample(
                patch_embeddings,
                projection.cuda().unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(2)
            .permute(0, 2, 1)
            .cpu()
            .numpy()
        )  # Shape: (Nimages, Npoints, 1024)

        patchtokens = np.mean(patchtokens, axis=0).astype(
            np.float16
        )  # Shape: (Npoints, 1024)
        assert patchtokens.shape[0] == voxel_grid.shape[0]
        assert patchtokens.shape[1] == 1024
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
        assert voxel_grid.shape[-1] == 1027
        if not args.dry_run:
            _save_featured_voxel(
                voxel_grid,
                output_file=voxel_path,
            )
        
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
