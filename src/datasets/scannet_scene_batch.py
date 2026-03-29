import logging
import os
import os.path as osp
import random
from collections import defaultdict
from typing import Union

import numpy as np
import torch
import torch.multiprocessing
import torch.utils.data as data
import tqdm

from configs import Config
from src.modules.sparse.basic import SparseTensor, sparse_batch_cat, sparse_cat
from utils import common, scannet

_LOGGER = logging.getLogger(__name__)


class ScanNetSceneBatchDataset(data.Dataset):
    def __init__(self, cfg: Config, split: str = "test"):
        self.cfg = cfg
        self.cfg.data.preload_slat = False

        self.seed = cfg.seed
        random.seed(self.seed)

        self.split = split
        self.preload_masks = cfg.data.preload_masks
        self.load_mesh = True if self.split == "test" else False
        self.suffix = "_dense" if cfg.data.from_gt else ""
        self.data_root_dir = cfg.data.root_dir
        self.scans_scenes_dir = osp.join(cfg.data.root_dir, "scenes")
        self.scans_files_dir = osp.join(cfg.data.root_dir, "files")
        self.rescan = cfg.data.rescan
        self.img_step = cfg.data.img.img_step
        self.pc_resolution = (
            self.cfg.train.pc_res if self.split == "train" else self.cfg.val.pc_res
        )
        self.scene_splat_subdir = getattr(
            cfg.data, "scene_splat_subdir", "scene_level_no_dilation_128_dinov2"
        )

        self.img_patch_feat_dim = self.cfg.autoencoder.encoder.img_patch_feat_dim
        self.obj_patch_num = self.cfg.data.scene_graph.obj_patch_num
        self.obj_topk = self.cfg.data.scene_graph.obj_topk
        self.use_pos_enc = self.cfg.autoencoder.encoder.use_pos_enc

        self._load_scan_ids()
        self._load_images()
        self._load_extrinsics()
        self._load_intrinsics()
        self._load_scene_graphs()
        self._load_gt_annos()

        self.load_split_frames()
        self.data_items = self._generate_data_items()
        _LOGGER.info(f"Total data items: {len(self.data_items)}")

    def _load_gt_annos(self):
        self.gt_2D_anno_folder = osp.join(
            self.scans_files_dir, "gt_projection/obj_id_pkl"
        )
        self.obj_2D_annos_pathes = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="2D annos"):
            self.obj_2D_annos_pathes[scan_id] = osp.join(
                self.gt_2D_anno_folder, "{}.pkl".format(scan_id)
            )

    def _load_scan_ids(self):
        subscan_ids_generated = np.genfromtxt(
            osp.join(self.cfg.data.root_dir, "files", "scannet_test_split.txt"),
            dtype=str,
        )
        self.rooms_info = {
            room_id: [
                scan_id
                for scan_id in subscan_ids_generated
                if scan_id.startswith(room_id)
            ]
            for room_id in np.unique(
                [scan_id.split("_")[0] for scan_id in subscan_ids_generated]
            )
        }

        self.all_scans_split = [
            scan_id for scans in self.rooms_info.values() for scan_id in scans
        ]
        self.scan_ids = [
            next(
                (
                    room_scan
                    for room_scan in room_scans
                    if osp.exists(
                        osp.join(
                            self.cfg.data.root_dir, "scene_graph_fusion", room_scans[0]
                        )
                    )
                ),
                None,
            )
            for room_scans in self.rooms_info.values()
        ]

        self.scan_ids = [scan_id for scan_id in self.scan_ids if scan_id is not None]
        self.scan2room = {
            scan_id: room_id
            for room_id, scans in self.rooms_info.items()
            for scan_id in scans
        }

        if self.rescan:
            self.scan_ids = self.all_scans_split

        if self.cfg.mode == "debug_few_scan":
            self.scan_ids = self.scan_ids[: int(0.1 * len(self.scan_ids))]

        # TODO(gaia): due to timing constraints
        self.scan_ids = self.scan_ids[:2]
        # self.scan_ids 
        _LOGGER.info(f"Total scans: {len(self.scan_ids)}")

    def _load_images(self):
        self.frame_idxs = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Frames"):
            frame_idxs = scannet.load_frame_idxs(
                self.data_root_dir, scan_id, self.img_step
            )
            self.frame_idxs[scan_id] = frame_idxs

    def _load_extrinsics(self):
        self.image_poses = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Extrinsics"):
            image_poses = scannet.load_frame_poses(
                self.data_root_dir, scan_id, self.img_step, type="quat_trans"
            )
            self.image_poses[scan_id] = image_poses

    def _load_intrinsics(self):
        self.image_intrinsics = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Intrinsics"):
            intrinsics = scannet.load_frame_intrinsics(self.data_root_dir, scan_id)
            self.image_intrinsics[scan_id] = intrinsics

    def _load_scene_graphs(self):
        sg_folder_name = "scene_graph_fusion"
        self.scene_graphs = {}
        self.obj_3D_anno = {}
        rel_dim = self.cfg.autoencoder.encoder.rel_dim
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Scene graphs"):
            if not osp.exists(
                osp.join(self.cfg.data.root_dir, sg_folder_name, scan_id)
            ):
                print(f"Scene graph not found for {scan_id}")
                continue
            sg_folder_scan = osp.join(self.cfg.data.root_dir, sg_folder_name, scan_id)
            points, _ = scannet.load_plydata_npy(osp.join(sg_folder_scan, "data.npy"))
            pcl_center = np.mean(points, axis=0)
            scene_graph_dict = common.load_pkl_data(
                osp.join(sg_folder_scan, "{}.pkl".format(scan_id))
            )
            object_ids = scene_graph_dict["objects_id"]
            global_object_ids = scene_graph_dict["objects_cat"]
            object_points = (
                scene_graph_dict["obj_points"][self.pc_resolution] - pcl_center
            )
            object_points = torch.from_numpy(object_points).type(torch.FloatTensor)

            data_dict = {}
            edges = scene_graph_dict["edges"]
            edges = torch.from_numpy(edges)
            if "bow_vec_object_edge_feats" in scene_graph_dict:
                bow_vec_obj_edge_feats = torch.from_numpy(
                    scene_graph_dict["bow_vec_object_edge_feats"]
                )
            else:
                bow_vec_obj_edge_feats = torch.zeros(edges.shape[0], rel_dim)
            data_dict["graph_per_edge_count"] = np.array([edges.shape[0]])
            data_dict["tot_bow_vec_object_edge_feats"] = bow_vec_obj_edge_feats
            rel_pose = torch.from_numpy(scene_graph_dict["rel_trans"])
            data_dict["tot_rel_pose"] = rel_pose
            data_dict["edges"] = edges
            if "bow_vec_object_attr_feats" in scene_graph_dict:
                bow_vec_obj_attr_feats = torch.from_numpy(
                    scene_graph_dict["bow_vec_object_attr_feats"]
                )
            else:
                attri_dim = self.cfg.autoencoder.encoder.attr_dim
                bow_vec_obj_attr_feats = torch.zeros(object_points.shape[0], attri_dim)
            data_dict["tot_bow_vec_object_attr_feats"] = bow_vec_obj_attr_feats

            data_dict["obj_ids"] = object_ids
            data_dict["tot_obj_pts"] = object_points
            data_dict["graph_per_obj_count"] = np.array([object_points.shape[0]])
            data_dict["tot_obj_count"] = object_points.shape[0]
            data_dict["scene_ids"] = [scan_id]
            data_dict["pcl_center"] = pcl_center
            data_dict["global_obj_ids"] = global_object_ids

            self.scene_graphs[scan_id] = data_dict

            self.obj_3D_anno[scan_id] = {}
            for idx, obj_id in enumerate(object_ids):
                self.obj_3D_anno[scan_id][obj_id] = (
                    scan_id,
                    obj_id,
                    global_object_ids[idx],
                )

    def load_split_frames(self):
        split_path = osp.join(
            self.scans_files_dir, "test_train_test_splits_scannet.json"
        )
        self.test_frames = defaultdict(dict)
        self.train_frames = defaultdict(dict)
        if not osp.exists(split_path):
            _LOGGER.warning(
                "Split file not found (%s); skipping per-object frame splits.",
                split_path,
            )
            return
        test_train_test_splits = common.load_json(split_path)
        for scan_id in self.scan_ids:
            if scan_id not in test_train_test_splits:
                continue
            for obj_id in test_train_test_splits[scan_id]:
                self.test_frames[scan_id][int(obj_id)] = test_train_test_splits[
                    scan_id
                ][obj_id]["test"]
                self.train_frames[scan_id][int(obj_id)] = test_train_test_splits[
                    scan_id
                ][obj_id]["train"]

    def _load_splats(
        self, scan_id: str, obj_id: int = None, preload_slat: bool = False
    ) -> dict:
        data_dict = {}
        if preload_slat:
            gs_path = os.path.join(
                self.scans_files_dir,
                "gs_embeddings",
                f"{scan_id}_slat.npz",
            )
            try:
                file = np.load(gs_path, mmap_mode="r")
                coords = file["coords"]
                feats = file["feats"]
                mean = torch.from_numpy(file["mean"]).float()
                scale = torch.from_numpy(file["scale"]).float()
                splat = SparseTensor(
                    feats=torch.from_numpy(feats).float(),
                    coords=torch.from_numpy(coords).int(),
                )
                if "obj_id" in file.files:
                    obj_id = file["obj_id"]
                if "R_cans" in file.files:
                    R_cans = torch.from_numpy(file["R_cans"]).float()
                elif "R_can" in file.files:
                    R_cans = torch.from_numpy(file["R_can"]).float()
                else:
                    R_cans = torch.eye(3, dtype=torch.float32)
            except (FileNotFoundError, KeyError):
                coords = torch.zeros((1, 3), device="cpu")
                feats = torch.zeros(
                    (coords.shape[0], self.img_patch_feat_dim), device="cpu"
                )
                splat = SparseTensor(feats=feats, coords=coords.int())
                mean = torch.zeros((3)).float()
                scale = torch.ones(()).float()
                R_cans = torch.eye(3, dtype=torch.float32)
        else:
            gs_path = os.path.join(
                self.scans_files_dir,
                "gs_annotations_scannet",
                scan_id,
                self.scene_splat_subdir,
                f"voxel_output{self.suffix}.npz",
            )
            try:
                file = np.load(gs_path, mmap_mode="r")
                gs = torch.from_numpy(file["arr_0"]).float()
                coords = gs[:, :3]
                feats = gs[:, 3:]
            except (FileNotFoundError, KeyError):
                coords = torch.zeros((1, 3), device="cpu")
                feats = torch.zeros(
                    (coords.shape[0], self.img_patch_feat_dim), device="cpu"
                )
            splat = SparseTensor(feats=feats, coords=coords.int())
            mean_scale_path = os.path.join(
                self.scans_files_dir,
                "gs_annotations_scannet",
                scan_id,
                self.scene_splat_subdir,
                f"mean_scale{self.suffix}.npz",
            )
            if os.path.exists(mean_scale_path):
                mean_scale = np.load(mean_scale_path)
                mean = torch.from_numpy(mean_scale["mean"]).float()
                scale = torch.from_numpy(mean_scale["scale"]).float()
            else:
                mean = torch.zeros((3)).float()
                scale = torch.ones(()).float()
            R_cans = torch.eye(3, dtype=torch.float32)

        if isinstance(R_cans, torch.Tensor) and R_cans.ndim == 3:
            R_cans = R_cans[0]

        data_dict["tot_obj_splat"] = splat
        data_dict["mean_obj_splat"] = mean
        data_dict["scale_obj_splat"] = scale
        data_dict["R_cans"] = R_cans
        data_dict["obj_id"] = obj_id
        data_dict["scan_id"] = scan_id
        return data_dict

    def _generate_data_items(self) -> list:
        data_items = []
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Data items"):
            if scan_id not in self.scene_graphs:
                continue
            frame_idxs = self.frame_idxs.get(scan_id, [])
            valid_frames = [
                fid for fid in frame_idxs if fid in self.image_poses.get(scan_id, {})
            ]
            for frame_idx in valid_frames:
                data_item_dict = {}
                data_item_dict["scan_id"] = scan_id
                data_item_dict["frame_idx"] = frame_idx
                data_items.append(data_item_dict)
        return data_items

    def _item_to_dict(self, data_item: dict) -> dict:
        data_dict = {}
        scan_id = data_item["scan_id"]
        data_dict["scan_id"] = scan_id
        data_dict["frame_idx"] = data_item["frame_idx"]
        return data_dict

    def _aggregate(
        self, data_dict: dict, key: str, mode: str
    ) -> Union[torch.Tensor, np.ndarray]:
        if mode == "torch_cat":
            return torch.cat([data[key] for data in data_dict])
        elif mode == "torch_stack":
            return torch.stack([data[key] for data in data_dict])
        elif mode == "np_concat":
            return np.concatenate([data[key] for data in data_dict])
        elif mode == "np_stack":
            return np.stack([data[key] for data in data_dict])
        elif mode == "sparse_cat":
            return sparse_batch_cat([data[key] for data in data_dict])
        else:
            raise NotImplementedError

    def _collate(self, batch: list) -> dict:
        scans_batch = [data["scan_id"] for data in batch if data is not None]
        batch_size = len(batch)
        data_dict = {"batch_size": batch_size}
        scene_graphs_ = {}
        scene_graphs_["scene_ids"] = np.array([[sid] for sid in scans_batch])
        scene_graphs_["frame_ids"] = [
            sample["frame_idx"] for sample in batch if sample is not None
        ]

        scene_graphs_["obj_intrinsics"] = {
            sid: self.image_intrinsics[sid] for sid in scans_batch
        }

        preload_slat = getattr(self.cfg.data, "preload_slat", False)
        splat_dicts = [
            self._load_splats(sid, preload_slat=False) for sid in scans_batch
        ]
        scene_graphs_["tot_obj_splat"] = sparse_cat(
            [d["tot_obj_splat"] for d in splat_dicts]
        ).float()

        def _reduce_vec3(t: torch.Tensor) -> torch.Tensor:
            t = t.float()
            if t.ndim == 0:
                return t.new_full((3,), t.item())
            if t.ndim == 1:
                if t.shape[0] == 3:
                    return t
                if t.shape[0] < 3:
                    pad = torch.zeros(3 - t.shape[0], device=t.device, dtype=t.dtype)
                    return torch.cat([t, pad], dim=0)
                return t[:3]
            flat = t.view(-1, t.shape[-1])
            if flat.shape[1] < 3:
                pad = torch.zeros(
                    flat.shape[0], 3 - flat.shape[1], device=t.device, dtype=t.dtype
                )
                flat = torch.cat([flat, pad], dim=1)
            return flat[:, :3].mean(dim=0)

        scene_graphs_["mean_obj_splat"] = torch.stack(
            [_reduce_vec3(d["mean_obj_splat"]) for d in splat_dicts]
        ).float()
        scene_graphs_["scale_obj_splat"] = torch.stack(
            [_reduce_vec3(d["scale_obj_splat"]) for d in splat_dicts]
        ).float()
        scene_graphs_["R_cans"] = torch.stack(
            [d["R_cans"] for d in splat_dicts]
        ).float()

        image_frames = defaultdict(list)
        for sample in batch:
            if sample is None:
                continue
            sid = sample["scan_id"]
            fid = sample["frame_idx"]
            image_frames[sid].append(fid)
        image_frames = {
            sid: list(dict.fromkeys(fids)) for sid, fids in image_frames.items()
        }
        scene_graphs_["image_frames"] = image_frames

        obj_2D_masks = {}
        for sid, fids in image_frames.items():
            scan_annos = common.load_pkl_data(self.obj_2D_annos_pathes[sid])
            object_ids = self.scene_graphs[sid]["obj_ids"]
            obj_2D_masks.setdefault(sid, {})
            for fid0 in fids:
                key = fid0 if fid0 in scan_annos else str(fid0)
                if key not in scan_annos:
                    continue
                obj_2D_masks[sid][fid0] = np.isin(scan_annos[key], object_ids)

        data_dict["scene_graphs"] = scene_graphs_
        data_dict["scene_graphs"]["obj_2D_masks"] = obj_2D_masks
        if len(batch) > 0:
            return data_dict
        return None

    def __getitem__(self, idx: int) -> dict:
        data_dict = self._item_to_dict(self.data_items[idx])
        return data_dict

    def collate_fn(self, batch: list) -> dict:
        return self._collate(batch)

    def __len__(self) -> int:
        return len(self.data_items)
