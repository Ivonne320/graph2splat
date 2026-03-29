#!/usr/bin/env python3
"""Sample frames for 3RScan test scenes."""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import sys
from typing import List

import numpy as np

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import scan3r

def load_test_scene_ids(list_root: str, split: str, scan_type: str, resplit: bool) -> List[str]:
    if scan_type == "scan":
        # prefix = "resplit_" if resplit else ""
        prefix = "resplit_"
        filename = f"{split}_{prefix}scans.txt"
    else:
        filename = f"{split}_scans_subscenes.txt"
    scan_list = osp.join(list_root, "files", filename)
    if not osp.isfile(scan_list):
        raise FileNotFoundError(f"Scan list not found: {scan_list}")
    raw_ids = np.genfromtxt(scan_list, dtype=str)
    if raw_ids.size == 0:
        return []
    if raw_ids.ndim == 0:
        ref_ids = [str(raw_ids)]
    else:
        ref_ids = raw_ids.tolist()

    scans_meta_path = osp.join(list_root, "files", "3RScan.json")
    if not osp.isfile(scans_meta_path):
        raise FileNotFoundError(f"3RScan metadata not found: {scans_meta_path}")
    with open(scans_meta_path, "r", encoding="utf-8") as f:
        scans_meta = json.load(f)

    ref_set = set(ref_ids)
    expanded: List[str] = []
    for entry in scans_meta:
        ref_scan = entry.get("reference")
        if ref_scan not in ref_set:
            continue
        expanded.append(ref_scan)
        for scan in entry.get("scans", []):
            sub_ref = scan.get("reference")
            if sub_ref:
                expanded.append(sub_ref)
    return expanded

def sample_frames_for_scene(scenes_dir: str, scene_id: str, num_input: int, num_eval: int):
    frame_ids = scan3r.load_frame_idxs(scenes_dir, scene_id)
    if not frame_ids:
        return []
    inputs = frame_ids if num_input >= len(frame_ids) else random.sample(frame_ids, num_input)
    results = []
    for fid in inputs:
        eval_pool = [f for f in frame_ids if f != fid]
        if not eval_pool:
            eval_ids = [fid]
        elif num_eval >= len(eval_pool):
            eval_ids = eval_pool[:]
        else:
            eval_ids = random.sample(eval_pool, num_eval)
        results.append((fid, sorted(eval_ids)))
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Sample frames for 3RScan test scenes")
    parser.add_argument("--root", required=True, help="Path to 3RScan root (scenes)")
    parser.add_argument("--output", required=True, help="Output txt path")
    parser.add_argument("--num-input", type=int, default=5)
    parser.add_argument("--num-eval", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scan-type", choices=["scan", "subscene"], default="scan")
    parser.add_argument("--resplit", action="store_true")
    parser.add_argument(
        "--list-root",
        default=REPO_ROOT,
        help="Directory containing files/{split}_scans*.txt (defaults to repo root)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    scenes_dir = osp.join(args.root, "scenes")
    scene_ids = load_test_scene_ids(args.list_root, args.split, args.scan_type, args.resplit)

    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for scene_id in scene_ids:
            samples = sample_frames_for_scene(scenes_dir, scene_id, args.num_input, args.num_eval)
            for input_fid, eval_ids in samples:
                f.write(" ".join([scene_id, input_fid] + eval_ids) + "\n")

if __name__ == "__main__":
    main()
