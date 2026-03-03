# The code has been adapted from https://github.com/microsoft/TRELLIS/blob/main/trellis/models/structured_latent_vae/decoder_gs.py

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import itertools

_LOGGER = logging.getLogger(__name__) 
from src.models.backbones.base import SparseTransformerBase
from src.modules import sparse as sp
from src.representations import Gaussian
from utils.random_utils import hammersley_sequence

_REPRESENTATION_CONFIG = {
    "perturb_offset": True,
    "voxel_size": 1.5,
    "num_gaussians": 8,
    "2d_filter_kernel_size": 0.1,
    "3d_filter_kernel_size": 9e-4,
    # "3d_filter_kernel_size": 4e-3,
    "scaling_bias": 8e-4,
    "opacity_bias": 0.1,
    "scaling_activation": "softplus",
    "scale_clamp": (-20, 1.0),
    "opacity_clamp": (-20.0, 20.0),
    "sh_degree": 0,
    "lr": {
        "_xyz": 1.0,
        "_features_dc": 1.0,
        "_features_rest": 1.0,  
        "_scaling": 1.0,
        "_rotation": 0.1,
        "_opacity": 1.0,
    },
}
def _check_finite(name, t):
    if not torch.isfinite(t).all():
        bad = ~torch.isfinite(t)
        n_bad = bad.sum().item()
        print(f"[NaNCHK] {name}: non-finite={n_bad}/{t.numel()} "
              f"min={t[torch.isfinite(t)].min().item() if torch.isfinite(t).any() else 'NA'} "
              f"max={t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else 'NA'}")
        # Optional: save a small slice for inspection
        bad_idx = bad.nonzero(as_tuple=False)[:10]
        print(f"[NaNCHK] {name} sample bad idx:", bad_idx.tolist())
        raise RuntimeError(f"Non-finite in {name}")
def _summarize_nonfinite(feats, layout, max_fields=8, max_idxs=8) -> str:
    bad = ~torch.isfinite(feats)
    if not bad.any().item():
        return "all finite"
    M, C = feats.shape
    r0, c0 = bad.nonzero(as_tuple=False)[0].tolist()
    v0 = feats[r0, c0]
    parts = []
    for k, v in layout.items():
        a, b = v["range"]
        sl_bad = ~torch.isfinite(feats[:, a:b])
        if sl_bad.any().item():
            rr, cc = sl_bad.nonzero(as_tuple=False)[0].tolist()
            parts.append(f"{k}[{a}:{b}] -> (row={rr}, C={a+cc}, val={feats[rr, a+cc]})")
            if len(parts) >= max_fields:
                break
    bad_cols = bad.any(dim=0).nonzero(as_tuple=False).squeeze(1)[:max_idxs].tolist()
    return (f"bad={int(bad.sum().item())}/{M*C}, first=(row={r0}, C={c0}, val={v0}); "
            f"fields: {', '.join(parts) if parts else 'n/a'}; bad_channels(sample)={bad_cols}")

def _ping_sparse_fields(name: str, h: sp.SparseTensor, layout: dict, x: Optional[sp.SparseTensor]=None):
    feats = h.feats  # [M, C]

    # --- robust "any bad?" gate (no ambiguous tensor-in-if) ---
    bad_mask_all = ~torch.isfinite(feats)
    if not bad_mask_all.any().item():
        return

    print(f"\n[NaN-PING] {name}: feats has non-finite values "
          f"(bad={int(bad_mask_all.sum().item())}/{feats.numel()})")

    # Per-field (by your layout)
    for k, v in layout.items():
        a, b = v["range"]
        sl = feats[:, a:b]
        bad_mask = ~torch.isfinite(sl)
        if bad_mask.any().item():
            r, c = bad_mask.nonzero(as_tuple=False)[0].tolist()
            val = sl[r, c]
            print(f"  -> field '{k}' [{a}:{b}]  first bad at (row={r}, C={a+c})  val={val}")

    # Per-batch attribution (robust to slice/int/tensor/list)
    if x is None or not hasattr(x, "layout"):
        return

    def _rows_to_index_tensor(rows, total_rows: int) -> Optional[torch.Tensor]:
        if isinstance(rows, slice):
            start = 0 if rows.start is None else rows.start
            stop  = total_rows if rows.stop  is None else rows.stop
            step  = 1 if rows.step  is None else rows.step or 1
            return torch.arange(start, stop, step, device=feats.device, dtype=torch.long)
        if isinstance(rows, int):
            return torch.tensor([rows], device=feats.device, dtype=torch.long)
        if isinstance(rows, torch.Tensor):
            return rows.to(device=feats.device, dtype=torch.long)
        try:
            return torch.as_tensor(list(rows), device=feats.device, dtype=torch.long)
        except Exception:
            return None

    total_rows = feats.size(0)
    for bi, rows in enumerate(x.layout):
        idx = _rows_to_index_tensor(rows, total_rows)
        if idx is None or idx.numel() == 0:
            continue
        sub = feats.index_select(0, idx)
        bad_sub = ~torch.isfinite(sub)
        if bad_sub.any().item():
            rr, cc = bad_sub.nonzero(as_tuple=False)[0].tolist()
            print(f"  [batch {bi}] local(row={rr}, C={cc}) -> global(row={idx[rr].item()}, C={cc}), "
                  f"val={sub[rr, cc]}")
def _tripwire(name, t):
    bad = ~torch.isfinite(t)
    if bad.any().item():
        r, c = bad.nonzero(as_tuple=False)[0].tolist()
        print(f"[TRIP] {name}: first non-finite at (row={r}, C={c}), val={t[r, c]}")

class SLatGaussianDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config or _REPRESENTATION_CONFIG
        self._calc_layout()
        self.out_layer = sp.SparseLinear(model_channels, self.out_channels)
        self._build_rotations()
        self._build_perturbation()
        

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def _build_rotations(self) -> None:
        """
        Precompute the 24 proper rotations of the cube (det=+1).
        We'll index them per-voxel by hashing voxel integer coords, to avoid a global repeated pattern.
        """
        mats = []
        axes = torch.eye(3)
        for perm in itertools.permutations([0, 1, 2], 3):
            P = axes[list(perm)]  # permutation matrix
            for signs in itertools.product([-1.0, 1.0], repeat=3):
                S = torch.diag(torch.tensor(signs))
                R = S @ P
                if torch.det(R) > 0.5:  # det == +1 (numerical)
                    mats.append(R)
        mats = torch.stack(mats, dim=0).float()  # [24,3,3]
        self.register_buffer("rot_mats", mats, persistent=False)

    @staticmethod
    def _hash_coords(coords_ijk: torch.Tensor, mod: int) -> torch.Tensor:
        """
        coords_ijk: [V,3] int (voxel coords)
        returns: [V] in [0, mod)
        """
        # Large-ish primes for spatial hashing (works fine on int64)
        x = coords_ijk[:, 0].long()
        y = coords_ijk[:, 1].long()
        z = coords_ijk[:, 2].long()
        h = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
        h = torch.remainder(h, mod)
        return h

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _build_perturbation(self) -> None:
        perturbation = [
            hammersley_sequence(3, i, self.rep_config["num_gaussians"])
            for i in range(self.rep_config["num_gaussians"])
        ]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        perturbation = perturbation / self.rep_config["voxel_size"]
        perturbation = perturbation.clamp_(-1.0 + 1e-6, 1.0 - 1e-6)
        perturbation = torch.atanh(perturbation).to(self.device)
        self.register_buffer("offset_perturbation", perturbation)
        # """
        # Store a Kx3 template in Euclidean offset space (pre-tanh), in roughly "fraction of voxel" units.
        # We'll rotate it per voxel later.
        # """
        # base = [
        #     hammersley_sequence(3, i, self.rep_config["num_gaussians"])
        #     for i in range(self.rep_config["num_gaussians"])
        # ]
        # base = torch.tensor(base).float() * 2 - 1  # [-1,1], shape [K,3]

        # # Scale the template so it sits inside the voxel. Larger voxel_size -> smaller template in normalized space.
        # base = base / float(self.rep_config["voxel_size"])
        # base = base.clamp_(-1.0 + 1e-6, 1.0 - 1e-6)  # keep safely inside tanh range

        # # Keep this in Euclidean space (NOT atanh). We'll add it after tanh().
        # self.register_buffer("offset_perturb_euc", base.to(self.device), persistent=False)
 

    def _calc_layout(self) -> None:
        sh_degree = self.rep_config['sh_degree']
        num_sh_rest = (sh_degree + 1)**2 - 1  # exclude DC (1)
        num_features_rest = num_sh_rest * 3  # RGB per SH coef
        if sh_degree == 0:
            self.layout = {
                "_xyz": {
                    "shape": (self.rep_config["num_gaussians"], 3),
                    "size": self.rep_config["num_gaussians"] * 3,
                },
                "_features_dc": {
                    "shape": (self.rep_config["num_gaussians"], 1, 3),
                    "size": self.rep_config["num_gaussians"] * 3,
                },
                # "_features_rest": {
                #     "shape": (self.rep_config["num_gaussians"], num_sh_rest, 3),
                #     "size": self.rep_config["num_gaussians"] * num_features_rest,
                # },
                "_scaling": {
                    "shape": (self.rep_config["num_gaussians"], 3),
                    "size": self.rep_config["num_gaussians"] * 3,
                },
                "_rotation": {
                    "shape": (self.rep_config["num_gaussians"], 4),
                    "size": self.rep_config["num_gaussians"] * 4,
                },
                "_opacity": {
                    "shape": (self.rep_config["num_gaussians"], 1),
                    "size": self.rep_config["num_gaussians"],
                },
            }
        else:
            self.layout = {
                "_xyz": {
                    "shape": (self.rep_config["num_gaussians"], 3),
                    "size": self.rep_config["num_gaussians"] * 3,
                },
                "_features_dc": {
                    "shape": (self.rep_config["num_gaussians"], 1, 3),
                    "size": self.rep_config["num_gaussians"] * 3,
                },
                "_features_rest": {
                    "shape": (self.rep_config["num_gaussians"], num_sh_rest, 3),
                    "size": self.rep_config["num_gaussians"] * num_features_rest,
                },
                "_scaling": {
                    "shape": (self.rep_config["num_gaussians"], 3),
                    "size": self.rep_config["num_gaussians"] * 3,
                },
                "_rotation": {
                    "shape": (self.rep_config["num_gaussians"], 4),
                    "size": self.rep_config["num_gaussians"] * 4,
                },
                "_opacity": {
                    "shape": (self.rep_config["num_gaussians"], 1),
                    "size": self.rep_config["num_gaussians"],
                },
            }
            
        start = 0
        for k, v in self.layout.items():
            v["range"] = (start, start + v["size"])
            start += v["size"]
        self.out_channels = start

    def to_representation(self, x: sp.SparseTensor) -> List[Gaussian]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            representation = Gaussian(
                # sh_degree=1,
                sh_degree=self.rep_config["sh_degree"],
                aabb=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
                # mininum_kernel_size=0.002,
                scaling_bias=self.rep_config["scaling_bias"],
                opacity_bias=self.rep_config["opacity_bias"],
                scaling_activation=self.rep_config["scaling_activation"],
            )

            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
            for k, v in self.layout.items():
                if k == "_xyz":
                    offset = x.feats[x.layout[i]][
                        :, v["range"][0] : v["range"][1]
                    ].reshape(-1, *v["shape"])
                    offset = offset * self.rep_config["lr"][k]
                    # _LOGGER.info(f"perturb_offset: {self.rep_config["perturb_offset"]}")
                    # _LOGGER.info(f'perturb_offset: {self.rep_config["perturb_offset"]}')
                    if self.rep_config["perturb_offset"]:
                        offset = offset + self.offset_perturbation
                    offset = torch.nan_to_num(offset, nan=0.0, posinf=1e3, neginf=-1e3)
                    offset = (
                        torch.tanh(offset)
                        / self.resolution
                        * 0.5
                        * self.rep_config["voxel_size"]
                    )
                    _xyz = xyz.unsqueeze(1) + offset

                    # # sanitize then squash to [-1,1] in Euclidean space
                    # offset = torch.nan_to_num(offset, nan=0.0, posinf=1e3, neginf=-1e3)
                    # offset = torch.tanh(offset)  # now in [-1,1], shape [V,K,3]

                    # # Add a rotated per-voxel template AFTER tanh, to encourage intra-voxel coverage
                    # if self.rep_config["perturb_offset"]:
                    #     coords_ijk = x.coords[x.layout[i]][:, 1:].long()  # [V,3]
                    #     ridx = self._hash_coords(coords_ijk, int(self.rot_mats.shape[0]))  # [V]
                    #     R = self.rot_mats.index_select(0, ridx)  # [V,3,3]
                    #     # rotate template: [V,3,3] x [K,3] -> [V,K,3]
                    #     tmpl = torch.einsum("vij,kj->vki", R, self.offset_perturb_euc)
                    #     offset = offset + tmpl

                    # # Finally scale from "fraction of voxel" to normalized coords
                    # offset = offset / self.resolution * 0.5 * float(self.rep_config["voxel_size"])

                    # _xyz = xyz.unsqueeze(1) + offset
                    setattr(representation, k, _xyz.flatten(0, 1))
                else:
                    feats = (
                        x.feats[x.layout[i]][:, v["range"][0] : v["range"][1]]
                        .reshape(-1, *v["shape"])
                        .flatten(0, 1)
                    )
                    feats = feats * self.rep_config["lr"][k]
                    # if k == "_opacity":
                    #     # Clamp or squash to valid range to avoid NaNs in loss
                    #     # op_min, op_max = self.rep_config.get("opacity_clamp", (-20.0, 20.0))
                    #     # feats = torch.clamp(feats, op_min, op_max)
                    #     feats = feats
                    # elif k == "_scaling":
                    #     scale_min, scale_max = self.rep_config.get("scale_clamp", (-60.0, 1))
                    #     # feats = torch.clamp(feats, min=scale_min, max=scale_max)
                    #     feats = feats
                    # elif k == "_rotation":
                    #     feats = feats

                    # Optional NaN/Inf guard for all feats
                    feats = torch.nan_to_num(feats, nan=0.0, posinf=1e3, neginf=-1e3)
                    setattr(representation, k, feats)
            ret.append(representation)
        return ret

    def forward(self, x: sp.SparseTensor) -> List[Gaussian]:
        if (~torch.isfinite(x.feats)).any().item():
            _LOGGER.error("[input to decoder] %s", _summarize_nonfinite(x.feats, self.layout))
            raise RuntimeError("Non-finite in input feats to decoder")
        h = super().forward(x)
        h = h.type(x.dtype)
        if (~torch.isfinite(h.feats)).any().item():
            _LOGGER.error("[pre_layer_norm] %s", _summarize_nonfinite(h.feats, self.layout))
            raise RuntimeError("Non-finite in pre_layer_norm")
        # h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:],eps=1e-5 ))
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        if (~torch.isfinite(h.feats)).any().item():
            _LOGGER.error("[pre_out_layer_input] %s", _summarize_nonfinite(h.feats, self.layout))
            raise RuntimeError("Non-finite in pre_out_layer_input")

        h = self.out_layer(h)
        if (~torch.isfinite(h.feats)).any().item():
            _LOGGER.error("[post_out_layer_output] %s", _summarize_nonfinite(h.feats, self.layout))
            raise RuntimeError("Non-finite in post_out_layer_output")
        return self.to_representation(h)
