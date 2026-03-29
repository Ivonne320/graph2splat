import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


def _load_images(img_path: str, angles: list[float], size: int) -> list[Image.Image]:
    img = Image.open(img_path).convert("RGB")
    images = []
    for angle in angles:
        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False)
        images.append(rotated.resize((size, size)))
    return images


def _upright_angle_from_pose(pose_path: str) -> float:
    pose = np.loadtxt(pose_path).reshape(4, 4)
    r_cw = pose[:3, :3]
    r_wc = r_cw.T
    up_w = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up_cam = r_wc @ up_w
    norm_xy = np.linalg.norm(up_cam[:2])
    if norm_xy < 1e-6:
        return 0.0
    angle = np.degrees(np.arctan2(up_cam[0], -up_cam[1]))
    return float(angle)


def _save_upright_preview(
    img_path: str, angle_deg: float, size: int, out_path: str
) -> None:
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((size, size))
    img_rot = img.rotate(angle_deg, resample=Image.BICUBIC, expand=False).resize(
        (size, size)
    )
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.axis("off")
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(img_rot)
    plt.axis("off")
    plt.title(f"Rotated {angle_deg:.1f}°")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)


def _to_batch(images: list[Image.Image]) -> torch.Tensor:
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    tensors = [preprocess(im) for im in images]
    return torch.stack(tensors, dim=0).cuda()


def _load_model() -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    model.eval().cuda()
    return model


def _extract_patch_tokens(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        feats = model.forward_features(batch)
        if "x_norm_patchtokens" in feats:
            return feats["x_norm_patchtokens"]  # (B, Npatch, C)
        tok = feats.get("x_prenorm")
        if tok is None:
            raise KeyError("DINOv2 features missing x_norm_patchtokens/x_prenorm")
        reg = getattr(model, "num_register_tokens", 0)
        tok = tok[:, 1 + reg :, :]
        return tok


def _tokens_to_maps(patch_tokens: torch.Tensor) -> torch.Tensor:
    bsz, n_patch, c_dim = patch_tokens.shape
    h_patch = w_patch = int(n_patch**0.5)
    return patch_tokens.reshape(bsz, h_patch, w_patch, c_dim).permute(0, 3, 1, 2)


def _pca_project_batch(patch_tokens: torch.Tensor) -> tuple[np.ndarray, int, int]:
    tokens = patch_tokens.detach().cpu().numpy()
    bsz, n_patch, c_dim = tokens.shape
    h_patch = w_patch = int(n_patch**0.5)

    flat = tokens.reshape(bsz * n_patch, c_dim).T  # (C, B*N)
    mean = flat.mean(axis=1, keepdims=True)
    flat_centered = flat - mean
    cov = np.cov(flat_centered)
    _, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, -3:]

    maps = []
    for b in range(bsz):
        xb = tokens[b].T - mean
        proj = eigvecs.T @ xb  # (3, N)
        maps.append(proj.T.reshape(h_patch, w_patch, 3))

    stack = np.stack(maps, axis=0)  # (B, H, W, 3)
    pca_min = stack.min(axis=(0, 1, 2), keepdims=True)
    pca_max = stack.max(axis=(0, 1, 2), keepdims=True)
    stack = (stack - pca_min) / (pca_max - pca_min + 1e-6)
    return stack, h_patch, w_patch


def _upsample_pca(pca_stack: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(pca_stack).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=(size, size), mode="bicubic", align_corners=False)
    return tensor.permute(0, 2, 3, 1).numpy()


def _rotation_grid(
    size: int,
    h_patch: int,
    w_patch: int,
    angle_deg: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if abs(angle_deg) < 1e-6:
        yy, xx = torch.meshgrid(
            torch.arange(h_patch, device=device, dtype=torch.float32),
            torch.arange(w_patch, device=device, dtype=torch.float32),
            indexing="ij",
        )
        x_norm = 2.0 * xx / (w_patch - 1) - 1.0
        y_norm = 2.0 * yy / (h_patch - 1) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
        valid = torch.ones_like(x_norm, dtype=torch.bool)
        return grid, valid
    patch = size / float(w_patch)
    yy, xx = torch.meshgrid(
        torch.arange(h_patch, device=device, dtype=torch.float32),
        torch.arange(w_patch, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x = (xx + 0.5) * patch - 0.5
    y = (yy + 0.5) * patch - 0.5
    cx = (size - 1) * 0.5
    cy = (size - 1) * 0.5
    theta = torch.tensor(angle_deg * np.pi / 180.0, device=device)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x_rot = cos_t * (x - cx) - sin_t * (y - cy) + cx
    y_rot = sin_t * (x - cx) + cos_t * (y - cy) + cy
    x_patch = (x_rot + 0.5) / patch - 0.5
    y_patch = (y_rot + 0.5) / patch - 0.5
    x_norm = 2.0 * x_patch / (w_patch - 1) - 1.0
    y_norm = 2.0 * y_patch / (h_patch - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
    valid = (x_norm >= -1.0) & (x_norm <= 1.0) & (y_norm >= -1.0) & (y_norm <= 1.0)
    return grid, valid


def _feature_diff_maps_same_pixel(
    patch_tokens: torch.Tensor,
    angles: list[float],
    size: int,
) -> np.ndarray:
    maps = _tokens_to_maps(patch_tokens).float()
    maps = F.normalize(maps, dim=1)
    bsz, _, h_patch, w_patch = maps.shape
    base = maps[0:1]
    diffs = []
    for idx in range(bsz):
        grid, valid = _rotation_grid(
            size, h_patch, w_patch, angles[idx], maps.device
        )
        sampled = F.grid_sample(
            maps[idx : idx + 1], grid, mode="bilinear", align_corners=True
        )
        diff = (sampled - base).norm(dim=1, keepdim=True)
        diff = diff * valid.unsqueeze(0).unsqueeze(0)
        diffs.append(diff.squeeze(0).squeeze(0).cpu().numpy())
    return np.stack(diffs, axis=0)


def _upsample_diff(diff_stack: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(diff_stack).unsqueeze(1).float()
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.squeeze(1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        default="/cluster/project/cvg/Shared_datasets/3RScan/scenes/fcf66d88-622d-291c-871f-699b2d063630/sequence/frame-000000.color.jpg",
    )
    parser.add_argument("--angles", type=str, default="0,5,-5")
    parser.add_argument("--out_path", type=str, default="./results/dino_rotation_compare.png")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument(
        "--upright_angle",
        type=float,
        default=None,
        help="If set, save a side-by-side preview of the rotated image.",
    )
    parser.add_argument(
        "--pose_path",
        type=str,
        default=None,
        help="Optional pose file path to auto-compute upright rotation.",
    )
    args = parser.parse_args()

    angles = [float(a.strip()) for a in args.angles.split(",") if a.strip()]
    if 0.0 not in angles:
        angles = [0.0] + angles

    images = _load_images(args.img_path, angles, args.size)
    batch = _to_batch(images)
    model = _load_model()
    patch_tokens = _extract_patch_tokens(model, batch)

    pca_stack, _, _ = _pca_project_batch(patch_tokens)
    pca_up = _upsample_pca(pca_stack, args.size)
    diff_maps = _feature_diff_maps_same_pixel(patch_tokens, angles, args.size)
    diff_up = _upsample_diff(diff_maps, args.size)
    if diff_up.shape[0] > 1:
        diff_vmax = float(diff_up[1:].max())
    else:
        diff_vmax = float(diff_up.max())
    if diff_vmax <= 0:
        diff_vmax = 1.0

    n_cols = len(angles)
    plt.figure(figsize=(4 * n_cols, 9))
    for idx, (angle, img) in enumerate(zip(angles, images)):
        plt.subplot(3, n_cols, idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Input {angle:.1f}°")

        plt.subplot(3, n_cols, n_cols + idx + 1)
        plt.imshow(pca_up[idx])
        plt.axis("off")
        plt.title("DINO PCA")

        plt.subplot(3, n_cols, 2 * n_cols + idx + 1)
        plt.imshow(diff_up[idx], cmap="magma", vmin=0.0, vmax=diff_vmax)
        plt.axis("off")
        plt.title("Feat Δ (same pixel)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    plt.savefig(args.out_path, dpi=200)

    upright_angle = args.upright_angle
    if upright_angle is None and args.pose_path:
        upright_angle = _upright_angle_from_pose(args.pose_path)

    if upright_angle is not None:
        upright_out = os.path.splitext(args.out_path)[0] + "_upright.png"
        _save_upright_preview(args.img_path, upright_angle, args.size, upright_out)


if __name__ == "__main__":
    main()
