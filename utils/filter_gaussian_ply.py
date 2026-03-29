import argparse
import os

import numpy as np
from plyfile import PlyData, PlyElement


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def filter_gaussian_ply(
    input_path: str, output_path: str, opacity_threshold: float, keep_below: bool
) -> None:
    ply = PlyData.read(input_path)
    if "vertex" not in ply:
        raise ValueError(f"No vertex element in {input_path}")
    vertex = ply["vertex"]
    if "opacity" not in vertex.data.dtype.names:
        raise ValueError(f"No opacity field in {input_path}")

    opacity_raw = np.asarray(vertex["opacity"], dtype=np.float64)
    opacity = _sigmoid(opacity_raw)
    mask = opacity <= opacity_threshold if keep_below else opacity >= opacity_threshold

    filtered = vertex.data[mask]
    elements = [PlyElement.describe(filtered, "vertex")]
    elements.extend(el for el in ply.elements if el.name != "vertex")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    PlyData(elements, text=ply.text).write(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter Gaussian PLY by opacity and save a new PLY."
    )
    parser.add_argument("--input", required=True, help="Input Gaussian PLY path.")
    parser.add_argument("--output", required=True, help="Output Gaussian PLY path.")
    parser.add_argument(
        "--opacity_threshold",
        type=float,
        default=0.5,
        help="Opacity threshold in [0,1].",
    )
    parser.add_argument(
        "--keep_below",
        action="store_true",
        help="Keep opacities <= threshold (default keeps >= threshold).",
    )
    args = parser.parse_args()

    filter_gaussian_ply(
        args.input, args.output, args.opacity_threshold, args.keep_below
    )


if __name__ == "__main__":
    main()
