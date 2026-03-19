from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .stage1_runtime import run_stage1_folder_reconstruction
except ImportError:
    from stage1_runtime import run_stage1_folder_reconstruction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct an image folder with the JAX RAE stage-1 path.")
    parser.add_argument("--config", required=True, help="Path to the repo YAML config.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory to reconstruct.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for reconstructed images.")
    parser.add_argument("--output-ext", default=".png", help="Extension used for reconstructed files.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0, help="Threaded image loading workers.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of images.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--backend-dir", default=None)
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="OmegaConf override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run_stage1_folder_reconstruction(args)
    print(f"Saved reconstructed images to {output_dir}")


if __name__ == "__main__":
    main()
