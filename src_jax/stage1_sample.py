from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .stage1_runtime import run_stage1_reconstruction
except ImportError:
    from stage1_runtime import run_stage1_reconstruction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct a single image with the JAX RAE stage-1 path.")
    parser.add_argument("--config", required=True, help="Path to the repo YAML config.")
    parser.add_argument("--image", type=Path, default=Path("assets/pixabay_cat.png"))
    parser.add_argument("--output", type=Path, default=Path("recon_jax.png"))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--backend-dir", default=None)
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="OmegaConf override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_stage1_reconstruction(args)


if __name__ == "__main__":
    main()
