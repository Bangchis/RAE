from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .stage1_runtime import run_stage1_latent_stats
except ImportError:
    from stage1_runtime import run_stage1_latent_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage-1 latent normalization statistics with the JAX RAE path.")
    parser.add_argument("--config", required=True, help="Path to the repo YAML config.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory used to estimate latent stats.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .pt file containing mean/var tensors.")
    parser.add_argument("--batch-size", type=int, default=16)
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
    output_path = run_stage1_latent_stats(args)
    print(f"Saved Stage-1 latent stats to {output_path}")


if __name__ == "__main__":
    main()
