from __future__ import annotations

import argparse

try:
    from .stage2_runtime import run_stage2_sampling_ddp
except ImportError:
    from stage2_runtime import run_stage2_sampling_ddp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed Stage-2 sampling on JAX/NNX.")
    parser.add_argument("--config", required=True, help="Path to the repo YAML config.")
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint override.")
    parser.add_argument("--sample-dir", default="samples_jax", help="Directory for PNG outputs.")
    parser.add_argument("--num-samples", type=int, default=5_000, help="Total number of images to generate.")
    parser.add_argument("--per-proc-batch-size", type=int, default=4, help="Batch size generated per JAX process.")
    parser.add_argument("--label-sampling", choices=["equal", "random"], default="equal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--save-npz", action="store_true", help="Also save per-rank NPZ shards.")
    parser.add_argument("--fid-ref", default=None, help="Optional .npz reference stats for FID.")
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--fid-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fid-num-threads", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--backend-dir", default=None)
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="OmegaConf override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_stage2_sampling_ddp(args)


if __name__ == "__main__":
    main()
