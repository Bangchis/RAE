from __future__ import annotations

import argparse

try:
    from .stage2_runtime import run_stage2_sampling
except ImportError:
    from stage2_runtime import run_stage2_sampling


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample Stage-2 images on JAX/NNX.")
    parser.add_argument("--config", required=True, help="Path to the repo YAML config.")
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint override.")
    parser.add_argument("--output", default="sample_jax.png", help="PNG grid output path.")
    parser.add_argument("--class-labels", default=None, help="Comma-separated class labels.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=None, help="Override sampler.params.num_steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override guidance.scale.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use EMA weights when available.")
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--wandb", action="store_true", help="Enable backend wandb hooks if needed.")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--backend-dir", default=None)
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="OmegaConf override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_stage2_sampling(args)


if __name__ == "__main__":
    main()
