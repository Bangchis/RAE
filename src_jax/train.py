from __future__ import annotations

import argparse

try:
    from .stage2_runtime import run_stage2_training
except ImportError:
    from stage2_runtime import run_stage2_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage-2 RAE diffusion on JAX/NNX.")
    parser.add_argument("--config", required=True, help="Path to the repo YAML config.")
    parser.add_argument("--data-path", required=True, help="ImageNet root directory.")
    parser.add_argument("--results-dir", default="results_jax", help="Base directory for training runs.")
    parser.add_argument("--workdir", default=None, help="Explicit workdir. Overrides --results-dir.")
    parser.add_argument("--image-size", type=int, default=None, help="Override output image size.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed.")
    parser.add_argument("--num-train-samples", type=int, default=1_281_167, help="Dataset size used to derive total_steps when the config only provides epochs.")
    parser.add_argument("--exp-name", default=None, help="Override experiment name.")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb-entity", default=None, help="Override WANDB entity.")
    parser.add_argument("--wandb-project", default=None, help="Override WANDB project name.")
    parser.add_argument("--hf-repo-id", default=None, help="Upload the finished workdir to this Hugging Face repo.")
    parser.add_argument("--hf-private", action="store_true", help="Create the HF repo as private if needed.")
    parser.add_argument("--hf-revision", default="main", help="HF revision to upload to.")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable containing the HF token.")
    parser.add_argument("--hf-commit-message", default=None, help="Optional commit message for the HF upload.")
    parser.add_argument("--backend-dir", default=None, help="Override the cached diffuse_nnx checkout path.")
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="OmegaConf override, e.g. training.global_batch_size=256")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_stage2_training(args)


if __name__ == "__main__":
    main()
