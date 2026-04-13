from __future__ import annotations

import argparse
import json

try:
    from .stage1_jax_config import load_stage1_jax_runtime_config, runtime_config_to_dict
except ImportError:
    from stage1_jax_config import load_stage1_jax_runtime_config, runtime_config_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-1 all-JAX training entrypoint scaffold.")
    parser.add_argument("--config", required=True, help="Path to the Stage-1 repo YAML config.")
    parser.add_argument("--data-path", default=None, help="Optional TFDS/ImageFolder root override.")
    parser.add_argument("--data-format", choices=["imagefolder", "tfds"], default=None)
    parser.add_argument("--dataset-name", default=None, help="TFDS dataset name override, e.g. celebahq256.")
    parser.add_argument("--train-split", default=None, help="Optional TFDS train split override.")
    parser.add_argument("--eval-split", default=None, help="Optional TFDS eval split override.")
    parser.add_argument("--results-dir", default=None, help="Override experiment output directory.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="OmegaConf override.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved Stage-1 JAX runtime config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config wiring and exit successfully.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime_cfg, _repo_cfg, _cfg_path = load_stage1_jax_runtime_config(
        args.config,
        args.set_values,
        data_path=args.data_path,
        data_format=args.data_format,
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        eval_split=args.eval_split,
        results_dir=args.results_dir,
        precision=args.precision,
        seed=args.seed,
    )

    if args.print_config:
        print(json.dumps(runtime_config_to_dict(runtime_cfg), indent=2))

    if args.dry_run:
        print(
            "Stage-1 all-JAX dry-run succeeded. The runtime/config scaffold is wired, "
            "but the full generator/LPIPS/GAN trainer has not landed yet."
        )
        return

    raise NotImplementedError(
        "Stage-1 all-JAX trainer is not implemented yet in this branch. "
        "Use --dry-run to validate the Kaggle launch contract for now."
    )


if __name__ == "__main__":
    main()
