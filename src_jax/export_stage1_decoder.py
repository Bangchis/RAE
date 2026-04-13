from __future__ import annotations

import argparse
from pathlib import Path
import sys

from flax import nnx
import torch


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .stage1_checkpoint import decoder_to_torch_state_dict, load_stage1_checkpoint, restore_decoder_state
    from .stage1_components import build_stage1_decoder, build_stage1_encoder, resolve_stage1_paths
    from .stage1_jax_config import load_stage1_jax_runtime_config
except ImportError:
    from stage1_checkpoint import decoder_to_torch_state_dict, load_stage1_checkpoint, restore_decoder_state
    from stage1_components import build_stage1_decoder, build_stage1_encoder, resolve_stage1_paths
    from stage1_jax_config import load_stage1_jax_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Stage-1 JAX decoder checkpoint to the PyTorch .pt format used by Stage-2.")
    parser.add_argument("--config", required=True, help="Stage-1 YAML config used for training.")
    parser.add_argument("--ckpt", required=True, help="Stage-1 JAX checkpoint path, e.g. checkpoints/ep-last.pt")
    parser.add_argument("--out", required=True, help="Output decoder .pt file.")
    parser.add_argument("--use-ema", action="store_true", help="Export the EMA decoder weights instead of the live model weights.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-format", choices=["imagefolder", "tfds"], default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backend-dir", default=None, help="Optional diffuse_nnx checkout override.")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--set", dest="set_values", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.backend_dir:
        os.environ["RAE_JAX_BACKEND_DIR"] = args.backend_dir
    runtime_cfg, _repo_cfg, cfg_path = load_stage1_jax_runtime_config(
        args.config,
        args.set_values,
        data_path=args.data_path,
        data_format=args.data_format,
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        eval_split=args.eval_split,
        precision=args.precision,
        seed=args.seed,
    )
    paths = resolve_stage1_paths(
        config_path=cfg_path,
        stage1_params=runtime_cfg.stage1_params,
        dino_disc_ckpt_path=runtime_cfg.discriminator.arch.dino_ckpt_path,
    )
    _encoder, encoder_spec = build_stage1_encoder(
        paths=paths,
        stage1_params=runtime_cfg.stage1_params,
        precision=runtime_cfg.training.precision,
    )
    decoder = build_stage1_decoder(
        paths=paths,
        encoder_spec=encoder_spec,
        stage1_params=runtime_cfg.stage1_params,
        precision=runtime_cfg.training.precision,
        seed=runtime_cfg.training.seed,
    )
    graph, state = nnx.split(decoder)
    checkpoint = load_stage1_checkpoint(args.ckpt)
    state = restore_decoder_state(state, checkpoint, use_ema=args.use_ema)
    decoder = nnx.merge(graph, state)

    decoder_state = decoder_to_torch_state_dict(decoder)
    if args.dtype != "fp32":
        target_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        decoder_state = {
            key: value.to(dtype=target_dtype) if torch.is_floating_point(value) else value
            for key, value in decoder_state.items()
        }

    output_path = Path(args.out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder_state, output_path)
    print(f"Saved decoder weights to {output_path}")


if __name__ == "__main__":
    main()
