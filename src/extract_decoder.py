#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from utils.model_utils import instantiate_from_config


def _strip_prefix(key: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key_out = key
        changed = True
        while changed:
            previous = key_out
            key_out = _strip_prefix(key_out, ("module.",))
            key_out = _strip_prefix(key_out, ("_orig_mod.",))
            changed = key_out != previous
        normalized[key_out] = value
    return normalized


def _load_checkpoint(path: str) -> Any:
    return torch.load(path, map_location="cpu")


def _select_model_state(ckpt_obj: Any, use_ema: bool) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and ("model" in ckpt_obj or "ema" in ckpt_obj):
        key = "ema" if use_ema else "model"
        if key not in ckpt_obj:
            raise KeyError(f"Checkpoint does not contain {key!r}.")
        state_dict = ckpt_obj[key]
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint entry {key!r} is not a state_dict.")
        return state_dict
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise TypeError(f"Unrecognized checkpoint format: {type(ckpt_obj)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract decoder weights from a Stage-1 checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="YAML config used to instantiate RAE.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional full Stage-1 checkpoint.")
    parser.add_argument("--use-ema", action="store_true", help="Load the EMA model from the checkpoint.")
    parser.add_argument("--out", type=str, required=True, help="Output decoder checkpoint path.")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32", help="Optional output dtype cast.")
    args = parser.parse_args()

    full_cfg = OmegaConf.load(args.config)
    rae_cfg = full_cfg.get("stage_1")
    if rae_cfg is None:
        raise KeyError("Config must contain a stage_1 section.")

    rae = instantiate_from_config(rae_cfg)
    if args.ckpt is not None:
        checkpoint = _load_checkpoint(args.ckpt)
        state_dict = _normalize_state_dict_keys(_select_model_state(checkpoint, use_ema=args.use_ema))
        rae.load_state_dict(state_dict, strict=False)

    decoder_state = rae.decoder.state_dict()
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
