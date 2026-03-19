from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .config_adapter import build_backend_config_dict, load_repo_config
    from .stage2_runtime import _to_config_dict
    from .vendor import activate_backend
except ImportError:
    from config_adapter import build_backend_config_dict, load_repo_config
    from stage2_runtime import _to_config_dict
    from vendor import activate_backend


def run_stage1_reconstruction(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser().resolve()
    repo_cfg, _ = load_repo_config(args.config, args.set_values)
    backend_cfg_dict = build_backend_config_dict(
        repo_cfg,
        config_path=config_path,
        mode="sample",
        image_size=args.image_size,
        precision=args.precision,
        seed=args.seed,
        exp_name=args.exp_name,
        enable_eval=False,
    )

    activate_backend(args.backend_dir)

    import jax.numpy as jnp
    from utils import initialize as init_utils
    from utils import wandb_utils as backend_wandb

    backend_wandb.initialize = lambda *_a, **_k: None
    backend_cfg = _to_config_dict(backend_cfg_dict)
    encoder = init_utils.instantiate_encoder(backend_cfg)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    arr = arr[None, ...]

    latents = encoder.encode(jnp.asarray(arr))
    recon = np.asarray(encoder.decode(latents)[0], dtype=np.uint8)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(recon).save(output_path)
    return output_path
