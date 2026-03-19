from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

try:
    from .config_adapter import build_backend_config_dict, cfg_to_dict, load_repo_config, write_json
    from .hf_utils import upload_path
    from .vendor import activate_backend
except ImportError:
    from config_adapter import build_backend_config_dict, cfg_to_dict, load_repo_config, write_json
    from hf_utils import upload_path
    from vendor import activate_backend


def _bridge_legacy_wandb_env(entity: str | None, project: str | None) -> None:
    if "WANDB_API_KEY" not in os.environ and "WANDB_KEY" in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]
    if entity and "WANDB_ENTITY" not in os.environ:
        os.environ["WANDB_ENTITY"] = entity
    elif "WANDB_ENTITY" not in os.environ and "ENTITY" in os.environ:
        os.environ["WANDB_ENTITY"] = os.environ["ENTITY"]
    if project and "PROJECT" not in os.environ:
        os.environ["PROJECT"] = project


def _disable_backend_wandb(wandb_utils: Any) -> None:
    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    wandb_utils.initialize = _noop
    wandb_utils.log = _noop
    wandb_utils.log_copy = _noop
    wandb_utils.log_images = _noop
    wandb_utils.log_line_plot = _noop


def _to_config_dict(payload: dict[str, Any]) -> Any:
    from ml_collections import ConfigDict

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return ConfigDict({k: convert(v) for k, v in value.items()})
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, tuple):
            return tuple(convert(v) for v in value)
        return value

    return convert(payload)


def _load_local_fid_utils() -> Any:
    module_path = Path(__file__).resolve().parents[1] / "src" / "utils" / "fid_utils.py"
    spec = importlib.util.spec_from_file_location("repo_fid_utils", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load fid_utils from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sanitize_torch_state(raw_state: Any, prefer_ema: bool = True) -> dict[str, Any]:
    if isinstance(raw_state, dict):
        if prefer_ema and isinstance(raw_state.get("ema"), dict):
            raw_state = raw_state["ema"]
        elif isinstance(raw_state.get("model"), dict):
            raw_state = raw_state["model"]
    sanitized: dict[str, Any] = {}
    if not isinstance(raw_state, dict):
        raise TypeError("Expected a torch checkpoint dictionary for stage-2 weight loading.")
    for key, value in raw_state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        sanitized[key] = value
    return sanitized


def _load_torch_weights_into_model(
    *,
    torch_ckpt: str,
    config: Any,
    model: Any,
    ema: Any,
    nnx: Any,
    port_module: Any,
    torch: Any,
) -> None:
    raw_state = torch.load(torch_ckpt, map_location="cpu")
    state_dict = _sanitize_torch_state(raw_state)
    num_encoder_blocks = int(config.network.get("num_encoder_blocks", config.network.get("depth", 0)))
    num_decoder_blocks = int(config.network.get("num_decoder_blocks", 0))
    depth = num_encoder_blocks + num_decoder_blocks
    nnx_state = port_module.convert_torch_to_flax(
        state_dict,
        depth=depth,
        encoder_depth=num_encoder_blocks,
    )
    target_model = model.network if hasattr(model, "network") else model
    target_ema = ema.ema.network if hasattr(ema.ema, "network") else ema.ema
    nnx.update(target_model, nnx_state)
    nnx.update(target_ema, nnx_state)


def _restore_orbax_checkpoint(
    *,
    ckpt_path: str,
    config: Any,
    optimizer: Any,
    ema: Any,
    nnx: Any,
    ckpt_utils: Any,
    step: int | None = None,
) -> tuple[Any, Any]:
    opt_graph, opt_rng_state, opt_state = nnx.split(optimizer, nnx.RngKey, ...)
    _, _, ema_state = nnx.split(ema, nnx.RngKey, ...)
    manager = ckpt_utils.build_checkpoint_manager(ckpt_path, **config.checkpoint.options)
    restore_step = manager.latest_step() if step is None else step
    if restore_step is None:
        raise FileNotFoundError(f"No Orbax checkpoint found under {ckpt_path}")
    loaded_state, _loaded_rng_state, loaded_ema_state = ckpt_utils.restore_checkpoints(
        ckpt_path,
        restore_step,
        opt_state,
        opt_rng_state,
        ema_state,
        mngr=manager,
    )
    nnx.update(optimizer, loaded_state)
    nnx.update(ema, loaded_ema_state)
    return optimizer.model, ema.ema


def _load_models_for_inference(
    args: argparse.Namespace,
    *,
    config_path: Path,
    mode: str,
):
    repo_cfg, _ = load_repo_config(str(config_path), args.set_values)
    backend_cfg_dict = build_backend_config_dict(
        repo_cfg,
        config_path=config_path,
        mode=mode,
        data_path=getattr(args, "data_path", None),
        image_size=getattr(args, "image_size", None),
        precision=getattr(args, "precision", "bf16"),
        seed=getattr(args, "seed", None),
        exp_name=getattr(args, "exp_name", None),
        wandb_project=getattr(args, "wandb_project", None),
        enable_eval=False,
    )

    activate_backend(getattr(args, "backend_dir", None))

    import jax
    import jax.numpy as jnp
    import torch
    from flax import nnx
    from utils import checkpoint as ckpt_utils
    from utils import initialize as init_utils
    from networks.transformers import port_torch_to_nnx as port_module
    from utils import wandb_utils as backend_wandb

    if getattr(args, "wandb", False):
        _bridge_legacy_wandb_env(getattr(args, "wandb_entity", None), getattr(args, "wandb_project", None))
    else:
        _disable_backend_wandb(backend_wandb)

    backend_cfg = _to_config_dict(backend_cfg_dict)
    encoder, model, optimizer, sampler, ema, _learning_rate_fn = init_utils.build_models(backend_cfg)

    stage2_cfg = cfg_to_dict(repo_cfg.get("stage_2"))
    ckpt_override = getattr(args, "ckpt", None)
    ckpt_path = ckpt_override or stage2_cfg.get("ckpt")
    if ckpt_path is None:
        raise ValueError("Stage-2 sampling requires a checkpoint. Set stage_2.ckpt or pass --ckpt.")

    if str(ckpt_path).endswith((".pt", ".pth", ".bin")):
        _load_torch_weights_into_model(
            torch_ckpt=str(ckpt_path),
            config=backend_cfg,
            model=model,
            ema=ema,
            nnx=nnx,
            port_module=port_module,
            torch=torch,
        )
        model_to_use = ema.ema if getattr(args, "use_ema", True) else model
    else:
        model, ema_model = _restore_orbax_checkpoint(
            ckpt_path=str(ckpt_path),
            config=backend_cfg,
            optimizer=optimizer,
            ema=ema,
            nnx=nnx,
            ckpt_utils=ckpt_utils,
            step=getattr(args, "step", None),
        )
        model_to_use = ema_model if getattr(args, "use_ema", True) else model

    guidance_cfg = cfg_to_dict(repo_cfg.get("guidance"))
    guidance_method = str(guidance_cfg.get("method", "cfg"))
    guidance_scale = float(getattr(args, "guidance_scale", guidance_cfg.get("scale", 1.0)))
    guidance_model = None

    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = cfg_to_dict(repo_cfg.get("guidance", {}).get("guidance_model"))
        if not guidance_model_cfg:
            raise ValueError("Autoguidance requires guidance.guidance_model in the config.")
        guidance_cfg_omega = repo_cfg.get("guidance").get("guidance_model")
        repo_cfg_clone = OmegaConf.create(OmegaConf.to_container(repo_cfg, resolve=False))
        repo_cfg_clone.stage_2 = guidance_cfg_omega
        repo_cfg_clone.guidance = {}
        guide_backend_cfg_dict = build_backend_config_dict(
            repo_cfg_clone,
            config_path=config_path,
            mode="sample",
            precision=getattr(args, "precision", "bf16"),
            seed=getattr(args, "seed", None),
            exp_name=f"{backend_cfg_dict['exp_name']}-guide",
            wandb_project=getattr(args, "wandb_project", None),
            enable_eval=False,
        )
        guide_backend_cfg = _to_config_dict(guide_backend_cfg_dict)
        _guide_encoder, guidance_model, guide_optimizer, _guide_sampler, guide_ema, _guide_lr = init_utils.build_models(guide_backend_cfg)
        guidance_ckpt = guidance_model_cfg.get("ckpt")
        if guidance_ckpt is None:
            raise ValueError("guidance.guidance_model.ckpt is required for autoguidance.")
        if str(guidance_ckpt).endswith((".pt", ".pth", ".bin")):
            _load_torch_weights_into_model(
                torch_ckpt=str(guidance_ckpt),
                config=guide_backend_cfg,
                model=guidance_model,
                ema=guide_ema,
                nnx=nnx,
                port_module=port_module,
                torch=torch,
            )
            guidance_model = guide_ema.ema if getattr(args, "use_ema", True) else guidance_model
        else:
            guidance_model, guidance_ema_model = _restore_orbax_checkpoint(
                ckpt_path=str(guidance_ckpt),
                config=guide_backend_cfg,
                optimizer=guide_optimizer,
                ema=guide_ema,
                nnx=nnx,
                ckpt_utils=ckpt_utils,
                step=getattr(args, "guidance_step", None),
            )
            guidance_model = guidance_ema_model if getattr(args, "use_ema", True) else guidance_model

    return {
        "repo_cfg": repo_cfg,
        "backend_cfg": backend_cfg,
        "encoder": encoder,
        "model": model_to_use,
        "sampler": sampler,
        "guidance_model": guidance_model,
        "guidance_method": guidance_method,
        "guidance_scale": guidance_scale,
        "jax": jax,
        "jnp": jnp,
        "nnx": nnx,
    }


def _make_grid(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"Expected 4D image array, got {images.shape}")
    n, h, w, c = images.shape
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    canvas = np.zeros((nrows * h, ncols * w, c), dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // ncols
        col = index % ncols
        canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    return canvas


def _save_png(path: str | Path, image: np.ndarray) -> None:
    from PIL import Image

    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path)


def _parse_class_labels(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return list(default)
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one class label is required.")
    return [int(label) for label in labels]


def _build_label_schedule(
    *,
    total_samples: int,
    num_classes: int,
    label_sampling: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if label_sampling == "random":
        return rng.integers(0, num_classes, size=total_samples, dtype=np.int32)
    if label_sampling != "equal":
        raise ValueError(f"Unsupported label sampling mode: {label_sampling}")
    labels = np.arange(num_classes, dtype=np.int32)
    repeats = int(math.ceil(total_samples / num_classes))
    tiled = np.tile(labels, repeats)[:total_samples]
    return tiled


def run_stage2_training(args: argparse.Namespace) -> Path:
    if not args.data_path:
        raise ValueError("--data-path is required for Stage-2 JAX training.")

    repo_cfg, config_path = load_repo_config(args.config, args.set_values)
    backend_cfg_dict = build_backend_config_dict(
        repo_cfg,
        config_path=config_path,
        mode="train",
        data_path=args.data_path,
        image_size=args.image_size,
        precision=args.precision,
        seed=args.global_seed,
        num_train_samples=args.num_train_samples,
        exp_name=args.exp_name,
        wandb_project=args.wandb_project,
        enable_eval=True,
    )

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else Path(args.results_dir).expanduser().resolve() / backend_cfg_dict["exp_name"]
    workdir.mkdir(parents=True, exist_ok=True)
    write_json(workdir / "jax_adapter_config.json", backend_cfg_dict)

    activate_backend(args.backend_dir)

    import torch
    from flax import nnx
    from networks.transformers import port_torch_to_nnx as port_module
    from trainers import dit_imagenet as trainer
    from utils import initialize as init_utils
    from utils import wandb_utils as backend_wandb

    if args.wandb:
        _bridge_legacy_wandb_env(args.wandb_entity, args.wandb_project)
    else:
        _disable_backend_wandb(backend_wandb)

    backend_cfg = _to_config_dict(backend_cfg_dict)

    original_build_models = init_utils.build_models

    if backend_cfg_dict.get("torch_ckpt"):
        def patched_build_models(config: Any):
            encoder, model, optimizer, sampler, ema, learning_rate_fn = original_build_models(config)
            _load_torch_weights_into_model(
                torch_ckpt=backend_cfg_dict["torch_ckpt"],
                config=config,
                model=model,
                ema=ema,
                nnx=nnx,
                port_module=port_module,
                torch=torch,
            )
            return encoder, model, optimizer, sampler, ema, learning_rate_fn

        init_utils.build_models = patched_build_models

    try:
        trainer.train_and_evaluate(backend_cfg, str(workdir))
    finally:
        init_utils.build_models = original_build_models

    if args.hf_repo_id:
        upload_path(
            str(workdir),
            args.hf_repo_id,
            private=args.hf_private,
            token_env=args.hf_token_env,
            revision=args.hf_revision,
            commit_message=args.hf_commit_message or f"upload training run {backend_cfg_dict['exp_name']}",
        )

    return workdir


def run_stage2_sampling(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser().resolve()
    runtime = _load_models_for_inference(args, config_path=config_path, mode="sample")
    repo_cfg = runtime["repo_cfg"]
    backend_cfg = runtime["backend_cfg"]
    encoder = runtime["encoder"]
    model = runtime["model"]
    sampler = runtime["sampler"]
    guidance_model = runtime["guidance_model"]
    guidance_method = runtime["guidance_method"]
    guidance_scale = runtime["guidance_scale"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    nnx = runtime["nnx"]

    misc_cfg = cfg_to_dict(repo_cfg.get("misc"))
    labels = _parse_class_labels(args.class_labels, default=[207, 360])
    rngs = nnx.Rngs((args.seed or 0) + jax.process_index())
    input_size = int(backend_cfg.network.input_size)
    in_channels = int(backend_cfg.network.in_channels)
    noise = jax.random.normal(rngs(), (len(labels), input_size, input_size, in_channels), dtype=jnp.float32)
    label_arr = jnp.asarray(labels, dtype=jnp.int32)

    g_net = None
    if guidance_scale > 1.0:
        g_net = model if guidance_method == "cfg" else guidance_model

    samples = sampler.sample(
        rngs,
        model,
        noise,
        y=label_arr,
        g_net=g_net,
        guidance_scale=guidance_scale,
        num_sampling_steps=args.num_steps,
    )
    decoded = encoder.decode(samples)
    gathered = jax.experimental.multihost_utils.process_allgather(decoded, tiled=True)

    output_path = Path(args.output).expanduser().resolve()
    if jax.process_index() == 0:
        _save_png(output_path, _make_grid(np.asarray(gathered, dtype=np.uint8)))
    return output_path


def run_stage2_sampling_ddp(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser().resolve()
    runtime = _load_models_for_inference(args, config_path=config_path, mode="sample")
    repo_cfg = runtime["repo_cfg"]
    backend_cfg = runtime["backend_cfg"]
    encoder = runtime["encoder"]
    model = runtime["model"]
    sampler = runtime["sampler"]
    guidance_model = runtime["guidance_model"]
    guidance_method = runtime["guidance_method"]
    guidance_scale = runtime["guidance_scale"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    nnx = runtime["nnx"]

    sample_dir = Path(args.sample_dir).expanduser().resolve()
    sample_dir.mkdir(parents=True, exist_ok=True)

    num_classes = int(cfg_to_dict(repo_cfg.get("misc")).get("num_classes", 1000))
    all_labels = _build_label_schedule(
        total_samples=args.num_samples,
        num_classes=num_classes,
        label_sampling=args.label_sampling,
        seed=args.seed or 0,
    )

    process_count = max(1, jax.process_count())
    process_index = jax.process_index()
    per_process_total = int(math.ceil(args.num_samples / process_count))
    start = process_index * per_process_total
    stop = min(args.num_samples, start + per_process_total)
    local_labels = all_labels[start:stop]

    input_size = int(backend_cfg.network.input_size)
    in_channels = int(backend_cfg.network.in_channels)
    rngs = nnx.Rngs((args.seed or 0) + process_index)
    g_net = None
    if guidance_scale > 1.0:
        g_net = model if guidance_method == "cfg" else guidance_model

    local_images: list[np.ndarray] = []
    for offset in range(0, len(local_labels), args.per_proc_batch_size):
        batch_labels = local_labels[offset : offset + args.per_proc_batch_size]
        if len(batch_labels) == 0:
            continue
        noise = jax.random.normal(rngs(), (len(batch_labels), input_size, input_size, in_channels), dtype=jnp.float32)
        label_arr = jnp.asarray(batch_labels, dtype=jnp.int32)
        latents = sampler.sample(
            rngs,
            model,
            noise,
            y=label_arr,
            g_net=g_net,
            guidance_scale=guidance_scale,
            num_sampling_steps=args.num_steps,
        )
        decoded = np.asarray(encoder.decode(latents), dtype=np.uint8)
        for local_index, image in enumerate(decoded):
            global_index = start + offset + local_index
            if global_index >= args.num_samples:
                break
            _save_png(sample_dir / f"{global_index:06d}.png", image)
        if args.save_npz:
            local_images.append(decoded)

    jax.experimental.multihost_utils.sync_global_devices("sample_ddp_complete")

    if args.save_npz and local_images:
        shard_path = sample_dir / f"samples_rank{process_index:03d}.npz"
        np.savez_compressed(shard_path, images=np.concatenate(local_images, axis=0))

    if args.fid_ref and process_index == 0:
        fid_utils = _load_local_fid_utils()
        mu_gen, sigma_gen, _num_samples = fid_utils.calculate_stats_from_image_folder(
            sample_dir,
            batch_size=args.fid_batch_size,
            device=args.fid_device,
            num_threads=args.fid_num_threads,
        )
        mu_ref, sigma_ref = fid_utils.load_reference_stats(args.fid_ref)
        fid_value = fid_utils._fid_from_moments(mu_gen, sigma_gen, mu_ref, sigma_ref)
        print(f"FID={fid_value:.6f}")

    return sample_dir
