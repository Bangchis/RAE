from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf


IMAGENET1K_TRAIN_SAMPLES = 1_281_167
FID_CACHE_DIR = Path.home() / ".cache" / "rae_jax" / "fid_refs"


def load_repo_config(config_path: str, overrides: list[str] | None = None) -> tuple[DictConfig, Path]:
    cfg_path = Path(config_path).expanduser().resolve()
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg, cfg_path


def cfg_to_dict(section: Any) -> dict[str, Any]:
    if section is None:
        return {}
    container = OmegaConf.to_container(section, resolve=True)
    if container is None:
        return {}
    if not isinstance(container, dict):
        raise TypeError(f"Expected mapping section, got {type(container)!r}")
    return dict(container)


def resolve_repo_value(raw: Any, *, config_path: Path) -> Any:
    if raw is None or not isinstance(raw, str):
        return raw
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return str(cwd_candidate.resolve())
    cfg_candidate = config_path.parent / raw
    if cfg_candidate.exists():
        return str(cfg_candidate.resolve())
    return raw


def parse_guidance_value(cfg: dict[str, Any], key: str, default: float) -> float:
    if key in cfg:
        return float(cfg[key])
    dashed = key.replace("_", "-")
    if dashed in cfg:
        return float(cfg[dashed])
    return float(default)


def infer_network_class(stage2_target: str) -> str:
    target = stage2_target.lower()
    if "ddt" in target:
        return "lightning_ddt"
    if "lightningdit" in target or "lightning_dit" in target:
        return "lightning_dit"
    if "dit" in target:
        return "dit"
    raise ValueError(f"Unsupported Stage-2 target for JAX adapter: {stage2_target}")


def _derive_image_size(stage1_params: dict[str, Any], misc_cfg: dict[str, Any], fallback: int | None) -> int:
    if fallback is not None:
        return int(fallback)
    latent_size = misc_cfg.get("latent_size")
    decoder_patch = int(stage1_params.get("decoder_patch_size", 16))
    if latent_size and len(latent_size) >= 3:
        return int(latent_size[1]) * decoder_patch
    return int(stage1_params.get("encoder_input_size", 256))


def _convert_schedule_type(schedule_type: str | None) -> str:
    if not schedule_type:
        return "constant"
    normalized = str(schedule_type).strip().lower()
    if normalized in {"constant", "linear", "cosine", "polynomial"}:
        return normalized
    if normalized == "linear-constant":
        return normalized
    return "constant"


def maybe_convert_fid_reference(reference_path: str | None) -> str | None:
    if not reference_path:
        return None
    ref_path = Path(reference_path).expanduser().resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"FID reference not found: {ref_path}")
    if ref_path.suffix in {".pkl", ".pickle"}:
        return str(ref_path)
    if ref_path.suffix != ".npz":
        raise ValueError(f"Unsupported FID reference format: {ref_path.suffix}")

    with np.load(ref_path) as archive:
        if "mu" not in archive.files or "sigma" not in archive.files:
            raise KeyError(f"{ref_path} must contain 'mu' and 'sigma'. Keys={archive.files}")
        payload = {
            "fid": {
                "mu": archive["mu"],
                "sigma": archive["sigma"],
            }
        }

    digest = hashlib.sha256(str(ref_path).encode("utf-8")).hexdigest()[:16]
    FID_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FID_CACHE_DIR / f"{ref_path.stem}-{digest}.pkl"
    if not out_path.exists():
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle)
    return str(out_path)


def make_experiment_name(config_path: Path, network_class: str, precision: str, suffix: str) -> str:
    stem = config_path.stem
    return f"{stem}-{network_class}-{precision}-{suffix}"


def build_backend_config_dict(
    repo_cfg: DictConfig,
    *,
    config_path: Path,
    mode: str,
    data_path: str | None = None,
    image_size: int | None = None,
    precision: str = "bf16",
    seed: int | None = None,
    num_train_samples: int = IMAGENET1K_TRAIN_SAMPLES,
    exp_name: str | None = None,
    wandb_project: str | None = None,
    enable_eval: bool = True,
    require_stage2: bool = True,
) -> dict[str, Any]:
    stage1_cfg = cfg_to_dict(repo_cfg.get("stage_1"))
    stage2_cfg = cfg_to_dict(repo_cfg.get("stage_2"))
    transport_cfg = cfg_to_dict(repo_cfg.get("transport"))
    sampler_cfg = cfg_to_dict(repo_cfg.get("sampler"))
    guidance_cfg = cfg_to_dict(repo_cfg.get("guidance"))
    misc_cfg = cfg_to_dict(repo_cfg.get("misc"))
    training_cfg = cfg_to_dict(repo_cfg.get("training"))
    eval_cfg = cfg_to_dict(repo_cfg.get("eval"))

    stage1_params = dict(stage1_cfg.get("params", {}))
    stage2_params = dict(stage2_cfg.get("params", {}))
    transport_params = dict(transport_cfg.get("params", {}))
    sampler_params = dict(sampler_cfg.get("params", {}))

    stage2_target = str(stage2_cfg.get("target", "")).strip()
    if stage2_target:
        network_class = infer_network_class(stage2_target)
    elif require_stage2:
        raise ValueError("stage_2.target is required for this JAX adapter flow.")
    else:
        network_class = "stage1_only"
    resolved_image_size = _derive_image_size(stage1_params, misc_cfg, image_size)
    latent_size = misc_cfg.get("latent_size", [stage2_params.get("in_channels", 768), stage2_params.get("input_size", 16), stage2_params.get("input_size", 16)])
    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    cfg_seed = seed if seed is not None else int(training_cfg.get("global_seed", 0))

    stage1_encoder_model = resolve_repo_value(
        stage1_params.get("encoder_params", {}).get("dinov2_path") or stage1_params.get("encoder_config_path"),
        config_path=config_path,
    )
    decoder_ckpt = resolve_repo_value(stage1_params.get("pretrained_decoder_path"), config_path=config_path)
    stats_path = resolve_repo_value(stage1_params.get("normalization_stat_path"), config_path=config_path)

    batch_size = int(training_cfg.get("global_batch_size", 1024))
    epochs = int(training_cfg.get("epochs", 1))
    steps_per_epoch = max(1, math.ceil(num_train_samples / max(batch_size, 1)))
    total_steps = int(training_cfg.get("total_steps", epochs * steps_per_epoch))

    backend_cfg: dict[str, Any] = {
        "trainer": "DiT_ImageNet",
        "exp_name": exp_name or make_experiment_name(config_path, network_class, precision, mode),
        "project_name": wandb_project or "RAE-JAX",
        "seed": cfg_seed,
        "dtype": "bfloat16" if precision == "bf16" else "float32",
        "standalone_eval": mode != "train",
        "total_steps": total_steps,
        "log_every_steps": int(training_cfg.get("log_every", 100)),
        "save_every_steps": int(training_cfg.get("ckpt_every", 5_000)),
        "visualize_every_steps": int(training_cfg.get("sample_every", 10_000)),
        "data": {
            "data_dir": resolve_repo_value(data_path or eval_cfg.get("data_path"), config_path=config_path),
            "stat_dir": maybe_convert_fid_reference(resolve_repo_value(eval_cfg.get("fid_ref"), config_path=config_path)),
            "batch_size": batch_size,
            "image_size": resolved_image_size,
            "latent_dataset": False,
            "num_train_samples": num_train_samples,
            "num_workers": int(training_cfg.get("num_workers", 4)),
            "seed": cfg_seed,
            "seed_pt": cfg_seed,
        },
        "encoder_class": "RAE",
        "encoder": {
            "pretrained_path": decoder_ckpt,
            "stats_path": stats_path,
            "resolution": int(stage1_params.get("encoder_input_size", 224)),
            "encoded_pixels": False,
            "pretrained_model_name_or_path": stage1_encoder_model,
        },
        "network_class": network_class,
        "network": {
            "input_size": int(stage2_params.get("input_size", latent_size[1])),
            "patch_size": int(stage2_params.get("patch_size", 1)),
            "in_channels": int(stage2_params.get("in_channels", latent_size[0])),
            "continuous_time_embed": True,
            "freq_embed_size": 512,
            "num_classes": int(misc_cfg.get("num_classes", stage2_params.get("num_classes", 1000))),
            "class_dropout_prob": float(stage2_params.get("class_dropout_prob", 0.1)),
            "enable_dropout": True,
            "qk_norm": bool(stage2_params.get("use_qknorm", False)),
            "use_rope": bool(stage2_params.get("use_rope", True)),
            "swiglu": bool(stage2_params.get("use_swiglu", True)),
            "rms_norm": bool(stage2_params.get("use_rmsnorm", True)),
            "adaln_shift": not bool(stage2_params.get("wo_shift", False)),
            "mlp_ratio": float(stage2_params.get("mlp_ratio", 4.0)),
            "mlp_dropout": float(stage2_params.get("mlp_dropout", 0.0)),
            "attn_w_dropout": float(stage2_params.get("attn_w_dropout", 0.0)),
            "attn_o_dropout": float(stage2_params.get("attn_o_dropout", 0.0)),
        },
        "interface_class": "sit",
        "interface": {
            "train_time_dist_type": str(transport_params.get("time_dist_type", "uniform")),
            "t_mu": 0.0,
            "t_sigma": 1.0,
            "n_mu": 0.0,
            "n_sigma": 1.0,
            "x_sigma": 0.5,
            "t_shift_base": int(misc_cfg.get("time_dist_shift_base", 4096)),
        },
        "optimizer_class": "adamw",
        "optimizer": {
            "b1": float(training_cfg.get("beta", [0.9, 0.95])[0]),
            "b2": float(training_cfg.get("beta", [0.9, 0.95])[1]),
            "eps": 1e-8,
            "weight_decay": float(training_cfg.get("wd", 0.0)),
        },
        "learning_rate": float(training_cfg.get("base_lr", 2e-4)),
        "learning_rate_schedule": _convert_schedule_type(training_cfg.get("schedule_type")),
        "warmup_steps": int(training_cfg.get("warmup_steps", 0)),
        "min_abs_lr": float(training_cfg.get("final_lr", 0.0)),
        "sampler_class": str(sampler_params.get("sampling_method", "euler")).replace("_", "-").lower(),
        "sampler": {
            "num_sampling_steps": int(sampler_params.get("num_steps", sampler_params.get("steps", 50))),
            "sampling_time_dist": str(transport_params.get("time_dist_type", "uniform")).lower(),
            "sampling_time_kwargs": {
                "t_start": 1.0,
                "t_end": 0.0,
                "t_shift_base": int(misc_cfg.get("time_dist_shift_base", 4096)),
                "t_shift_cur": int(misc_cfg.get("time_dist_shift_dim", math.prod(latent_size))),
            },
        },
        "ema_class": "ema",
        "ema": {
            "decay": float(training_cfg.get("ema_decay", 0.9995)),
        },
        "checkpoint": {
            "options": {
                "save_interval_steps": int(training_cfg.get("ckpt_every", 5_000)),
                "max_to_keep": 8,
                "keep_period": int(training_cfg.get("ckpt_every", 5_000)) * 2,
                "enable_async_checkpointing": False,
            }
        },
        "visualize": {
            "on": True,
            "num_samples": 64,
            "guidance_scale": guidance_scale,
            "visualize_reconstruction": False,
        },
        "eval": {
            "on": False,
            "on_load": False,
            "seed": 42,
            "detector": "inception",
            "batch_size": int(eval_cfg.get("fid_per_proc_batch_size", 4)),
            "inception_batch_size": int(eval_cfg.get("fid_batch_size", 64)),
            "save_samples_path": "",
            "all_guidance_scales": (guidance_scale,),
            "all_eval_samples_nums": ((int(eval_cfg.get("fid_num_samples", 0)),),),
            "eval_every_steps": ((int(eval_cfg.get("fid_every", eval_cfg.get("eval_every", 0))),),),
        },
        "sharding": {
            "mesh": [("data", -1)],
            "data_axis": "data",
            "strategy": [(".*", "replicate")],
            "rules": [("act_batch", "data")],
            "allow_split_physical_axes": False,
        },
        "repo_config_path": str(config_path),
        "torch_ckpt": None,
        "guidance_method": str(guidance_cfg.get("method", "cfg")),
        "guidance_scale": guidance_scale,
    }

    if network_class == "lightning_ddt":
        hidden_size = stage2_params.get("hidden_size", [1152, 2048])
        depth = stage2_params.get("depth", [28, 2])
        num_heads = stage2_params.get("num_heads", [16, 16])
        backend_cfg["network"].update(
            {
                "num_encoder_blocks": int(depth[0]),
                "num_decoder_blocks": int(depth[1]),
                "encoder_hidden_size": int(hidden_size[0]),
                "encoder_num_heads": int(num_heads[0]),
                "decoder_hidden_size": int(hidden_size[1]),
                "decoder_num_heads": int(num_heads[1]),
            }
        )
    else:
        backend_cfg["network"].update(
            {
                "hidden_size": int(stage2_params.get("hidden_size", 1152)),
                "depth": int(stage2_params.get("depth", 28)),
                "num_heads": int(stage2_params.get("num_heads", 16)),
            }
        )

    stage2_ckpt = resolve_repo_value(stage2_cfg.get("ckpt"), config_path=config_path)
    if stage2_ckpt:
        if str(stage2_ckpt).endswith((".pt", ".pth", ".bin")):
            backend_cfg["torch_ckpt"] = str(stage2_ckpt)
        else:
            backend_cfg["pretrained_ckpt"] = str(stage2_ckpt)

    if enable_eval and eval_cfg.get("fid_ref") and int(eval_cfg.get("fid_num_samples", 0)) > 0:
        backend_cfg["eval"]["on"] = True

    if mode != "train":
        backend_cfg["eval"]["on"] = False
        backend_cfg["visualize"]["on"] = False

    return backend_cfg


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
