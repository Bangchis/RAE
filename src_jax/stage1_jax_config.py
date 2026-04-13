from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_adapter import (
    _derive_image_size,
    cfg_to_dict,
    infer_data_source,
    infer_default_training_random_flip,
    load_repo_config,
    normalize_data_source,
    resolve_repo_value,
)


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    lr: float
    beta1: float
    beta2: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    type: str
    base_lr: float
    final_lr: float
    warmup_epochs: int
    decay_end_epoch: int
    warmup_from_zero: bool


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int
    global_batch_size: int
    ema_decay: float
    clip_grad: float
    log_every: int
    checkpoint_interval: int
    sample_every: int
    precision: str
    seed: int
    optimizer: OptimizerConfig
    scheduler: ScheduleConfig


@dataclass(frozen=True, slots=True)
class DataConfig:
    source: str
    data_dir: str | None
    dataset_name: str | None
    train_split: str
    eval_split: str | None
    image_size: int
    shuffle_buffer: int
    random_flip: bool


@dataclass(frozen=True, slots=True)
class DiscriminatorArchConfig:
    dino_ckpt_path: str | None
    ks: int
    norm_type: str
    using_spec_norm: bool
    recipe: str


@dataclass(frozen=True, slots=True)
class DiscriminatorConfig:
    arch: DiscriminatorArchConfig
    optimizer: OptimizerConfig
    scheduler: ScheduleConfig
    augment_prob: float
    augment_cutout: float


@dataclass(frozen=True, slots=True)
class GanLossConfig:
    disc_loss: str
    gen_loss: str
    disc_weight: float
    perceptual_weight: float
    disc_start_epoch: int
    disc_update_start_epoch: int
    lpips_start_epoch: int
    max_d_weight: float
    disc_updates: int


@dataclass(frozen=True, slots=True)
class EvalConfig:
    enabled: bool
    eval_every: int
    eval_model: bool
    metrics: tuple[str, ...]
    reference_npz_path: str | None


@dataclass(frozen=True, slots=True)
class Stage1JaxRuntimeConfig:
    config_path: Path
    experiment_name: str
    results_dir: Path
    stage1_target: str
    stage1_params: dict[str, Any]
    data: DataConfig
    training: TrainingConfig
    discriminator: DiscriminatorConfig
    gan_loss: GanLossConfig
    evaluation: EvalConfig


def _build_optimizer_cfg(section: dict[str, Any], *, default_lr: float = 2e-4) -> OptimizerConfig:
    betas = section.get("betas", [0.9, 0.95])
    return OptimizerConfig(
        lr=float(section.get("lr", default_lr)),
        beta1=float(betas[0]),
        beta2=float(betas[1]),
        weight_decay=float(section.get("weight_decay", 0.0)),
    )


def _build_schedule_cfg(section: dict[str, Any], optimizer: OptimizerConfig, *, default_epochs: int) -> ScheduleConfig:
    schedule_type = str(section.get("type", "constant")).strip().lower()
    return ScheduleConfig(
        type=schedule_type,
        base_lr=float(section.get("base_lr", optimizer.lr)),
        final_lr=float(section.get("final_lr", 0.0 if schedule_type == "constant" else optimizer.lr)),
        warmup_epochs=int(section.get("warmup_epochs", 0)),
        decay_end_epoch=int(section.get("decay_end_epoch", default_epochs)),
        warmup_from_zero=bool(section.get("warmup_from_zero", False)),
    )


def _build_eval_cfg(section: dict[str, Any]) -> EvalConfig:
    eval_every = int(section.get("eval_interval", section.get("eval_every", 0)))
    reference_npz_path = section.get("reference_npz_path") or section.get("fid_ref")
    metrics = tuple(str(metric) for metric in section.get("metrics", ()))
    return EvalConfig(
        enabled=eval_every > 0 or bool(reference_npz_path),
        eval_every=eval_every,
        eval_model=bool(section.get("eval_model", False)),
        metrics=metrics,
        reference_npz_path=str(reference_npz_path) if reference_npz_path else None,
    )


def _validate_runtime_config(config: Stage1JaxRuntimeConfig) -> None:
    if config.data.source == "tfds" and not config.data.dataset_name:
        raise ValueError("Stage-1 JAX TFDS mode requires data.dataset_name or --dataset-name.")
    if config.data.source == "imagefolder" and not config.data.data_dir:
        raise ValueError("Stage-1 JAX imagefolder mode requires a data path.")
    if config.training.global_batch_size <= 0:
        raise ValueError("training.global_batch_size must be positive.")
    if config.training.epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    if config.gan_loss.disc_weight > 0.0 and not config.discriminator.arch.dino_ckpt_path:
        raise ValueError("GAN training requires gan.disc.arch.dino_ckpt_path.")


def load_stage1_jax_runtime_config(
    config_path: str,
    overrides: list[str] | None = None,
    *,
    data_path: str | None = None,
    data_format: str | None = None,
    dataset_name: str | None = None,
    train_split: str | None = None,
    eval_split: str | None = None,
    results_dir: str | None = None,
    precision: str = "bf16",
    seed: int | None = None,
) -> tuple[Stage1JaxRuntimeConfig, Any, Path]:
    repo_cfg, cfg_path = load_repo_config(config_path, overrides)

    data_cfg = cfg_to_dict(repo_cfg.get("data"))
    stage1_cfg = cfg_to_dict(repo_cfg.get("stage_1"))
    training_cfg = cfg_to_dict(repo_cfg.get("training"))
    gan_cfg = cfg_to_dict(repo_cfg.get("gan"))
    eval_cfg = cfg_to_dict(repo_cfg.get("eval"))
    misc_cfg = cfg_to_dict(repo_cfg.get("misc"))

    stage1_params = dict(stage1_cfg.get("params", {}))
    image_size = _derive_image_size(stage1_params, misc_cfg, None)
    resolved_data_dir = resolve_repo_value(data_path or data_cfg.get("data_dir") or data_cfg.get("data_path"), config_path=cfg_path)
    requested_source = normalize_data_source(data_format or data_cfg.get("format") or data_cfg.get("source"))
    resolved_dataset_name = dataset_name or data_cfg.get("dataset_name")
    data_source = infer_data_source(
        requested_source=requested_source,
        resolved_data_dir=resolved_data_dir,
        dataset_name=resolved_dataset_name,
    )

    random_flip = bool(
        training_cfg.get(
            "random_flip",
            infer_default_training_random_flip(config_path=cfg_path, data_path=resolved_dataset_name or resolved_data_dir),
        )
    )
    resolved_seed = int(training_cfg.get("global_seed", 0) if seed is None else seed)
    optimizer_cfg = _build_optimizer_cfg(dict(training_cfg.get("optimizer", {})))
    scheduler_cfg = _build_schedule_cfg(
        dict(training_cfg.get("scheduler", {})),
        optimizer_cfg,
        default_epochs=int(training_cfg.get("epochs", 1)),
    )

    disc_cfg = dict(gan_cfg.get("disc", {}))
    disc_optimizer_cfg = _build_optimizer_cfg(dict(disc_cfg.get("optimizer", {})), default_lr=optimizer_cfg.lr)
    disc_scheduler_cfg = _build_schedule_cfg(
        dict(disc_cfg.get("scheduler", {})),
        disc_optimizer_cfg,
        default_epochs=int(training_cfg.get("epochs", 1)),
    )
    disc_arch_cfg = dict(disc_cfg.get("arch", {}))
    disc_aug_cfg = dict(disc_cfg.get("augment", {}))
    loss_cfg = dict(gan_cfg.get("loss", {}))

    runtime_cfg = Stage1JaxRuntimeConfig(
        config_path=cfg_path,
        experiment_name=f"{cfg_path.stem}-stage1-jax",
        results_dir=Path(results_dir).expanduser().resolve() if results_dir else (Path("results") / "stage1_jax" / cfg_path.stem).resolve(),
        stage1_target=str(stage1_cfg.get("target", "stage1.RAE")),
        stage1_params=stage1_params,
        data=DataConfig(
            source=data_source,
            data_dir=str(resolved_data_dir) if resolved_data_dir is not None else None,
            dataset_name=str(resolved_dataset_name) if resolved_dataset_name is not None else None,
            train_split=str(train_split or data_cfg.get("train_split") or "train"),
            eval_split=str(eval_split or data_cfg.get("eval_split")) if (eval_split or data_cfg.get("eval_split")) is not None else None,
            image_size=int(image_size),
            shuffle_buffer=int(data_cfg.get("shuffle_buffer", 20_000)),
            random_flip=random_flip,
        ),
        training=TrainingConfig(
            epochs=int(training_cfg.get("epochs", 1)),
            global_batch_size=int(training_cfg.get("global_batch_size", 1)),
            ema_decay=float(training_cfg.get("ema_decay", 0.9999)),
            clip_grad=float(training_cfg.get("clip_grad", 0.0)),
            log_every=int(training_cfg.get("log_interval", training_cfg.get("log_every", 100))),
            checkpoint_interval=int(training_cfg.get("checkpoint_interval", 1)),
            sample_every=int(training_cfg.get("sample_every", 10_000)),
            precision=str(precision),
            seed=resolved_seed,
            optimizer=optimizer_cfg,
            scheduler=scheduler_cfg,
        ),
        discriminator=DiscriminatorConfig(
            arch=DiscriminatorArchConfig(
                dino_ckpt_path=str(resolve_repo_value(disc_arch_cfg.get("dino_ckpt_path"), config_path=cfg_path)) if disc_arch_cfg.get("dino_ckpt_path") else None,
                ks=int(disc_arch_cfg.get("ks", 9)),
                norm_type=str(disc_arch_cfg.get("norm_type", "bn")),
                using_spec_norm=bool(disc_arch_cfg.get("using_spec_norm", True)),
                recipe=str(disc_arch_cfg.get("recipe", "S_8")),
            ),
            optimizer=disc_optimizer_cfg,
            scheduler=disc_scheduler_cfg,
            augment_prob=float(disc_aug_cfg.get("prob", 1.0)),
            augment_cutout=float(disc_aug_cfg.get("cutout", 0.0)),
        ),
        gan_loss=GanLossConfig(
            disc_loss=str(loss_cfg.get("disc_loss", "hinge")),
            gen_loss=str(loss_cfg.get("gen_loss", "vanilla")),
            disc_weight=float(loss_cfg.get("disc_weight", 0.0)),
            perceptual_weight=float(loss_cfg.get("perceptual_weight", 1.0)),
            disc_start_epoch=int(loss_cfg.get("disc_start", 0)),
            disc_update_start_epoch=int(loss_cfg.get("disc_upd_start", 0)),
            lpips_start_epoch=int(loss_cfg.get("lpips_start", 0)),
            max_d_weight=float(loss_cfg.get("max_d_weight", 1e4)),
            disc_updates=int(loss_cfg.get("disc_updates", 1)),
        ),
        evaluation=_build_eval_cfg(eval_cfg),
    )
    _validate_runtime_config(runtime_cfg)
    return runtime_cfg, repo_cfg, cfg_path


def runtime_config_to_dict(config: Stage1JaxRuntimeConfig) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "results_dir": str(config.results_dir),
        "stage1_target": config.stage1_target,
        "data": {
            "source": config.data.source,
            "data_dir": config.data.data_dir,
            "dataset_name": config.data.dataset_name,
            "train_split": config.data.train_split,
            "eval_split": config.data.eval_split,
            "image_size": config.data.image_size,
            "shuffle_buffer": config.data.shuffle_buffer,
            "random_flip": config.data.random_flip,
        },
        "training": {
            "epochs": config.training.epochs,
            "global_batch_size": config.training.global_batch_size,
            "ema_decay": config.training.ema_decay,
            "clip_grad": config.training.clip_grad,
            "log_every": config.training.log_every,
            "checkpoint_interval": config.training.checkpoint_interval,
            "sample_every": config.training.sample_every,
            "precision": config.training.precision,
            "seed": config.training.seed,
            "optimizer": {
                "lr": config.training.optimizer.lr,
                "beta1": config.training.optimizer.beta1,
                "beta2": config.training.optimizer.beta2,
                "weight_decay": config.training.optimizer.weight_decay,
            },
            "scheduler": {
                "type": config.training.scheduler.type,
                "base_lr": config.training.scheduler.base_lr,
                "final_lr": config.training.scheduler.final_lr,
                "warmup_epochs": config.training.scheduler.warmup_epochs,
                "decay_end_epoch": config.training.scheduler.decay_end_epoch,
                "warmup_from_zero": config.training.scheduler.warmup_from_zero,
            },
        },
        "gan": {
            "disc_weight": config.gan_loss.disc_weight,
            "perceptual_weight": config.gan_loss.perceptual_weight,
            "disc_start_epoch": config.gan_loss.disc_start_epoch,
            "disc_update_start_epoch": config.gan_loss.disc_update_start_epoch,
            "lpips_start_epoch": config.gan_loss.lpips_start_epoch,
            "disc_updates": config.gan_loss.disc_updates,
            "disc_arch": {
                "dino_ckpt_path": config.discriminator.arch.dino_ckpt_path,
                "ks": config.discriminator.arch.ks,
                "norm_type": config.discriminator.arch.norm_type,
                "using_spec_norm": config.discriminator.arch.using_spec_norm,
                "recipe": config.discriminator.arch.recipe,
            },
            "augment": {
                "prob": config.discriminator.augment_prob,
                "cutout": config.discriminator.augment_cutout,
            },
        },
        "evaluation": {
            "enabled": config.evaluation.enabled,
            "eval_every": config.evaluation.eval_every,
            "eval_model": config.evaluation.eval_model,
            "metrics": list(config.evaluation.metrics),
            "reference_npz_path": config.evaluation.reference_npz_path,
        },
    }
