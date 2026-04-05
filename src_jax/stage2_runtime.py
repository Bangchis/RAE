from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import os
import time
from collections import defaultdict
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


WANDB_RESUME_METADATA_FILENAME = "wandb_run.json"


def _bridge_legacy_wandb_env(entity: str | None, project: str | None) -> None:
    if "WANDB_API_KEY" not in os.environ and "WANDB_KEY" in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]
    if entity and "WANDB_ENTITY" not in os.environ:
        os.environ["WANDB_ENTITY"] = entity
    elif "WANDB_ENTITY" not in os.environ and "ENTITY" in os.environ:
        os.environ["WANDB_ENTITY"] = os.environ["ENTITY"]
    if project and "PROJECT" not in os.environ:
        os.environ["PROJECT"] = project


def _wandb_resume_metadata_path(workdir: Path) -> Path:
    return workdir / WANDB_RESUME_METADATA_FILENAME


def _load_wandb_resume_metadata(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = ("run_id", "entity", "project", "exp_name")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"{path} is missing required W&B resume keys: {', '.join(missing)}")
    return {key: str(payload[key]) for key in required}


def _latest_orbax_checkpoint_step(workdir: Path) -> int | None:
    latest_step: int | None = None
    for path in workdir.glob("checkpoint_*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        latest_step = step if latest_step is None else max(latest_step, step)
    return latest_step


def _generate_fresh_wandb_run_id(wandb_utils: Any) -> str:
    generate_id = getattr(getattr(getattr(wandb_utils, "wandb", None), "util", None), "generate_id", None)
    if callable(generate_id):
        run_id = generate_id()
        if run_id:
            return str(run_id)
    return os.urandom(4).hex()


def _resolve_wandb_resume_binding(
    *,
    workdir: Path,
    entity: str,
    project_name: str,
    exp_name: str,
    explicit_run_id: str | None,
    wandb_utils: Any,
) -> tuple[dict[str, str], dict[str, str]]:
    metadata_path = _wandb_resume_metadata_path(workdir)
    metadata = _load_wandb_resume_metadata(metadata_path)
    checkpoint_step = _latest_orbax_checkpoint_step(workdir)
    if metadata is not None:
        if explicit_run_id and explicit_run_id != metadata["run_id"]:
            raise ValueError(
                f"Explicit W&B run id {explicit_run_id!r} does not match stored metadata {metadata['run_id']!r} in {metadata_path}."
            )
        mismatches: list[str] = []
        for key, current_value in (
            ("entity", entity),
            ("project", project_name),
            ("exp_name", exp_name),
        ):
            if metadata[key] != current_value:
                mismatches.append(f"{key}: stored={metadata[key]!r}, current={current_value!r}")
        if mismatches:
            raise ValueError(
                f"W&B resume metadata mismatch for {workdir}: " + "; ".join(mismatches)
            )
        if checkpoint_step is None:
            raise FileNotFoundError(
                f"Cannot rewind W&B run {metadata['run_id']!r} for {workdir} because no Orbax checkpoint_* directory exists."
            )
        return metadata, {"resume_from": f"{metadata['run_id']}?_step={checkpoint_step}"}

    if explicit_run_id:
        if checkpoint_step is None:
            raise FileNotFoundError(
                f"Cannot bind legacy W&B run {explicit_run_id!r} for {workdir} because no Orbax checkpoint_* directory exists."
            )
        return {
            "run_id": str(explicit_run_id),
            "entity": entity,
            "project": project_name,
            "exp_name": exp_name,
        }, {"resume_from": f"{explicit_run_id}?_step={checkpoint_step}"}

    if checkpoint_step is not None:
        raise FileNotFoundError(
            f"Missing {_wandb_resume_metadata_path(workdir)} for resume workdir {workdir}. "
            "Pass --wandb-run-id <existing-run-id> once to bind this legacy workdir to the correct W&B run."
        )

    fresh_run_id = _generate_fresh_wandb_run_id(wandb_utils)
    return {
        "run_id": fresh_run_id,
        "entity": entity,
        "project": project_name,
        "exp_name": exp_name,
    }, {"id": fresh_run_id, "resume": "never"}


def _install_strict_wandb_initializer(
    wandb_utils: Any,
    *,
    workdir: Path,
    explicit_run_id: str | None,
) -> None:
    def initialize(config: Any, exp_name: str = "dit", project_name: str = "tpu-dit") -> None:
        if not wandb_utils.is_main_process():
            return

        api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")
        if api_key is None:
            raise RuntimeError("WANDB_API_KEY is not set. Export it in your shell before launching training.")

        entity = os.environ.get("WANDB_ENTITY")
        if entity is None:
            raise RuntimeError("WANDB_ENTITY is not set. Export it in your shell before launching training.")

        metadata, init_kwargs = _resolve_wandb_resume_binding(
            workdir=workdir,
            entity=entity,
            project_name=project_name,
            exp_name=exp_name,
            explicit_run_id=explicit_run_id,
            wandb_utils=wandb_utils,
        )

        config_dict = config.to_dict() if hasattr(config, "to_dict") else config
        wandb_utils.wandb.login(key=api_key)
        run = wandb_utils.wandb.init(
            entity=entity,
            project=project_name,
            name=exp_name,
            config=config_dict,
            **init_kwargs,
        )

        payload = dict(metadata)
        payload["run_id"] = str(getattr(run, "id", metadata["run_id"])) if run is not None else metadata["run_id"]
        write_json(_wandb_resume_metadata_path(workdir), payload)

    wandb_utils.initialize = initialize


def _disable_backend_wandb(wandb_utils: Any) -> None:
    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    wandb_utils.initialize = _noop
    wandb_utils.log = _noop
    wandb_utils.log_copy = _noop
    wandb_utils.log_images = _noop
    wandb_utils.log_line_plot = _noop


def _running_on_kaggle() -> bool:
    return "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def _patch_backend_metric_writer_for_kaggle(trainer: Any) -> Any | None:
    if not _running_on_kaggle():
        return None

    original_create_default_writer = trainer.metric_writers.create_default_writer

    def patched_create_default_writer(
        logdir: str | os.PathLike[str] | None = None,
        *,
        just_logging: bool = False,
        asynchronous: bool = True,
        collection: str | None = None,
    ):
        del logdir, just_logging, asynchronous
        return original_create_default_writer(
            logdir=None,
            just_logging=True,
            asynchronous=False,
            collection=collection,
        )

    trainer.metric_writers.create_default_writer = patched_create_default_writer
    return original_create_default_writer


def _log_named_fid_scores(
    wandb_utils: Any,
    fid_scores: dict[int, float],
    *,
    guidance_scale: float,
    step: int,
    tag: str,
) -> None:
    if not fid_scores:
        return
    payload: dict[str, float | int] = {"train_step": int(step)}
    for num_samples, fid_value in fid_scores.items():
        payload[f"FID-{num_samples // 1000}K/{tag} (cfg={guidance_scale})"] = float(fid_value)
    wandb_utils.log(payload)


def _calculate_backend_fid(
    trainer: Any,
    config: Any,
    dataset: Any,
    sampler: Any,
    generator: Any,
    encoder: Any,
    guidance_scale: float,
    sample_sizes: Any,
    step: int,
    mesh: Any,
    *,
    tag: str | None = None,
) -> dict[int, float]:
    wandb_utils = trainer.wandb_utils
    original_log = wandb_utils.log

    if tag is not None:
        wandb_utils.log = lambda *_args, **_kwargs: None

    try:
        fid_scores = trainer.fid.calculate_fid(
            config,
            dataset,
            sampler,
            generator,
            encoder,
            guidance_scale,
            None,
            sample_sizes,
            step,
            mesh=mesh,
        )
    finally:
        if tag is not None:
            wandb_utils.log = original_log

    if tag is not None:
        _log_named_fid_scores(
            wandb_utils,
            fid_scores,
            guidance_scale=guidance_scale,
            step=step,
            tag=tag,
        )
    return fid_scores


def _build_backend_eval_dataset(trainer: Any, config: Any) -> Any | None:
    eval_data_dir = config.eval.get("data_dir")
    if not eval_data_dir:
        return None

    eval_root = Path(str(eval_data_dir)).expanduser().resolve()
    if not eval_root.exists():
        raise FileNotFoundError(f"Validation ImageFolder not found: {eval_root}")

    if config.data.get("latent_dataset", False):
        return trainer.local_imagenet_dataset.LatentDataset(
            str(eval_root),
            use_labels=True,
            cache=False,
        )

    transform = _build_backend_raw_transform(
        trainer,
        int(config.data.image_size),
        random_flip=bool(config.eval.get("random_flip", False)),
    )
    return trainer.local_imagenet_dataset.datasets.ImageFolder(root=str(eval_root), transform=transform)


def _build_backend_raw_transform(trainer: Any, image_size: int, *, random_flip: bool) -> Any:
    from torchvision import transforms

    crop_fn = lambda image: trainer.local_imagenet_dataset.utils.center_crop_arr(image, image_size)
    transform_steps: list[Any] = [crop_fn]
    if random_flip:
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(transform_steps)


def _build_backend_train_dataset(trainer: Any, config: Any, image_size: int) -> Any:
    if config.data.get("latent_dataset", False):
        return trainer.local_imagenet_dataset.LatentDataset(
            config.data.data_dir,
            use_labels=True,
            cache=False,
        )

    train_root = Path(str(config.data.data_dir)).expanduser().resolve() / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Training ImageFolder split not found: {train_root}")

    transform = _build_backend_raw_transform(
        trainer,
        image_size,
        random_flip=bool(config.data.get("random_flip", False)),
    )
    return trainer.local_imagenet_dataset.datasets.ImageFolder(
        root=str(train_root),
        transform=transform,
    )


def _resolve_prefetch_factor(raw: Any, *, num_workers: int) -> int | None:
    if num_workers <= 0:
        return None
    if raw is None:
        return 2
    value = int(raw)
    if value <= 0:
        raise ValueError("prefetch_factor must be greater than 0 when num_workers > 0.")
    return value


def _build_backend_train_loader(trainer: Any, config: Any, dataset: Any, *, offset_seed: int) -> Any:
    import torch

    batch_size = int(config.data.batch_size)
    local_batch_size = batch_size // max(1, trainer.jax.process_count())
    num_workers = int(config.data.num_workers)
    if num_workers < 0:
        raise ValueError("training.num_workers must be non-negative.")

    sampler = trainer.local_imagenet_dataset.InfiniteSampler(
        dataset,
        num_replicas=max(1, trainer.jax.process_count()),
        rank=trainer.jax.process_index(),
        shuffle=True,
        seed=int(config.data.seed),
    )

    rng_torch = torch.Generator()
    rng_torch.manual_seed(offset_seed)

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "sampler": sampler,
        "batch_size": local_batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": True,
        "generator": rng_torch,
    }
    if num_workers > 0:
        loader_kwargs.update(
            {
                "worker_init_fn": functools.partial(
                    trainer.local_imagenet_dataset.seed_worker,
                    offset_seed=offset_seed,
                    global_seed=int(config.data.seed_pt),
                ),
                "persistent_workers": True,
                "timeout": 1800.0,
                "prefetch_factor": _resolve_prefetch_factor(
                    config.data.get("prefetch_factor"),
                    num_workers=num_workers,
                ),
            }
        )

    return torch.utils.data.DataLoader(**loader_kwargs)


def _build_backend_eval_loader(trainer: Any, config: Any, dataset: Any) -> Any:
    import torch

    per_device_batch_size = int(config.eval.get("loss_batch_size", 4))
    if per_device_batch_size <= 0:
        raise ValueError("eval.batch_size must be greater than 0 for JAX validation loss.")

    local_batch_size = per_device_batch_size * max(1, trainer.jax.local_device_count())
    num_workers = int(config.eval.get("num_workers", config.data.num_workers))
    if num_workers < 0:
        raise ValueError("eval.num_workers must be non-negative.")
    prefetch_factor = _resolve_prefetch_factor(
        config.eval.get("prefetch_factor", config.data.get("prefetch_factor")),
        num_workers=num_workers,
    )

    process_index = trainer.jax.process_index()
    process_count = max(1, trainer.jax.process_count())
    local_indices = list(range(process_index, len(dataset), process_count))
    subset = torch.utils.data.Subset(dataset, local_indices)
    loader_kwargs: dict[str, Any] = {
        "dataset": subset,
        "batch_size": local_batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs.update(
            {
                "worker_init_fn": functools.partial(
                    trainer.local_imagenet_dataset.seed_worker,
                    offset_seed=0,
                    global_seed=int(config.data.seed_pt),
                ),
                "persistent_workers": True,
                "timeout": 60.0,
                "prefetch_factor": prefetch_factor,
            }
        )

    return torch.utils.data.DataLoader(**loader_kwargs)


def _metric_tree_to_host(metric_dict: Any) -> dict[str, float]:
    import jax

    host_metrics = jax.device_get(metric_dict)
    return {key: float(np.asarray(value)) for key, value in host_metrics.items()}


def _set_intermediate_feature_logging(model: Any, enabled: bool) -> None:
    if hasattr(model, "ema"):
        model = model.ema
    interface = model.interface if hasattr(model, "interface") else model
    network = getattr(interface, "network", None)
    if network is not None and hasattr(network, "return_intermediate_features"):
        network.return_intermediate_features = bool(enabled)


def _build_sitdh_activation_names(num_encoder_blocks: int, num_decoder_blocks: int) -> list[tuple[str, int]]:
    names: list[tuple[str, int]] = []
    for idx in range(num_encoder_blocks):
        names.append(("enc", idx))
    for idx in range(num_decoder_blocks):
        names.append(("dec", idx))
    return names


def _patch_backend_train_loop_for_eval(trainer: Any) -> Any:
    original_train_and_evaluate = trainer.train_and_evaluate

    def eval_step(state: Any, batch: dict[str, Any], graph: Any, *, use_ema: bool) -> dict[str, Any]:
        merged = trainer.nnx.merge(graph, state)
        if use_ema:
            model = merged.ema
        else:
            model = merged.model
        model = model.interface if hasattr(model, "interface") else model

        latents, labels = batch["latents"], batch["labels"]
        if "features" in batch:
            loss_dict = model(latents, batch["features"], y=labels)
        else:
            loss_dict = model(latents, y=labels)
        return {loss_type: loss.mean() for loss_type, loss in loss_dict.items()}

    def run_validation_pass(
        *,
        state: Any,
        graph: Any,
        use_ema: bool,
        loader: Any,
        encoder: Any,
        detector: Any,
        mesh: Any,
        p_eval_step: Any,
        max_batches: int,
    ) -> dict[str, float]:
        metric_sums: dict[str, float] = {}
        total_samples = 0
        used_batches = 0
        start_t = time.time()

        for raw_batch in loader:
            batch_images = raw_batch[0]
            if int(batch_images.shape[0]) % max(1, trainer.jax.local_device_count()) != 0:
                continue

            parsed_batch = trainer.data_utils.parse_batch(raw_batch, encoder, mesh, detector=detector)
            with mesh:
                metric_dict = p_eval_step(state, parsed_batch, graph)

            batch_metrics = _metric_tree_to_host(metric_dict)
            batch_size = int(batch_images.shape[0]) * max(1, trainer.jax.process_count())
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size
            total_samples += batch_size
            used_batches += 1
            if max_batches > 0 and used_batches >= max_batches:
                break

        duration = time.time() - start_t
        summary: dict[str, float] = {}
        prefix = "ema" if use_ema else "model"

        count_vec = np.array([total_samples, used_batches, duration], dtype=np.float64)
        gathered_counts = np.asarray(
            trainer.jax.experimental.multihost_utils.process_allgather(count_vec, tiled=False)
        )
        if gathered_counts.ndim == 1:
            global_counts = gathered_counts
        else:
            global_counts = gathered_counts.sum(axis=0)
            global_counts[2] = gathered_counts[:, 2].max()

        global_total_samples = int(global_counts[0])
        global_used_batches = int(global_counts[1])
        max_duration = float(global_counts[2])

        if metric_sums:
            metric_names = sorted(metric_sums)
            metric_vec = np.asarray([metric_sums[name] for name in metric_names], dtype=np.float64)
            gathered_metrics = np.asarray(
                trainer.jax.experimental.multihost_utils.process_allgather(metric_vec, tiled=False)
            )
            global_metric_vec = gathered_metrics if gathered_metrics.ndim == 1 else gathered_metrics.sum(axis=0)
            if global_total_samples > 0:
                for idx, name in enumerate(metric_names):
                    summary[f"eval/{prefix}_{name}"] = float(global_metric_vec[idx] / global_total_samples)

        summary[f"eval/{prefix}_batches"] = float(global_used_batches)
        summary[f"eval/{prefix}_samples"] = float(global_total_samples)
        summary[f"eval/{prefix}_duration_sec"] = max_duration
        return summary

    def patched_train_and_evaluate(config: Any, workdir: str):
        diagnostics_cfg = config.get("diagnostics", {})
        log_rae_latent_stats = bool(diagnostics_cfg.get("log_rae_latent_stats", False))
        log_activation_stats = bool(diagnostics_cfg.get("log_activation_stats", False))
        num_encoder_blocks = int(config.network.get("num_encoder_blocks", 0))
        num_decoder_blocks = int(config.network.get("num_decoder_blocks", 0))
        activation_names = _build_sitdh_activation_names(num_encoder_blocks, num_decoder_blocks)
        if not config.eval.get("loss_on") and not log_rae_latent_stats and not log_activation_stats:
            return original_train_and_evaluate(config, workdir)

        image_size = config.data.image_size

        writer = trainer.metric_writers.create_default_writer(
            logdir=workdir, just_logging=trainer.jax.process_index() != 0
        )

        if config.standalone_eval:
            exp_name = f"{config.exp_name}_eval"
            project_name = "evaluation"
        else:
            exp_name = config.exp_name
            project_name = config.project_name

        trainer.wandb_utils.initialize(config, exp_name=exp_name, project_name=project_name)

        if config.data.batch_size % trainer.jax.device_count() > 0:
            raise ValueError("Batch size must be divisible by the number of devices")

        dataset = _build_backend_train_dataset(trainer, config, image_size)

        encoder, model, optimizer, sampler, ema, learning_rate_fn = trainer.init_utils.build_models(config)

        detector = None
        if config.get("repa"):
            detector = trainer.init_utils.instantiate_detector(config)
            model = trainer.init_utils.instantiate_repa(
                config, model, feature_dim=detector.network.config.hidden_size
            )
            optimizer, _ = trainer.init_utils.instantiate_optimizer(config, model)

        ckpt_mngr = trainer.ckpt_utils.build_checkpoint_manager(workdir, **config.checkpoint.options)
        if config.standalone_eval:
            restore_step = config.get("restore_step") if config.get("restore_step") else ckpt_mngr.latest_step()
        else:
            restore_step = ckpt_mngr.latest_step()

        opt_graph, opt_rng_state, opt_state = trainer.nnx.split(optimizer, trainer.nnx.RngKey, ...)

        if config.get("pretrained_ckpt"):
            _, _, ema_state = trainer.nnx.split(ema, trainer.nnx.RngState, ...)
            ema_state = trainer.ckpt_utils.restore_checkpoints(
                config.pretrained_ckpt, 0, opt_state, opt_rng_state, ema_state, ema_only=True
            )
            trainer.nnx.update(ema, ema_state)

        _, _, ema_state = trainer.nnx.split(ema, trainer.nnx.RngKey, ...)

        loaded_state, loaded_rng_state, loaded_ema_state = trainer.ckpt_utils.restore_checkpoints(
            workdir, restore_step, opt_state, opt_rng_state, ema_state, mngr=ckpt_mngr
        )

        mesh = trainer.sharding_utils.create_device_mesh(
            config.sharding.mesh,
            allow_split_physical_axes=config.sharding.get("mesh_allow_split_physical_axes", False),
        )

        repl_sharding = trainer.NamedSharding(mesh, trainer.P())
        (
            graphdef,
            state,
            ema_graphdef,
            ema_state,
            state_sharding,
            ema_state_sharding,
        ) = trainer.sharding_utils.update_model_sharding(
            opt_graph,
            loaded_state,
            loaded_rng_state,
            ema,
            loaded_ema_state,
            mesh=mesh,
            sharding_strategy=config.sharding.strategy,
        )

        del opt_state, opt_rng_state, loaded_state, loaded_ema_state

        optimizer = trainer.nnx.merge(graphdef, state)
        ema = trainer.nnx.merge(ema_graphdef, ema_state)
        model = optimizer.model.interface if hasattr(optimizer.model, "interface") else optimizer.model

        step = 0 if restore_step is None else restore_step

        loader = _build_backend_train_loader(trainer, config, dataset, offset_seed=step)
        eval_dataset = _build_backend_eval_dataset(trainer, config)
        eval_loader = _build_backend_eval_loader(trainer, config, eval_dataset) if eval_dataset is not None else None

        if config.visualize.get("on"):
            trainer.vis_utils.visualize(config, model, ema.ema, encoder, sampler, step, mesh=mesh)
            if config.visualize.get("visualize_reconstruction"):
                in_x, _ = next(iter(loader))
                in_x = in_x[: config.visualize.num_samples].permute([0, 2, 3, 1]).numpy()
                trainer.vis_utils.visualize_reconstruction(config, encoder, in_x, mesh=mesh)

        if config.eval.get("fid_on") and config.eval.get("on_load"):
            for guidance_scale, sample_sizes in zip(
                config.eval.all_guidance_scales,
                config.eval.all_eval_samples_nums,
            ):
                _calculate_backend_fid(
                    trainer,
                    config,
                    dataset,
                    sampler,
                    ema.ema,
                    encoder,
                    guidance_scale,
                    sample_sizes,
                    step,
                    mesh,
                )
                if config.eval.get("fid_eval_model", False):
                    _calculate_backend_fid(
                        trainer,
                        config,
                        dataset,
                        sampler,
                        model,
                        encoder,
                        guidance_scale,
                        sample_sizes,
                        step,
                        mesh,
                        tag="model",
                    )

        if config.standalone_eval:
            return

        metrics_history = defaultdict(list)
        metrics_interval = defaultdict(list)
        train_metrics_last_t = time.time()
        loader_iter = iter(loader)

        def diag_stat_rms(x: Any) -> Any:
            x = trainer.jnp.asarray(x, dtype=trainer.jnp.float32)
            return trainer.jnp.sqrt(trainer.jnp.mean(trainer.jnp.square(x)))

        def diag_stat_var(x: Any) -> Any:
            x = trainer.jnp.asarray(x, dtype=trainer.jnp.float32)
            return trainer.jnp.var(x)

        def patched_train_step(
            state: Any,
            ema_state: Any,
            batch: Any,
            graph: Any,
            ema_graph: Any,
        ):
            optimizer = trainer.nnx.merge(graph, state)
            ema = trainer.nnx.merge(ema_graph, ema_state)
            model = optimizer.model

            latents, labels = batch["latents"], batch["labels"]

            def loss_fn(model):
                if log_activation_stats:
                    if "features" in batch:
                        loss_vec, net_out, intermediate_features = model(
                            latents,
                            batch["features"],
                            y=labels,
                            return_aux=True,
                        )
                    else:
                        loss_vec, net_out, intermediate_features = model(
                            latents,
                            y=labels,
                            return_aux=True,
                        )
                    loss_dict = {"loss": loss_vec}
                else:
                    if "features" in batch:
                        loss_dict = model(latents, batch["features"], y=labels)
                    else:
                        loss_dict = model(latents, y=labels)
                    net_out = None
                    intermediate_features = ()

                metric_dict = {}
                if log_rae_latent_stats:
                    metric_dict["rae_latent_rms"] = diag_stat_rms(latents)
                    metric_dict["rae_latent_var"] = diag_stat_var(latents)
                if log_activation_stats:
                    metric_dict["sitdh_output_rms"] = diag_stat_rms(net_out)
                    metric_dict["sitdh_output_var"] = diag_stat_var(net_out)
                    for (stage_name, block_idx), feature in zip(
                        activation_names,
                        intermediate_features,
                        strict=False,
                    ):
                        metric_dict[f"sitdh_act_{stage_name}_{block_idx:02d}_rms"] = diag_stat_rms(feature)
                        metric_dict[f"sitdh_act_{stage_name}_{block_idx:02d}_var"] = diag_stat_var(feature)
                return loss_dict["loss"].mean(), (loss_dict, metric_dict)

            grad_fn = trainer.nnx.value_and_grad(loss_fn, has_aux=True)
            (loss, (loss_dict, extra_metric_dict)), grads = grad_fn(model)

            optimizer.update(grads)

            grad_norm = trainer.jax.tree_util.tree_reduce(
                lambda a, b: a + b,
                trainer.jax.tree_util.tree_map(lambda g: trainer.jnp.sum(trainer.jnp.square(g)), grads),
                initializer=0.0,
            )

            if hasattr(model, "interface"):
                ema.update(model.interface)
            else:
                ema.update(model)
            metric_dict = {
                loss_type: loss.mean() for loss_type, loss in loss_dict.items()
            }
            metric_dict.update(extra_metric_dict)
            metric_dict["grad_norm"] = grad_norm

            _, state = trainer.nnx.split(optimizer)
            _, ema_state = trainer.nnx.split(ema)
            return state, ema_state, metric_dict

        p_train_step = trainer.jax.jit(
            patched_train_step,
            out_shardings=(state_sharding, ema_state_sharding, repl_sharding),
            static_argnums=(3, 4),
            donate_argnums=(0, 1),
        )

        p_eval_ema_step = trainer.jax.jit(
            lambda cur_state, batch, graph: eval_step(cur_state, batch, graph, use_ema=True),
            out_shardings=repl_sharding,
            static_argnums=(2,),
        )
        p_eval_model_step = trainer.jax.jit(
            lambda cur_state, batch, graph: eval_step(cur_state, batch, graph, use_ema=False),
            out_shardings=repl_sharding,
            static_argnums=(2,),
        )

        def sync_state(cur_state: Any):
            return cur_state

        p_sync_state = trainer.jax.jit(sync_state, out_shardings=repl_sharding)

        hooks = []
        report_progress = trainer.periodic_actions.ReportProgress(
            num_train_steps=config.total_steps, writer=writer
        )
        if trainer.jax.process_index() == 0:
            hooks += [
                report_progress,
                trainer.periodic_actions.Profile(logdir=workdir, num_profile_steps=5),
            ]

        for i in range(step, config.total_steps):
            batch = trainer.data_utils.parse_batch(next(loader_iter), encoder, mesh, detector=detector)

            with trainer.jax.profiler.StepTraceAnnotation("train", step_num=step):
                with mesh:
                    state, ema_state, metric_dict = p_train_step(state, ema_state, batch, graphdef, ema_graphdef)

                for key, value in metric_dict.items():
                    metrics_interval[key].append(value)

            if (restore_step is None or step == restore_step) and i == 0:
                trainer.logging.info("Initial compilation completed.")

            for hook in hooks:
                hook(step)

            if config.get("log_every_steps") and (step + 1) % config.log_every_steps == 0:
                for key, value in metrics_interval.items():
                    metrics_history[key].append(sum(value) / len(value))
                metrics_interval = defaultdict(list)

                summary = {f"train_{key}": float(value[-1]) for key, value in metrics_history.items()}
                summary["steps_per_second"] = config.log_every_steps / (time.time() - train_metrics_last_t)
                summary["learning_rate"] = learning_rate_fn(step)
                summary["step"] = step + 1

                trainer.wandb_utils.log_copy(summary)
                writer.write_scalars(step + 1, summary)
                metrics_history = defaultdict(list)
                train_metrics_last_t = time.time()

            loss_every_steps = int(config.eval.get("loss_every_steps", 0))
            if eval_loader is not None and loss_every_steps > 0 and (step + 1) % loss_every_steps == 0:
                eval_summary = run_validation_pass(
                    state=ema_state,
                    graph=ema_graphdef,
                    use_ema=True,
                    loader=eval_loader,
                    encoder=encoder,
                    detector=detector,
                    mesh=mesh,
                    p_eval_step=p_eval_ema_step,
                    max_batches=int(config.eval.get("max_batches", 0)),
                )
                if config.eval.get("eval_model", False):
                    eval_summary.update(
                        run_validation_pass(
                            state=state,
                            graph=graphdef,
                            use_ema=False,
                            loader=eval_loader,
                            encoder=encoder,
                            detector=detector,
                            mesh=mesh,
                            p_eval_step=p_eval_model_step,
                            max_batches=int(config.eval.get("max_batches", 0)),
                        )
                    )
                eval_summary["step"] = step + 1
                trainer.wandb_utils.log_copy(eval_summary)
                writer.write_scalars(step + 1, eval_summary)

            if config.visualize.get("on") and (step + 1) % config.visualize_every_steps == 0:
                trainer.nnx.update(ema, ema_state)
                trainer.nnx.update(optimizer, state)
                model = optimizer.model.interface if hasattr(optimizer.model, "interface") else optimizer.model
                trainer.vis_utils.visualize(config, model, ema.ema, encoder, sampler, step, mesh=mesh)

            if config.eval.get("fid_on"):
                for guidance_scale, sample_sizes, every_steps in zip(
                    config.eval.all_guidance_scales,
                    config.eval.all_eval_samples_nums,
                    config.eval.eval_every_steps,
                ):
                    time_for_fid, sample_sizes = trainer.logging_utils.is_it_time_for_fid(
                        sample_sizes, every_steps, step
                    )
                    if time_for_fid:
                        trainer.nnx.update(ema, ema_state)
                        _calculate_backend_fid(
                            trainer,
                            config,
                            dataset,
                            sampler,
                            ema.ema,
                            encoder,
                            guidance_scale,
                            sample_sizes,
                            step,
                            mesh,
                        )
                        if config.eval.get("fid_eval_model", False):
                            trainer.nnx.update(optimizer, state)
                            model = (
                                optimizer.model.interface
                                if hasattr(optimizer.model, "interface")
                                else optimizer.model
                            )
                            _calculate_backend_fid(
                                trainer,
                                config,
                                dataset,
                                sampler,
                                model,
                                encoder,
                                guidance_scale,
                                sample_sizes,
                                step,
                                mesh,
                                tag="model",
                            )

            if (step + 1) % config.save_every_steps == 0 or step + 1 == config.total_steps:
                trainer.nnx.update(ema, ema_state)
                trainer.nnx.update(optimizer, state)
                _, saved_rng_state, saved_state = trainer.nnx.split(optimizer, trainer.nnx.RngKey, ...)
                saved_state, saved_rng_state = trainer.jax.device_get(
                    (p_sync_state(saved_state), p_sync_state(saved_rng_state))
                )
                _, _, saved_ema_state = trainer.nnx.split(ema, trainer.nnx.RngKey, ...)
                saved_ema_state = trainer.jax.device_get(p_sync_state(saved_ema_state))
                trainer.ckpt_utils.save_checkpoints(
                    workdir,
                    step + 1,
                    saved_state,
                    saved_rng_state,
                    saved_ema_state,
                    mngr=ckpt_mngr,
                )
                del saved_state, saved_rng_state, saved_ema_state

            step += 1

        return metrics_history

    trainer.train_and_evaluate = patched_train_and_evaluate
    return original_train_and_evaluate


def _maybe_raise_backend_dependency_hint(exc: ImportError) -> None:
    message = str(exc)
    if (
        ("FlaxDinov2Model" in message and "transformers" in message)
        or "Dinov2WithRegistersModel" in message
        or "dinov2_with_registers" in message
    ):
        raise ImportError(
            "The patched diffuse_nnx backend expects transformers==4.57.1 so it can import "
            "the Dinov2 Flax and Dinov2-with-registers modules from their subpackages. "
            "Reinstall that version in the active environment, for example: "
            "uv pip install --reinstall \"transformers==4.57.1\"."
        ) from exc
    if "No module named 'google.cloud'" in message or "No module named 'google'" in message:
        raise ImportError(
            "The patched diffuse_nnx backend should lazy-load google-cloud-storage, so the "
            "RAE/DINO Stage-1 path does not need that package at import time. Sync the latest "
            "repo so `src_jax/vendor.py` can patch the cached backend checkout, or as a temporary "
            "workaround install it with: uv pip install google-cloud-storage."
        ) from exc


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

    try:
        from utils import checkpoint as ckpt_utils
        from utils import initialize as init_utils
        from networks.transformers import port_torch_to_nnx as port_module
        from utils import wandb_utils as backend_wandb
    except ImportError as exc:
        _maybe_raise_backend_dependency_hint(exc)
        raise

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

    try:
        from networks.transformers import port_torch_to_nnx as port_module
        from trainers import dit_imagenet as trainer
        from utils import initialize as init_utils
        from utils import wandb_utils as backend_wandb
    except ImportError as exc:
        _maybe_raise_backend_dependency_hint(exc)
        raise

    if args.wandb:
        _bridge_legacy_wandb_env(args.wandb_entity, args.wandb_project)
        _install_strict_wandb_initializer(
            backend_wandb,
            workdir=workdir,
            explicit_run_id=getattr(args, "wandb_run_id", None),
        )
        if getattr(trainer, "wandb_utils", None) is not backend_wandb:
            _install_strict_wandb_initializer(
                trainer.wandb_utils,
                workdir=workdir,
                explicit_run_id=getattr(args, "wandb_run_id", None),
            )
    else:
        _disable_backend_wandb(backend_wandb)

    backend_cfg = _to_config_dict(backend_cfg_dict)

    original_build_models = init_utils.build_models

    def patched_build_models(config: Any):
        encoder, model, optimizer, sampler, ema, learning_rate_fn = original_build_models(config)
        log_activation_stats = bool(config.get("diagnostics", {}).get("log_activation_stats", False))
        _set_intermediate_feature_logging(model, log_activation_stats)
        _set_intermediate_feature_logging(ema, log_activation_stats)
        if backend_cfg_dict.get("torch_ckpt"):
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

    original_create_default_writer = _patch_backend_metric_writer_for_kaggle(trainer)
    original_train_and_evaluate = _patch_backend_train_loop_for_eval(trainer)

    try:
        trainer.train_and_evaluate(backend_cfg, str(workdir))
    finally:
        if original_create_default_writer is not None:
            trainer.metric_writers.create_default_writer = original_create_default_writer
        trainer.train_and_evaluate = original_train_and_evaluate
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
