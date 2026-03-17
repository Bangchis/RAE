# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Stage-2 SiT training script for torch-xla devices.
"""
from __future__ import annotations

import argparse
from functools import partial
import logging
import math
import os
import random
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.amp import autocast
import torch_xla.distributed.parallel_loader as pl

from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport
from utils import wandb_utils
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.optim_utils import build_optimizer, build_scheduler
from utils.train_utils import initialize_cache, set_random_seed, parse_configs
from utils.sample_utils import manual_sample, make_timesteps  # noqa: E402
#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def cleanup() -> None:
    """Synchronize processes before shutdown."""
    xm.rendezvous("cleanup")


def create_logger(logging_dir: Optional[str], is_master: bool) -> logging.Logger:
    """Create a logger that writes to a log file and stdout on the master rank."""
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if is_master:
        formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(logging_dir, "log.txt"))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class IndexedImageFolder(ImageFolder):
    """ImageFolder variant that also returns the dataset index for weighted eval."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, target = super().__getitem__(index)
        return image, target, index


def build_eval_index_weights(sampler: DistributedSampler, dataset_size: int) -> torch.Tensor:
    """Compensate for DistributedSampler padding so eval means stay exact."""
    if dataset_size <= 0:
        raise ValueError("Evaluation dataset must contain at least one image.")

    indices = list(range(dataset_size))
    if not sampler.drop_last:
        padding_size = sampler.total_size - len(indices)
        if padding_size > 0:
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                repeats = math.ceil(padding_size / len(indices))
                indices += (indices * repeats)[:padding_size]
    else:
        indices = indices[: sampler.total_size]

    counts = torch.bincount(torch.tensor(indices, dtype=torch.long), minlength=dataset_size).to(torch.float32)
    return counts.reciprocal()


@torch.no_grad()
def run_validation_loss(
    model: Stage2ModelProtocol,
    rae: RAE,
    transport: Any,
    host_loader: DataLoader,
    device: torch.device,
    autocast_kwargs: Dict[str, Any],
    index_weights: torch.Tensor,
    reduce_prefix: str,
    max_batches: Optional[int] = None,
) -> tuple[float, int]:
    weighted_loss_sum = 0.0
    weighted_sample_count = 0.0
    total_batches = 0
    device_loader = pl.ParallelLoader(host_loader, [device]).per_device_loader(device)

    for batch_idx, (images, labels, indices) in enumerate(device_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)
        indices = indices.to(device=device, dtype=torch.long)

        with autocast(**autocast_kwargs):
            latents = rae.encode(images)
            batch_losses = transport.training_losses(model, latents, dict(y=labels))["loss"].to(torch.float32)

        sample_weights = index_weights.index_select(0, indices)
        weighted_loss_sum += (batch_losses * sample_weights).sum().item()
        weighted_sample_count += sample_weights.sum().item()
        total_batches += 1
        xm.mark_step()

    del device_loader

    total_loss_sum = xm.mesh_reduce(f"{reduce_prefix}_loss_sum", weighted_loss_sum, sum)
    total_weight = xm.mesh_reduce(f"{reduce_prefix}_sample_weight", weighted_sample_count, sum)
    global_batches = int(xm.mesh_reduce(f"{reduce_prefix}_batch_count", total_batches, sum))
    if total_weight <= 0:
        raise RuntimeError("Validation loss reduction produced zero total weight.")
    return float(total_loss_sum / total_weight), global_batches


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args: argparse.Namespace) -> None:
    """Train a SiT model using torch-xla native data parallelism."""
    initialize_cache(is_sample=False)
    full_cfg = OmegaConf.load(args.config)
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
    ) = parse_configs(args.config)

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section: Optional[OmegaConf]) -> Dict[str, Any]:
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)  # type: ignore[return-value]

    eval_cfg: Dict[str, Any] = to_dict(full_cfg.get("eval"))
    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    guidance_cfg: Dict[str, Any] = to_dict(guidance_config)
    training_cfg = to_dict(training_config)

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")

    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    if clip_grad <= 0:
        clip_grad = 0.0
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = int(training_cfg.get("global_batch_size", 1024))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 100))
    ckpt_every = int(training_cfg.get("ckpt_every", 5_000))
    sample_every = int(training_cfg.get("sample_every", 10_000))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed

    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    device = xm.xla_device()
    set_random_seed(global_seed)
    if global_batch_size % (world_size * grad_accum_steps) != 0:
        raise ValueError("Global batch size must be divisible by world_size * grad_accum_steps.")

    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    eval_enabled = bool(eval_cfg)
    eval_every = 0
    eval_model = False
    eval_batch_size = micro_batch_size
    eval_num_workers = num_workers
    eval_max_batches: Optional[int] = None
    eval_data_path: Optional[str] = None
    if eval_enabled:
        eval_data_path = eval_cfg.get("data_path")
        if not eval_data_path:
            raise ValueError("eval.data_path must be specified when the eval block is enabled.")
        eval_every = int(eval_cfg.get("eval_every", 0))
        if eval_every <= 0:
            raise ValueError("eval.eval_every must be greater than 0 when evaluation is enabled.")
        eval_batch_size = int(eval_cfg.get("batch_size", micro_batch_size))
        if eval_batch_size <= 0:
            raise ValueError("eval.batch_size must be greater than 0.")
        eval_num_workers = int(eval_cfg.get("num_workers", num_workers))
        if eval_num_workers < 0:
            raise ValueError("eval.num_workers must be >= 0.")
        raw_max_batches = eval_cfg.get("max_batches")
        if raw_max_batches is not None:
            eval_max_batches = int(raw_max_batches)
            if eval_max_batches <= 0:
                raise ValueError("eval.max_batches must be greater than 0 when provided.")
        eval_model = bool(eval_cfg.get("eval_model", False))
    use_bf16 = args.precision == "bf16"
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_kwargs = dict(device=device, dtype=autocast_dtype, enabled=use_bf16)

    transport_params = dict(transport_cfg.get("params", {}))
    path_type = transport_params.get("path_type", "Linear")
    prediction = transport_params.get("prediction", "velocity")
    loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper() if sampler_cfg else "ODE"
    sampler_params = dict(sampler_cfg.get("params", {})) if sampler_cfg else {}

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))

    is_master = xm.is_master_ordinal()
    wandb_utils.is_main_process = lambda: xm.is_master_ordinal()

    if is_master:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_target = str(model_config.get("target", "stage2"))
        model_string_name = model_target.split(".")[-1]
        precision_suffix = f"-{args.precision}" if args.precision == "bf16" else ""
        loss_weight_str = loss_weight if loss_weight is not None else "none"
        experiment_name = (
            f"{experiment_index:03d}-{model_string_name}-"
            f"{path_type}-{prediction}-{loss_weight_str}{precision_suffix}-acc{grad_accum_steps}"
        )
        experiment_dir = os.path.join(args.results_dir, experiment_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        logger = create_logger(experiment_dir, is_master=True)
        logger.info(f"Experiment directory created at {experiment_dir}")
        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(
                args,
                entity,
                experiment_name,
                project,
                run_config=OmegaConf.to_container(full_cfg, resolve=True),
            )
    else:
        experiment_dir = None
        checkpoint_dir = None
        logger = create_logger(None, is_master=False)

    xm.rendezvous("experiment_setup")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    xm.broadcast_master_param(model)
    xm.broadcast_master_param(ema)

    opt_state: Optional[Dict[str, Any]] = None
    sched_state: Optional[Dict[str, Any]] = None
    start_epoch = 0
    train_steps = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        opt_state = checkpoint.get("opt")
        sched_state = checkpoint.get("scheduler")
        start_epoch = int(checkpoint.get("epoch", 0))
        train_steps = int(checkpoint.get("train_steps", 0))
        logger.info(f"Resumed checkpoint {args.ckpt} at epoch={start_epoch}, train_steps={train_steps}.")

    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count / 1e6:.2f}M")

    opt, opt_msg = build_optimizer(model.parameters(), training_cfg)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=global_seed,
    )
    host_loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    eval_loader: Optional[DataLoader] = None
    eval_index_weights: Optional[torch.Tensor] = None
    if eval_enabled:
        eval_transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
            ]
        )
        eval_dataset = IndexedImageFolder(eval_data_path, transform=eval_transform)
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            sampler=eval_sampler,
            num_workers=eval_num_workers,
            pin_memory=True,
            drop_last=False,
        )
        eval_index_weights = build_eval_index_weights(eval_sampler, len(eval_dataset)).to(device)
        logger.info(
            f"Evaluation dataset contains {len(eval_dataset):,} images ({eval_data_path}), "
            f"eval_every={eval_every}, per-device eval batch={eval_batch_size}, max_batches={eval_max_batches}."
        )

    logger.info(
        f"Gradient accumulation: steps={grad_accum_steps}, micro batch={micro_batch_size}, "
        f"per-device batch={micro_batch_size * grad_accum_steps}, global batch={global_batch_size}"
    )
    logger.info(f"Precision mode: {args.precision}")

    loader_batches = len(host_loader)
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")

    schedl, sched_msg = build_scheduler(opt, steps_per_epoch, training_cfg, sched_state)
    if is_master:
        logger.info(f"Training configured for {epochs} epochs, {steps_per_epoch} steps per epoch.")
        logger.info(opt_msg + "\n" + sched_msg)

    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    if sampler_mode == "ODE":
        num_steps = int(sampler_params.get("num_steps", 50))
        schedule = make_timesteps(num_steps=num_steps, t_min=1 / 1000, t_max=1.0, shift=time_dist_shift)
        sample_latents = partial(manual_sample, schedule=schedule, device=device)
    elif sampler_mode == "SDE":
        raise NotImplementedError("SDE sampling is not implemented yet.")
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    log_steps = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    start_time = time()

    ys = torch.randint(num_classes, size=(micro_batch_size,), device=device)
    using_cfg = guidance_scale > 1.0
    n = ys.size(0)
    zs = torch.randn(n, *latent_size, device=device)

    if using_cfg:
        zs = torch.cat([zs, zs], dim=0)
        y_null = torch.full((n,), null_label, device=device)
        ys = torch.cat([ys, y_null], dim=0)
        sample_labels = ys
        sample_forward_kwargs: Dict[str, Any] = dict(
            cfg_scale=guidance_scale,
            cfg_interval=(t_min, t_max),
        )
        if guidance_method == "autoguidance":
            if guid_model_forward is None:
                raise RuntimeError("Guidance model forward is not initialized.")
            sample_forward_kwargs["additional_model_forward"] = guid_model_forward
            model_fn = ema.forward_with_autoguidance
        else:
            model_fn = ema.forward_with_cfg
    else:
        sample_labels = ys
        sample_forward_kwargs = {}
        model_fn = ema.forward

    logger.info(f"Training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        opt.zero_grad(set_to_none=True)
        accum_counter = 0
        step_loss_accum = 0.0
        epoch_step = 0
        device_loader = pl.ParallelLoader(host_loader, [device]).per_device_loader(device)
        for x, y in device_loader:
            x = x.to(device)
            y = y.to(device)
            model_kwargs = dict(y=y)
            with autocast(**autocast_kwargs):
                with torch.no_grad():
                    x = rae.encode(x)
                loss_tensor = transport.training_losses(model, x, model_kwargs)["loss"].mean()
            step_loss_accum += loss_tensor.item()
            (loss_tensor / grad_accum_steps).backward()
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            grad_norm = None
            if clip_grad > 0:
                grad_norm = clip_grad_norm_(model.parameters(), clip_grad)
            xm.optimizer_step(opt, barrier=True)
            schedl.step()
            update_ema(ema, model, decay=ema_decay)
            opt.zero_grad(set_to_none=True)

            running_loss += step_loss_accum / grad_accum_steps
            if grad_norm is not None:
                running_grad_norm += float(grad_norm)
            log_steps += 1
            train_steps += 1
            epoch_step += 1
            accum_counter = 0
            step_loss_accum = 0.0

            if log_every > 0 and train_steps % log_every == 0 and log_steps > 0:
                end_time = time()
                optimizer_steps_per_sec = log_steps / max(end_time - start_time, 1e-6)
                images_per_sec = optimizer_steps_per_sec * global_batch_size
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = xm.mesh_reduce("train_loss", avg_loss, lambda x: sum(x) / len(x))
                avg_grad_norm = None
                if clip_grad > 0:
                    grad_norm_tensor = torch.tensor(running_grad_norm / log_steps, device=device)
                    avg_grad_norm = xm.mesh_reduce("train_grad_norm", grad_norm_tensor, lambda x: sum(x) / len(x))
                if is_master:
                    log_stats = {
                        "train/loss": avg_loss.item(),
                        "train/lr": opt.param_groups[0]["lr"],
                        "train/optimizer_steps_per_sec": optimizer_steps_per_sec,
                        "train/images_per_sec": images_per_sec,
                        "train/epoch": epoch + (epoch_step / steps_per_epoch),
                    }
                    if avg_grad_norm is not None:
                        log_stats["train/grad_norm"] = avg_grad_norm.item()
                    logger.info(
                        f"(step={train_steps:07d}) "
                        f"train/loss={log_stats['train/loss']:.4f}, "
                        f"train/lr={log_stats['train/lr']:.6f}, "
                        f"train/optimizer_steps_per_sec={log_stats['train/optimizer_steps_per_sec']:.2f}, "
                        f"train/images_per_sec={log_stats['train/images_per_sec']:.2f}"
                    )
                    if args.wandb:
                        wandb_utils.log(log_stats, step=train_steps)
                running_loss = 0.0
                running_grad_norm = 0.0
                log_steps = 0
                start_time = time()
                xm.rendezvous(f"log_{train_steps}")
            if ckpt_every > 0 and train_steps % ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": schedl.state_dict(),
                    "train_steps": train_steps,
                    "epoch": epoch,
                    "config_path": args.config,
                    "training_cfg": training_cfg,
                    "cli_overrides": {
                        "data_path": args.data_path,
                        "results_dir": args.results_dir,
                        "image_size": args.image_size,
                        "precision": args.precision,
                        "global_seed": global_seed,
                    },
                }
                if checkpoint_dir is not None:
                    checkpoint_path = os.path.join(checkpoint_dir, f"{train_steps:07d}.pt")
                    xm.save(checkpoint, checkpoint_path)
                    if is_master:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                        if args.wandb:
                            wandb_utils.log({"checkpoint/last_saved_step": train_steps}, step=train_steps)
                xm.rendezvous(f"checkpoint_{train_steps}")

            if eval_enabled and eval_loader is not None and eval_index_weights is not None:
                if train_steps % eval_every == 0 and train_steps > 0:
                    xm.rendezvous(f"eval_start_{train_steps}")
                    logger.info("Running validation loss evaluation...")
                    eval_started_at = time()

                    ema_loss, eval_batches = run_validation_loss(
                        ema,
                        rae,
                        transport,
                        eval_loader,
                        device,
                        autocast_kwargs,
                        eval_index_weights,
                        reduce_prefix="eval_ema",
                        max_batches=eval_max_batches,
                    )
                    eval_stats: Dict[str, Any] = {
                        "eval/ema_loss": ema_loss,
                        "eval/duration_sec": time() - eval_started_at,
                        "eval/num_batches": eval_batches,
                    }

                    if eval_model:
                        model_was_training = model.training
                        model.eval()
                        model_loss, _ = run_validation_loss(
                            model,
                            rae,
                            transport,
                            eval_loader,
                            device,
                            autocast_kwargs,
                            eval_index_weights,
                            reduce_prefix="eval_model",
                            max_batches=eval_max_batches,
                        )
                        eval_stats["eval/model_loss"] = model_loss
                        if model_was_training:
                            model.train()

                    if is_master:
                        logger.info(
                            f"(step={train_steps:07d}) "
                            f"eval/ema_loss={eval_stats['eval/ema_loss']:.4f}, "
                            f"eval/num_batches={eval_stats['eval/num_batches']}, "
                            f"eval/duration_sec={eval_stats['eval/duration_sec']:.2f}"
                            + (
                                f", eval/model_loss={eval_stats['eval/model_loss']:.4f}"
                                if "eval/model_loss" in eval_stats
                                else ""
                            )
                        )
                        if args.wandb:
                            wandb_utils.log(eval_stats, step=train_steps)
                    xm.rendezvous(f"eval_done_{train_steps}")

            if sample_every > 0 and (train_steps % sample_every == 0 or train_steps == 1):
                logger.info("Generating EMA samples...")
                sample_started_at = time()
                with torch.no_grad():
                    with autocast(**autocast_kwargs):
                        samples = sample_latents(
                            model_fn,
                            zs,
                            sample_labels,
                            model_kwargs=sample_forward_kwargs,
                        )
                        if using_cfg:
                            samples, _ = samples.chunk(2, dim=0)
                        samples = rae.decode(samples.to(torch.float32))
                gathered_samples = xm.all_gather(samples, dim=0)
                if args.wandb and is_master:
                    wandb_utils.log({"sample/duration_sec": time() - sample_started_at}, step=train_steps)
                    wandb_utils.log_image(
                        xm._maybe_convert_to_cpu(gathered_samples),
                        step=train_steps,
                        key="samples/ema",
                    )
                logger.info("Generating EMA samples done.")
                xm.rendezvous(f"sample_{train_steps}")
            xm.mark_step()
        logger.info(f"Completed epoch {epoch}.")
        del device_loader

    model.eval()
    ema.eval()
    logger.info("Training complete.")
    cleanup()


def _mp_worker(rank: int, args: argparse.Namespace) -> None:
    del rank  # Unused, but required by xmp.spawn signature.
    main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training dataset root.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Compute precision.")
    parser.add_argument("--global-seed", type=int, default=None, help="Optional seed to override config.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    args = parser.parse_args()
    xmp.spawn(_mp_worker, args=(args,), start_method="fork")
