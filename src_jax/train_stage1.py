from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

from flax import jax_utils
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .stage1_checkpoint import (
        load_stage1_checkpoint,
        replicate_tree,
        restore_decoder_state,
        restore_optimizer_state,
        save_stage1_checkpoint,
        unreplicate_tree,
    )
    from .stage1_components import (
        DinoDiscriminatorHeads,
        build_stage1_decoder,
        build_stage1_encoder,
        compute_adaptive_weight,
        decode_latents_to_nhwc,
        diffaug,
        dino_backbone_forward,
        encode_images,
        hinge_d_loss,
        load_dino_backbone_weights,
        load_lpips_weights,
        lpips_loss,
        resolve_stage1_paths,
        stack_real_and_reconstruction_grid,
        tree_l2_norm,
        vanilla_g_loss,
    )
    from .stage1_jax_config import load_stage1_jax_runtime_config, runtime_config_to_dict
    from .stage1_jax_state import build_adamw_transform, build_learning_rate_schedule
    from .tfds_data import build_numpy_tfds_loader
except ImportError:
    from stage1_checkpoint import (
        load_stage1_checkpoint,
        replicate_tree,
        restore_decoder_state,
        restore_optimizer_state,
        save_stage1_checkpoint,
        unreplicate_tree,
    )
    from stage1_components import (
        DinoDiscriminatorHeads,
        build_stage1_decoder,
        build_stage1_encoder,
        compute_adaptive_weight,
        decode_latents_to_nhwc,
        diffaug,
        dino_backbone_forward,
        encode_images,
        hinge_d_loss,
        load_dino_backbone_weights,
        load_lpips_weights,
        lpips_loss,
        resolve_stage1_paths,
        stack_real_and_reconstruction_grid,
        tree_l2_norm,
        vanilla_g_loss,
    )
    from stage1_jax_config import load_stage1_jax_runtime_config, runtime_config_to_dict
    from stage1_jax_state import build_adamw_transform, build_learning_rate_schedule
    from tfds_data import build_numpy_tfds_loader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage-1 RAE with a real all-JAX reconstruction + LPIPS + GAN loop.")
    parser.add_argument("--config", required=True, help="Path to the Stage-1 repo YAML config.")
    parser.add_argument("--data-path", default=None, help="Optional TFDS/ImageFolder root override.")
    parser.add_argument("--data-format", choices=["imagefolder", "tfds"], default=None)
    parser.add_argument("--dataset-name", default=None, help="TFDS dataset name override, e.g. celebahq256.")
    parser.add_argument("--train-split", default=None, help="Optional TFDS train split override.")
    parser.add_argument("--eval-split", default=None, help="Optional TFDS eval split override.")
    parser.add_argument("--results-dir", default=None, help="Override experiment output directory.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backend-dir", default=None, help="Optional diffuse_nnx checkout override.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=None, help="Override WANDB project.")
    parser.add_argument("--wandb-entity", default=None, help="Override WANDB entity.")
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="OmegaConf override.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved Stage-1 JAX runtime config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config wiring and exit successfully.")
    return parser


def _create_logger(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("rae-stage1-jax")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "log.txt")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _maybe_init_wandb(args: argparse.Namespace, runtime_cfg: Any) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("wandb is required when --wandb is set.") from exc

    if "WANDB_API_KEY" not in os.environ and "WANDB_KEY" in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])

    return wandb.init(
        entity=args.wandb_entity or os.environ.get("WANDB_ENTITY"),
        project=args.wandb_project or os.environ.get("WANDB_PROJECT") or "RAE-stage1-jax",
        name=runtime_cfg.experiment_name,
        config=runtime_config_to_dict(runtime_cfg),
        resume="allow",
        reinit=True,
    )


def _wandb_log(run: Any | None, payload: dict[str, Any], *, step: int) -> None:
    if run is not None:
        run.log(payload, step=step)


def _wandb_log_image(run: Any | None, image: np.ndarray, *, step: int, key: str) -> None:
    if run is None:
        return
    import wandb

    run.log({key: wandb.Image(image)}, step=step)


def _make_train_loader(runtime_cfg: Any, *, epoch: int, global_batch_size: int):
    if runtime_cfg.data.source != "tfds":
        raise NotImplementedError("Stage-1 all-JAX trainer currently supports TFDS datasets only.")
    return build_numpy_tfds_loader(
        dataset_name=runtime_cfg.data.dataset_name,
        data_dir=runtime_cfg.data.data_dir,
        split=runtime_cfg.data.train_split,
        batch_size=global_batch_size,
        image_size=runtime_cfg.data.image_size,
        random_flip=runtime_cfg.data.random_flip,
        repeat=False,
        shuffle=True,
        seed=runtime_cfg.training.seed + epoch,
        rank=0,
        world_size=1,
        shuffle_buffer=runtime_cfg.data.shuffle_buffer,
        drop_remainder=True,
        channels_first=False,
        pixel_range="zero_one",
    )


def _make_eval_loader(runtime_cfg: Any, *, global_batch_size: int):
    if runtime_cfg.data.source != "tfds":
        raise NotImplementedError("Stage-1 all-JAX trainer currently supports TFDS datasets only.")
    eval_split = runtime_cfg.data.eval_split or runtime_cfg.data.train_split
    return build_numpy_tfds_loader(
        dataset_name=runtime_cfg.data.dataset_name,
        data_dir=runtime_cfg.data.data_dir,
        split=eval_split,
        batch_size=global_batch_size,
        image_size=runtime_cfg.data.image_size,
        random_flip=False,
        repeat=False,
        shuffle=False,
        seed=runtime_cfg.training.seed,
        rank=0,
        world_size=1,
        shuffle_buffer=runtime_cfg.data.shuffle_buffer,
        drop_remainder=True,
        channels_first=False,
        pixel_range="zero_one",
    )


def _reshape_for_pmap(images: np.ndarray, local_device_count: int, per_device_batch: int) -> np.ndarray:
    if images.shape[0] != local_device_count * per_device_batch:
        raise ValueError(f"Expected batch {local_device_count * per_device_batch}, got {images.shape[0]}")
    return images.reshape(local_device_count, per_device_batch, *images.shape[1:])


def _tree_item(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: float(np.asarray(x[0])) if np.asarray(x).shape else float(np.asarray(x)), tree)


def _build_pmapped_steps(
    *,
    encoder_graph: Any,
    decoder_opt_graph: Any,
    disc_opt_graph: Any,
    ema_graph: Any,
    encoder_spec: Any,
    ema_decay: float,
    perceptual_weight: float,
    disc_weight: float,
    max_d_weight: float,
    aug_prob: float,
    aug_cutout: float,
):
    @jax.pmap(axis_name="data")
    def generator_step(
        decoder_opt_state: Any,
        disc_opt_state: Any,
        ema_state: Any,
        encoder_state: Any,
        lpips_weights: Any,
        dino_backbone_weights: Any,
        batch_images: Array,
        rng_key: Array,
        use_lpips: Array,
        use_gan: Array,
    ):
        decoder_opt = nnx.merge(decoder_opt_graph, decoder_opt_state)
        disc_opt = nnx.merge(disc_opt_graph, disc_opt_state)
        ema_decoder = nnx.merge(ema_graph, ema_state)
        encoder = nnx.merge(encoder_graph, encoder_state)
        disc_heads = disc_opt.model

        key_noise, key_aug_recon, key_aug_gan = jax.random.split(rng_key, 3)
        use_lpips = jnp.asarray(use_lpips, dtype=bool)
        use_gan = jnp.asarray(use_gan, dtype=bool)

        def compute_terms(decoder_model: Any) -> tuple[Array, Array, dict[str, Array]]:
            latents = encode_images(encoder, batch_images, encoder_spec, train=True, noise_key=key_noise)
            latents = jax.lax.stop_gradient(latents)
            recon_nhwc = decode_latents_to_nhwc(decoder_model, latents, encoder_spec)
            rec_loss = jnp.mean(jnp.abs(recon_nhwc - batch_images))
            real_pm1 = batch_images * 2.0 - 1.0
            recon_pm1 = recon_nhwc * 2.0 - 1.0

            lpips_term = jax.lax.cond(
                use_lpips,
                lambda _: lpips_loss(lpips_weights, real_pm1, recon_pm1),
                lambda _: jnp.asarray(0.0, dtype=jnp.float32),
                operand=None,
            )
            recon_total = rec_loss + perceptual_weight * lpips_term

            def _gan_value(_: None) -> Array:
                fake_aug = diffaug(recon_pm1, key=key_aug_gan, prob=aug_prob, cutout=aug_cutout)
                logits_fake = disc_heads(dino_backbone_forward(dino_backbone_weights, fake_aug))
                return vanilla_g_loss(logits_fake)

            gan_term = jax.lax.cond(
                use_gan,
                _gan_value,
                lambda _: jnp.asarray(0.0, dtype=jnp.float32),
                operand=None,
            )
            aux = {
                "rec_loss": rec_loss,
                "lpips_loss": lpips_term,
                "gan_loss": gan_term,
            }
            return recon_total, gan_term, aux

        if disc_weight > 0.0:
            recon_grads = nnx.grad(lambda model: compute_terms(model)[0])(decoder_opt.model)
            recon_grads = jax.lax.pmean(recon_grads, axis_name="data")
            gan_grads = nnx.grad(lambda model: compute_terms(model)[1])(decoder_opt.model)
            gan_grads = jax.lax.pmean(gan_grads, axis_name="data")
            adaptive_weight = jax.lax.cond(
                use_gan,
                lambda _: compute_adaptive_weight(recon_grads, gan_grads, max_d_weight),
                lambda _: jnp.asarray(0.0, dtype=jnp.float32),
                operand=None,
            )
        else:
            adaptive_weight = jnp.asarray(0.0, dtype=jnp.float32)

        def total_loss_fn(decoder_model: Any):
            recon_total, gan_term, aux = compute_terms(decoder_model)
            total = recon_total + disc_weight * jax.lax.stop_gradient(adaptive_weight) * gan_term
            aux = dict(aux)
            aux["adaptive_weight"] = adaptive_weight
            aux["total_loss"] = total
            return total, aux

        (loss, aux), grads = nnx.value_and_grad(total_loss_fn, has_aux=True)(decoder_opt.model)
        grads = jax.lax.pmean(grads, axis_name="data")
        decoder_opt.update(grads)

        grad_norm = tree_l2_norm(grads)
        current_params = nnx.state(decoder_opt.model, nnx.Param)
        ema_params = nnx.state(ema_decoder, nnx.Param)
        ema_params = jax.tree_util.tree_map(
            lambda ema_value, value: ema_value * ema_decay + value * (1.0 - ema_decay),
            ema_params,
            current_params,
        )
        nnx.update(ema_decoder, ema_params)

        _, new_decoder_opt_state = nnx.split(decoder_opt)
        _, new_ema_state = nnx.split(ema_decoder)
        metrics = {
            "loss_total": jax.lax.pmean(aux["total_loss"], axis_name="data"),
            "loss_recon": jax.lax.pmean(aux["rec_loss"], axis_name="data"),
            "loss_lpips": jax.lax.pmean(aux["lpips_loss"], axis_name="data"),
            "loss_gan": jax.lax.pmean(aux["gan_loss"], axis_name="data"),
            "adaptive_weight": jax.lax.pmean(aux["adaptive_weight"], axis_name="data"),
            "grad_norm": jax.lax.pmean(grad_norm, axis_name="data"),
        }
        return new_decoder_opt_state, new_ema_state, metrics

    @jax.pmap(axis_name="data")
    def discriminator_step(
        decoder_opt_state: Any,
        disc_opt_state: Any,
        encoder_state: Any,
        dino_backbone_weights: Any,
        batch_images: Array,
        rng_key: Array,
    ):
        decoder_opt = nnx.merge(decoder_opt_graph, decoder_opt_state)
        disc_opt = nnx.merge(disc_opt_graph, disc_opt_state)
        encoder = nnx.merge(encoder_graph, encoder_state)
        key_real, key_fake = jax.random.split(rng_key)

        def disc_loss_fn(disc_heads_model: Any):
            latents = encode_images(encoder, batch_images, encoder_spec, train=False, noise_key=None)
            latents = jax.lax.stop_gradient(latents)
            recon_nhwc = decode_latents_to_nhwc(decoder_opt.model, latents, encoder_spec)
            real_pm1 = batch_images * 2.0 - 1.0
            fake_pm1 = jax.lax.stop_gradient(recon_nhwc * 2.0 - 1.0)
            fake_pm1 = jnp.round((jnp.clip(fake_pm1, -1.0, 1.0) + 1.0) * 127.5) / 127.5 - 1.0

            fake_aug = diffaug(fake_pm1, key=key_fake, prob=aug_prob, cutout=aug_cutout)
            real_aug = diffaug(real_pm1, key=key_real, prob=aug_prob, cutout=aug_cutout)
            logits_fake = disc_heads_model(dino_backbone_forward(dino_backbone_weights, fake_aug))
            logits_real = disc_heads_model(dino_backbone_forward(dino_backbone_weights, real_aug))
            loss = hinge_d_loss(logits_real, logits_fake)
            aux = {
                "logits_real": jnp.mean(logits_real),
                "logits_fake": jnp.mean(logits_fake),
                "disc_accuracy": jnp.mean((logits_real > logits_fake).astype(jnp.float32)),
            }
            return loss, aux

        (loss, aux), grads = nnx.value_and_grad(disc_loss_fn, has_aux=True)(disc_opt.model)
        grads = jax.lax.pmean(grads, axis_name="data")
        disc_opt.update(grads)
        _, new_disc_opt_state = nnx.split(disc_opt)
        metrics = {
            "loss_disc": jax.lax.pmean(loss, axis_name="data"),
            "disc_accuracy": jax.lax.pmean(aux["disc_accuracy"], axis_name="data"),
            "logits_real": jax.lax.pmean(aux["logits_real"], axis_name="data"),
            "logits_fake": jax.lax.pmean(aux["logits_fake"], axis_name="data"),
        }
        return new_disc_opt_state, metrics

    @jax.pmap(axis_name="data")
    def eval_step(
        decoder_state: Any,
        encoder_state: Any,
        lpips_weights: Any,
        batch_images: Array,
    ):
        decoder = nnx.merge(ema_graph, decoder_state)
        encoder = nnx.merge(encoder_graph, encoder_state)
        latents = encode_images(encoder, batch_images, encoder_spec, train=False, noise_key=None)
        recon_nhwc = decode_latents_to_nhwc(decoder, latents, encoder_spec)
        real_pm1 = batch_images * 2.0 - 1.0
        recon_pm1 = recon_nhwc * 2.0 - 1.0
        rec_loss = jnp.mean(jnp.abs(recon_nhwc - batch_images))
        lpips_term = lpips_loss(lpips_weights, real_pm1, recon_pm1)
        total = rec_loss + perceptual_weight * lpips_term
        return {
            "loss_recon": jax.lax.pmean(rec_loss, axis_name="data"),
            "loss_lpips": jax.lax.pmean(lpips_term, axis_name="data"),
            "loss_total": jax.lax.pmean(total, axis_name="data"),
        }

    return generator_step, discriminator_step, eval_step


def _build_sample_grid(
    *,
    ema_graph: Any,
    ema_state: Any,
    encoder_graph: Any,
    encoder_state: Any,
    encoder_spec: Any,
    sample_images: np.ndarray,
) -> np.ndarray:
    decoder = nnx.merge(ema_graph, ema_state)
    encoder = nnx.merge(encoder_graph, encoder_state)
    recon = decode_latents_to_nhwc(
        decoder,
        encode_images(encoder, jnp.asarray(sample_images), encoder_spec, train=False, noise_key=None),
        encoder_spec,
    )
    return stack_real_and_reconstruction_grid(
        np.asarray(sample_images, dtype=np.float32),
        np.asarray(recon, dtype=np.float32),
    )


def _run_eval_loop(
    *,
    runtime_cfg: Any,
    global_batch_size: int,
    local_device_count: int,
    per_device_batch: int,
    decoder_state: Any,
    encoder_state: Any,
    lpips_weights: Any,
    eval_step_fn: Any,
    max_batches: int,
) -> dict[str, float]:
    loader = _make_eval_loader(runtime_cfg, global_batch_size=global_batch_size)
    totals = {"loss_recon": 0.0, "loss_lpips": 0.0, "loss_total": 0.0}
    batches = 0
    for batch_idx, (images, _labels) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        if images.shape[0] != global_batch_size:
            continue
        batch = _reshape_for_pmap(np.asarray(images, dtype=np.float32), local_device_count, per_device_batch)
        metrics = _tree_item(eval_step_fn(decoder_state, encoder_state, lpips_weights, batch))
        for key in totals:
            totals[key] += float(metrics[key])
        batches += 1
    if batches == 0:
        return {f"eval/{key}": 0.0 for key in totals}
    return {f"eval/{key}": value / batches for key, value in totals.items()}


def _checkpoint_bundle(
    *,
    decoder_opt_graph: Any,
    decoder_opt_state: Any,
    disc_opt_graph: Any,
    disc_opt_state: Any,
    ema_graph: Any,
    ema_state: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    decoder_opt = nnx.merge(decoder_opt_graph, unreplicate_tree(decoder_opt_state))
    disc_opt = nnx.merge(disc_opt_graph, unreplicate_tree(disc_opt_state))
    ema_decoder = nnx.merge(ema_graph, unreplicate_tree(ema_state))
    return (
        nnx.state(decoder_opt.model),
        nnx.split(decoder_opt)[1],
        nnx.state(ema_decoder),
        nnx.state(disc_opt.model),
        nnx.split(disc_opt)[1],
    )


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
        results_dir=args.results_dir,
        precision=args.precision,
        seed=args.seed,
    )

    if args.print_config:
        print(json.dumps(runtime_config_to_dict(runtime_cfg), indent=2))

    if args.dry_run:
        print("Stage-1 all-JAX dry-run succeeded. Config wiring, dataset wiring, and runtime contract are valid.")
        return

    local_device_count = jax.local_device_count()
    if runtime_cfg.training.global_batch_size % local_device_count != 0:
        raise ValueError(
            f"training.global_batch_size={runtime_cfg.training.global_batch_size} must be divisible by local_device_count={local_device_count}"
        )
    per_device_batch = runtime_cfg.training.global_batch_size // local_device_count
    model_dtype = jnp.bfloat16 if runtime_cfg.training.precision == "bf16" else jnp.float32
    results_dir = runtime_cfg.results_dir
    checkpoint_dir = results_dir / "checkpoints"
    logger = _create_logger(results_dir)
    wandb_run = _maybe_init_wandb(args, runtime_cfg)

    logger.info("Stage-1 JAX workdir: %s", results_dir)
    logger.info("Devices: %s | Global batch: %s | Per-device batch: %s", local_device_count, runtime_cfg.training.global_batch_size, per_device_batch)

    paths = resolve_stage1_paths(
        config_path=cfg_path,
        stage1_params=runtime_cfg.stage1_params,
        dino_disc_ckpt_path=runtime_cfg.discriminator.arch.dino_ckpt_path,
    )

    encoder, encoder_spec = build_stage1_encoder(
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
    ema_decoder = copy.deepcopy(decoder)
    dino_backbone_weights = load_dino_backbone_weights(
        paths.dino_disc_ckpt_path,
        recipe=runtime_cfg.discriminator.arch.recipe,
        dtype=jnp.float32,
    )
    lpips_weights = load_lpips_weights(dtype=jnp.float32)
    disc_heads = DinoDiscriminatorHeads(
        embed_dim=dino_backbone_weights.embed_dim,
        num_heads=len(dino_backbone_weights.key_depths) + 1,
        ks=runtime_cfg.discriminator.arch.ks,
        norm_type=runtime_cfg.discriminator.arch.norm_type,
        using_spec_norm=runtime_cfg.discriminator.arch.using_spec_norm,
        dtype=model_dtype,
        rngs=nnx.Rngs(runtime_cfg.training.seed + 1),
    )

    train_loader = _make_train_loader(runtime_cfg, epoch=0, global_batch_size=runtime_cfg.training.global_batch_size)
    steps_per_epoch = len(train_loader)
    logger.info("Steps per epoch: %s", steps_per_epoch)

    gen_lr = build_learning_rate_schedule(
        runtime_cfg.training.scheduler,
        steps_per_epoch=steps_per_epoch,
        total_epochs=runtime_cfg.training.epochs,
    )
    disc_lr = build_learning_rate_schedule(
        runtime_cfg.discriminator.scheduler,
        steps_per_epoch=steps_per_epoch,
        total_epochs=runtime_cfg.training.epochs,
    )
    gen_tx = build_adamw_transform(
        runtime_cfg.training.optimizer,
        learning_rate=gen_lr,
        clip_grad=runtime_cfg.training.clip_grad,
    )
    disc_tx = build_adamw_transform(
        runtime_cfg.discriminator.optimizer,
        learning_rate=disc_lr,
        clip_grad=0.0,
    )
    decoder_opt = nnx.Optimizer(model=decoder, tx=gen_tx)
    disc_opt = nnx.Optimizer(model=disc_heads, tx=disc_tx)

    encoder_graph, encoder_state = nnx.split(encoder)
    decoder_opt_graph, decoder_opt_state_host = nnx.split(decoder_opt)
    disc_opt_graph, disc_opt_state_host = nnx.split(disc_opt)
    ema_graph, ema_state_host = nnx.split(ema_decoder)

    start_epoch = 0
    global_step = 0
    rng = jax.random.PRNGKey(runtime_cfg.training.seed)
    resume_path = checkpoint_dir / "ep-last.pt"
    if resume_path.exists():
        checkpoint = load_stage1_checkpoint(resume_path)
        decoder_opt_state_host = restore_optimizer_state(decoder_opt_state_host, checkpoint, key="decoder_opt_state")
        disc_opt_state_host = restore_optimizer_state(disc_opt_state_host, checkpoint, key="disc_opt_state")
        ema_state_host = restore_decoder_state(ema_state_host, checkpoint, use_ema=True)
        global_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", global_step // max(steps_per_epoch, 1)))
        if "rng_state" in checkpoint:
            rng = jax.random.wrap_key_data(jnp.asarray(checkpoint["rng_state"]))
        logger.info("Resumed from %s (epoch=%s, step=%s)", resume_path, start_epoch, global_step)

    encoder_state = replicate_tree(encoder_state)
    decoder_opt_state = replicate_tree(decoder_opt_state_host)
    disc_opt_state = replicate_tree(disc_opt_state_host)
    ema_state = replicate_tree(ema_state_host)
    lpips_weights = replicate_tree(lpips_weights)
    dino_backbone_weights = replicate_tree(dino_backbone_weights)

    generator_step, discriminator_step, eval_step = _build_pmapped_steps(
        encoder_graph=encoder_graph,
        decoder_opt_graph=decoder_opt_graph,
        disc_opt_graph=disc_opt_graph,
        ema_graph=ema_graph,
        encoder_spec=encoder_spec,
        ema_decay=runtime_cfg.training.ema_decay,
        perceptual_weight=runtime_cfg.gan_loss.perceptual_weight,
        disc_weight=runtime_cfg.gan_loss.disc_weight,
        max_d_weight=runtime_cfg.gan_loss.max_d_weight,
        aug_prob=runtime_cfg.discriminator.augment_prob,
        aug_cutout=runtime_cfg.discriminator.augment_cutout,
    )

    gan_start_step = runtime_cfg.gan_loss.disc_start_epoch * steps_per_epoch
    disc_update_step = runtime_cfg.gan_loss.disc_update_start_epoch * steps_per_epoch
    lpips_start_step = runtime_cfg.gan_loss.lpips_start_epoch * steps_per_epoch
    eval_enabled = bool(runtime_cfg.data.eval_split)
    eval_every = runtime_cfg.evaluation.eval_every if runtime_cfg.evaluation.eval_every > 0 else runtime_cfg.training.sample_every
    eval_max_batches = runtime_cfg.evaluation.max_batches

    for epoch in range(start_epoch, runtime_cfg.training.epochs):
        epoch_sums = {
            "loss_total": 0.0,
            "loss_recon": 0.0,
            "loss_lpips": 0.0,
            "loss_gan": 0.0,
            "loss_disc": 0.0,
            "disc_accuracy": 0.0,
        }
        epoch_batches = 0
        current_loader = _make_train_loader(runtime_cfg, epoch=epoch, global_batch_size=runtime_cfg.training.global_batch_size)

        if runtime_cfg.training.checkpoint_interval > 0 and epoch % runtime_cfg.training.checkpoint_interval == 0:
            decoder_state_host, decoder_opt_host, ema_state_ckpt, disc_state_host, disc_opt_host = _checkpoint_bundle(
                decoder_opt_graph=decoder_opt_graph,
                decoder_opt_state=decoder_opt_state,
                disc_opt_graph=disc_opt_graph,
                disc_opt_state=disc_opt_state,
                ema_graph=ema_graph,
                ema_state=ema_state,
            )
            save_stage1_checkpoint(
                checkpoint_dir / f"ep-{epoch:07d}.pt",
                epoch=epoch,
                step=global_step,
                decoder_state=decoder_state_host,
                decoder_opt_state=decoder_opt_host,
                ema_decoder_state=ema_state_ckpt,
                disc_state=disc_state_host,
                disc_opt_state=disc_opt_host,
                rng_state=rng,
                metadata={"config_path": str(cfg_path)},
            )

        for images, _labels in current_loader:
            if images.shape[0] != runtime_cfg.training.global_batch_size:
                continue
            batch = _reshape_for_pmap(np.asarray(images, dtype=np.float32), local_device_count, per_device_batch)
            use_lpips = runtime_cfg.gan_loss.perceptual_weight > 0.0 and global_step >= lpips_start_step
            use_gan = runtime_cfg.gan_loss.disc_weight > 0.0 and global_step >= gan_start_step
            train_disc = runtime_cfg.gan_loss.disc_weight > 0.0 and global_step >= disc_update_step

            rng, gen_rng = jax.random.split(rng)
            gen_keys = np.asarray(jax.random.split(gen_rng, local_device_count))
            use_lpips_flags = np.full((local_device_count,), use_lpips, dtype=bool)
            use_gan_flags = np.full((local_device_count,), use_gan, dtype=bool)

            start_t = time.time()
            decoder_opt_state, ema_state, gen_metrics = generator_step(
                decoder_opt_state,
                disc_opt_state,
                ema_state,
                encoder_state,
                lpips_weights,
                dino_backbone_weights,
                batch,
                gen_keys,
                use_lpips_flags,
                use_gan_flags,
            )
            step_metrics = _tree_item(gen_metrics)
            step_metrics["lr_generator"] = float(np.asarray(gen_lr(global_step)))
            step_metrics["steps_per_second"] = 1.0 / max(time.time() - start_t, 1e-6)

            if train_disc:
                for _ in range(runtime_cfg.gan_loss.disc_updates):
                    rng, disc_rng = jax.random.split(rng)
                    disc_keys = np.asarray(jax.random.split(disc_rng, local_device_count))
                    disc_opt_state, disc_metrics = discriminator_step(
                        decoder_opt_state,
                        disc_opt_state,
                        encoder_state,
                        dino_backbone_weights,
                        batch,
                        disc_keys,
                    )
                disc_metrics = _tree_item(disc_metrics)
                step_metrics.update(disc_metrics)
                step_metrics["lr_discriminator"] = float(np.asarray(disc_lr(global_step)))

            epoch_sums["loss_total"] += step_metrics["loss_total"]
            epoch_sums["loss_recon"] += step_metrics["loss_recon"]
            epoch_sums["loss_lpips"] += step_metrics["loss_lpips"]
            epoch_sums["loss_gan"] += step_metrics["loss_gan"]
            epoch_sums["loss_disc"] += step_metrics.get("loss_disc", 0.0)
            epoch_sums["disc_accuracy"] += step_metrics.get("disc_accuracy", 0.0)
            epoch_batches += 1

            if runtime_cfg.training.log_every > 0 and global_step % runtime_cfg.training.log_every == 0:
                logger.info(
                    "[Epoch %s | Step %s] total=%.4f recon=%.4f lpips=%.4f gan=%.4f grad=%.4f",
                    epoch,
                    global_step,
                    step_metrics["loss_total"],
                    step_metrics["loss_recon"],
                    step_metrics["loss_lpips"],
                    step_metrics["loss_gan"],
                    step_metrics["grad_norm"],
                )
                payload = {
                    "loss/total": step_metrics["loss_total"],
                    "loss/recon": step_metrics["loss_recon"],
                    "loss/lpips": step_metrics["loss_lpips"],
                    "loss/gan": step_metrics["loss_gan"],
                    "disc/weight": step_metrics["adaptive_weight"],
                    "grad_norm": step_metrics["grad_norm"],
                    "lr/generator": step_metrics["lr_generator"],
                    "perf/steps_per_second": step_metrics["steps_per_second"],
                }
                if "loss_disc" in step_metrics:
                    payload.update(
                        {
                            "loss/disc": step_metrics["loss_disc"],
                            "disc/accuracy": step_metrics["disc_accuracy"],
                            "disc/logits_real": step_metrics["logits_real"],
                            "disc/logits_fake": step_metrics["logits_fake"],
                            "lr/discriminator": step_metrics["lr_discriminator"],
                        }
                    )
                _wandb_log(wandb_run, payload, step=global_step)

            if runtime_cfg.training.sample_every > 0 and global_step % runtime_cfg.training.sample_every == 0:
                sample_images = np.asarray(images[:4], dtype=np.float32)
                grid = _build_sample_grid(
                    ema_graph=ema_graph,
                    ema_state=unreplicate_tree(ema_state),
                    encoder_graph=encoder_graph,
                    encoder_state=unreplicate_tree(encoder_state),
                    encoder_spec=encoder_spec,
                    sample_images=sample_images,
                )
                _wandb_log_image(wandb_run, grid, step=global_step, key="stage1/reconstructions")

            if eval_enabled and eval_every > 0 and global_step > 0 and global_step % eval_every == 0:
                eval_metrics = _run_eval_loop(
                    runtime_cfg=runtime_cfg,
                    global_batch_size=runtime_cfg.training.global_batch_size,
                    local_device_count=local_device_count,
                    per_device_batch=per_device_batch,
                    decoder_state=ema_state,
                    encoder_state=encoder_state,
                    lpips_weights=lpips_weights,
                    eval_step_fn=eval_step,
                    max_batches=eval_max_batches,
                )
                logger.info(
                    "[Eval | Step %s] recon=%.4f lpips=%.4f total=%.4f",
                    global_step,
                    eval_metrics["eval/loss_recon"],
                    eval_metrics["eval/loss_lpips"],
                    eval_metrics["eval/loss_total"],
                )
                _wandb_log(wandb_run, eval_metrics, step=global_step)

            global_step += 1

        if epoch_batches > 0:
            epoch_payload = {
                "epoch/loss_total": epoch_sums["loss_total"] / epoch_batches,
                "epoch/loss_recon": epoch_sums["loss_recon"] / epoch_batches,
                "epoch/loss_lpips": epoch_sums["loss_lpips"] / epoch_batches,
                "epoch/loss_gan": epoch_sums["loss_gan"] / epoch_batches,
            }
            if epoch_sums["loss_disc"] > 0:
                epoch_payload["epoch/loss_disc"] = epoch_sums["loss_disc"] / epoch_batches
                epoch_payload["epoch/disc_accuracy"] = epoch_sums["disc_accuracy"] / epoch_batches
            logger.info("[Epoch %s] %s", epoch, ", ".join(f"{key}={value:.4f}" for key, value in epoch_payload.items()))
            _wandb_log(wandb_run, epoch_payload, step=global_step)

    decoder_state_host, decoder_opt_host, ema_state_ckpt, disc_state_host, disc_opt_host = _checkpoint_bundle(
        decoder_opt_graph=decoder_opt_graph,
        decoder_opt_state=decoder_opt_state,
        disc_opt_graph=disc_opt_graph,
        disc_opt_state=disc_opt_state,
        ema_graph=ema_graph,
        ema_state=ema_state,
    )
    save_stage1_checkpoint(
        checkpoint_dir / "ep-last.pt",
        epoch=runtime_cfg.training.epochs,
        step=global_step,
        decoder_state=decoder_state_host,
        decoder_opt_state=decoder_opt_host,
        ema_decoder_state=ema_state_ckpt,
        disc_state=disc_state_host,
        disc_opt_state=disc_opt_host,
        rng_state=rng,
        metadata={
            "config_path": str(cfg_path),
            "experiment_name": runtime_cfg.experiment_name,
            "global_batch_size": runtime_cfg.training.global_batch_size,
        },
    )
    logger.info("Saved final Stage-1 JAX checkpoint to %s", checkpoint_dir / "ep-last.pt")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
