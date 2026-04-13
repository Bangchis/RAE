from __future__ import annotations

import math
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
import optax

from .stage1_jax_config import OptimizerConfig, ScheduleConfig


@flax.struct.dataclass
class ModelTrainState:
    step: int
    params: Any
    params_ema: Any
    opt_state: Any


def build_learning_rate_schedule(
    schedule_cfg: ScheduleConfig,
    *,
    steps_per_epoch: int,
    total_epochs: int,
) -> Callable[[int], jax.Array]:
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive.")
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")

    base_lr = float(schedule_cfg.base_lr)
    final_lr = float(schedule_cfg.final_lr)
    warmup_steps = max(0, int(schedule_cfg.warmup_epochs) * steps_per_epoch)
    decay_end_steps = max(warmup_steps, int(schedule_cfg.decay_end_epoch) * steps_per_epoch)
    total_steps = max(decay_end_steps, total_epochs * steps_per_epoch)
    init_lr = 0.0 if schedule_cfg.warmup_from_zero else base_lr
    schedule_type = schedule_cfg.type.strip().lower()

    def _linear_interp(step_value: jax.Array, start: float, end: float, transition_steps: int) -> jax.Array:
        if transition_steps <= 0:
            return jnp.asarray(end, dtype=jnp.float32)
        progress = jnp.clip(step_value / float(transition_steps), 0.0, 1.0)
        return jnp.asarray(start + (end - start) * progress, dtype=jnp.float32)

    def _schedule(step: int) -> jax.Array:
        step_value = jnp.asarray(step, dtype=jnp.float32)
        warmed = _linear_interp(step_value, init_lr, base_lr, warmup_steps) if warmup_steps > 0 else jnp.asarray(base_lr, dtype=jnp.float32)

        post_warmup_step = jnp.maximum(step_value - float(warmup_steps), 0.0)
        post_warmup_total = max(1, decay_end_steps - warmup_steps)
        decay_progress = jnp.clip(post_warmup_step / float(post_warmup_total), 0.0, 1.0)

        if schedule_type == "constant":
            value = jnp.asarray(base_lr, dtype=jnp.float32)
        elif schedule_type == "linear":
            value = jnp.asarray(base_lr + (final_lr - base_lr) * decay_progress, dtype=jnp.float32)
        elif schedule_type == "cosine":
            cosine = 0.5 * (1.0 + jnp.cos(jnp.asarray(math.pi, dtype=jnp.float32) * decay_progress))
            value = jnp.asarray(final_lr + (base_lr - final_lr) * cosine, dtype=jnp.float32)
        else:
            value = jnp.asarray(base_lr, dtype=jnp.float32)

        value = jnp.where(step_value < float(warmup_steps), warmed, value)
        tail_value = jnp.asarray(final_lr if schedule_type != "constant" else base_lr, dtype=jnp.float32)
        return jnp.where(step_value >= float(total_steps), tail_value, value)

    return _schedule


def build_adamw_transform(
    optimizer_cfg: OptimizerConfig,
    *,
    learning_rate: float | Callable[[int], jax.Array],
    clip_grad: float = 0.0,
) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    if clip_grad > 0:
        transforms.append(optax.clip_by_global_norm(float(clip_grad)))
    transforms.append(
        optax.adamw(
            learning_rate=learning_rate,
            b1=float(optimizer_cfg.beta1),
            b2=float(optimizer_cfg.beta2),
            weight_decay=float(optimizer_cfg.weight_decay),
        )
    )
    return optax.chain(*transforms)


def update_ema_params(ema_params: Any, params: Any, decay: float) -> Any:
    return jax.tree.map(
        lambda ema_value, current_value: ema_value * decay + current_value * (1.0 - decay),
        ema_params,
        params,
    )


def create_model_train_state(params: Any, tx: optax.GradientTransformation) -> ModelTrainState:
    return ModelTrainState(
        step=0,
        params=params,
        params_ema=params,
        opt_state=tx.init(params),
    )


def apply_gradients(
    state: ModelTrainState,
    grads: Any,
    *,
    tx: optax.GradientTransformation,
    ema_decay: float,
) -> ModelTrainState:
    updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_params_ema = update_ema_params(state.params_ema, new_params, decay=float(ema_decay))
    return state.replace(
        step=int(state.step) + 1,
        params=new_params,
        params_ema=new_params_ema,
        opt_state=new_opt_state,
    )
