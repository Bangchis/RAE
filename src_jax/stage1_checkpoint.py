from __future__ import annotations

from pathlib import Path
from typing import Any

from flax import jax_utils
from flax import nnx
import jax
import numpy as np
import torch


def _to_host_pure_dict(state: Any) -> Any:
    if hasattr(state, "to_pure_dict"):
        state = state.to_pure_dict()
    return jax.device_get(jax.tree_util.tree_map(lambda x: np.asarray(x) if hasattr(x, "shape") else x, state))


def _restore_from_pure_dict(template_state: Any, pure_dict: Any) -> Any:
    nnx.State.replace_by_pure_dict(template_state, pure_dict)
    return template_state


def unreplicate_tree(tree: Any) -> Any:
    return jax.device_get(jax_utils.unreplicate(tree))


def replicate_tree(tree: Any) -> Any:
    return jax_utils.replicate(tree)


def save_stage1_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    step: int,
    decoder_state: Any,
    decoder_opt_state: Any,
    ema_decoder_state: Any,
    disc_state: Any,
    disc_opt_state: Any,
    rng_state: Any,
    metadata: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "epoch": int(epoch),
        "step": int(step),
        "decoder_state": _to_host_pure_dict(decoder_state),
        "decoder_opt_state": _to_host_pure_dict(decoder_opt_state),
        "ema_decoder_state": _to_host_pure_dict(ema_decoder_state),
        "disc_state": _to_host_pure_dict(disc_state),
        "disc_opt_state": _to_host_pure_dict(disc_opt_state),
        "rng_state": jax.device_get(jax.random.key_data(rng_state)),
        "metadata": metadata or {},
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


def load_stage1_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    return torch.load(checkpoint_path, map_location="cpu")


def restore_decoder_state(template_state: Any, payload: dict[str, Any], *, use_ema: bool) -> Any:
    key = "ema_decoder_state" if use_ema else "decoder_state"
    return _restore_from_pure_dict(template_state, payload[key])


def restore_optimizer_state(template_state: Any, payload: dict[str, Any], *, key: str) -> Any:
    return _restore_from_pure_dict(template_state, payload[key])


def decoder_to_torch_state_dict(decoder: Any) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    state_dict["decoder_embed.weight"] = torch.from_numpy(np.asarray(decoder.decoder_embed.kernel.value.T, dtype=np.float32))
    if decoder.decoder_embed.bias is not None:
        state_dict["decoder_embed.bias"] = torch.from_numpy(np.asarray(decoder.decoder_embed.bias.value, dtype=np.float32))

    state_dict["decoder_pos_embed"] = torch.from_numpy(np.asarray(decoder.decoder_pos_embed.value, dtype=np.float32))
    state_dict["trainable_cls_token"] = torch.from_numpy(np.asarray(decoder.trainable_cls_token.value, dtype=np.float32))

    for idx, layer in enumerate(decoder.decoder_layers):
        prefix = f"decoder_layers.{idx}"
        state_dict[f"{prefix}.attention.attention.query.weight"] = torch.from_numpy(np.asarray(layer.attention.attention.query.kernel.value.T, dtype=np.float32))
        state_dict[f"{prefix}.attention.attention.query.bias"] = torch.from_numpy(np.asarray(layer.attention.attention.query.bias.value, dtype=np.float32))
        state_dict[f"{prefix}.attention.attention.key.weight"] = torch.from_numpy(np.asarray(layer.attention.attention.key.kernel.value.T, dtype=np.float32))
        state_dict[f"{prefix}.attention.attention.key.bias"] = torch.from_numpy(np.asarray(layer.attention.attention.key.bias.value, dtype=np.float32))
        state_dict[f"{prefix}.attention.attention.value.weight"] = torch.from_numpy(np.asarray(layer.attention.attention.value.kernel.value.T, dtype=np.float32))
        state_dict[f"{prefix}.attention.attention.value.bias"] = torch.from_numpy(np.asarray(layer.attention.attention.value.bias.value, dtype=np.float32))
        state_dict[f"{prefix}.attention.output.dense.weight"] = torch.from_numpy(np.asarray(layer.attention.output.dense.kernel.value.T, dtype=np.float32))
        state_dict[f"{prefix}.attention.output.dense.bias"] = torch.from_numpy(np.asarray(layer.attention.output.dense.bias.value, dtype=np.float32))
        state_dict[f"{prefix}.intermediate.dense.weight"] = torch.from_numpy(np.asarray(layer.intermediate.dense.kernel.value.T, dtype=np.float32))
        state_dict[f"{prefix}.intermediate.dense.bias"] = torch.from_numpy(np.asarray(layer.intermediate.dense.bias.value, dtype=np.float32))
        state_dict[f"{prefix}.output.dense.weight"] = torch.from_numpy(np.asarray(layer.output.dense.kernel.value.T, dtype=np.float32))
        state_dict[f"{prefix}.output.dense.bias"] = torch.from_numpy(np.asarray(layer.output.dense.bias.value, dtype=np.float32))
        state_dict[f"{prefix}.layernorm_before.weight"] = torch.from_numpy(np.asarray(layer.layernorm_before.scale.value, dtype=np.float32))
        state_dict[f"{prefix}.layernorm_before.bias"] = torch.from_numpy(np.asarray(layer.layernorm_before.bias.value, dtype=np.float32))
        state_dict[f"{prefix}.layernorm_after.weight"] = torch.from_numpy(np.asarray(layer.layernorm_after.scale.value, dtype=np.float32))
        state_dict[f"{prefix}.layernorm_after.bias"] = torch.from_numpy(np.asarray(layer.layernorm_after.bias.value, dtype=np.float32))

    state_dict["decoder_norm.weight"] = torch.from_numpy(np.asarray(decoder.decoder_norm.scale.value, dtype=np.float32))
    state_dict["decoder_norm.bias"] = torch.from_numpy(np.asarray(decoder.decoder_norm.bias.value, dtype=np.float32))
    state_dict["decoder_pred.weight"] = torch.from_numpy(np.asarray(decoder.decoder_pred.kernel.value.T, dtype=np.float32))
    state_dict["decoder_pred.bias"] = torch.from_numpy(np.asarray(decoder.decoder_pred.bias.value, dtype=np.float32))
    return state_dict
