from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoImageProcessor

from .config_adapter import resolve_repo_value
from .vendor import activate_backend

try:
    from src.disc.lpips_utils import get_ckpt_path
except ModuleNotFoundError:
    from disc.lpips_utils import get_ckpt_path


Array = jax.Array

DINO_RECIPES: dict[str, dict[str, Any]] = {
    "S_16": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "patch_size": 16,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "norm_eps": 1e-6,
    },
    "S_8": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "patch_size": 8,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "norm_eps": 1e-6,
    },
    "B_16": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "patch_size": 16,
        "embed_dim": 768,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "norm_eps": 1e-6,
    },
}


@flax.struct.dataclass
class LatentNormalization:
    mean_hwc: Array | None
    var_hwc: Array | None
    eps: float


@flax.struct.dataclass
class EncoderSpec:
    encoder_input_size: int
    encoder_mean_hwc: Array
    encoder_std_hwc: Array
    encoder_mean_chw: Array
    encoder_std_chw: Array
    encoder_patch_size: int
    latent_dim: int
    latent_hw: int
    noise_tau: float
    reshape_to_2d: bool
    latent_norm: LatentNormalization | None


@flax.struct.dataclass
class Conv2DWeights:
    kernel: Array
    bias: Array | None


@flax.struct.dataclass
class Conv1DWeights:
    kernel: Array
    bias: Array | None


@flax.struct.dataclass
class LinearWeights:
    kernel: Array
    bias: Array | None


@flax.struct.dataclass
class LayerNormWeights:
    scale: Array
    bias: Array


@flax.struct.dataclass
class LPIPSWeights:
    scaling_shift: Array
    scaling_scale: Array
    convs: tuple[Conv2DWeights, ...]
    linear_kernels: tuple[Array, ...]


@flax.struct.dataclass
class DinoBlockWeights:
    norm1: LayerNormWeights
    qkv: LinearWeights
    proj: LinearWeights
    norm2: LayerNormWeights
    fc1: LinearWeights
    fc2: LinearWeights


@flax.struct.dataclass
class DinoBackboneWeights:
    patch_embed: Conv2DWeights
    cls_token: Array
    pos_embed: Array
    blocks: tuple[DinoBlockWeights, ...]
    x_scale: Array
    x_shift: Array
    num_heads: int
    patch_size: int
    key_depths: tuple[int, ...]
    norm_eps: float
    embed_dim: int


@dataclass(slots=True)
class Stage1Paths:
    encoder_pretrained_path: str
    decoder_config_path: Path
    pretrained_decoder_path: str | None
    normalization_stat_path: str | None
    dino_disc_ckpt_path: str


def get_model_dtype(precision: str) -> jnp.dtype:
    return jnp.bfloat16 if precision == "bf16" else jnp.float32


def resolve_stage1_paths(
    *,
    config_path: Path,
    stage1_params: dict[str, Any],
    dino_disc_ckpt_path: str | None,
) -> Stage1Paths:
    encoder_pretrained_path = (
        stage1_params.get("encoder_params", {}).get("dinov2_path")
        or stage1_params.get("encoder_config_path")
        or "facebook/dinov2-with-registers-base"
    )
    encoder_pretrained_path = str(resolve_repo_value(encoder_pretrained_path, config_path=config_path))

    decoder_config_raw = stage1_params.get("decoder_config_path")
    if not decoder_config_raw:
        raise ValueError("stage_1.params.decoder_config_path is required for Stage-1 JAX training.")
    decoder_config_resolved = resolve_repo_value(decoder_config_raw, config_path=config_path)
    decoder_config_path = Path(decoder_config_resolved).expanduser().resolve()
    if decoder_config_path.is_dir():
        decoder_config_path = decoder_config_path / "config.json"
    if not decoder_config_path.is_file():
        raise FileNotFoundError(f"Decoder config not found: {decoder_config_path}")

    pretrained_decoder_path = resolve_repo_value(
        stage1_params.get("pretrained_decoder_path"),
        config_path=config_path,
    )
    normalization_stat_path = resolve_repo_value(
        stage1_params.get("normalization_stat_path"),
        config_path=config_path,
    )
    disc_ckpt = resolve_repo_value(dino_disc_ckpt_path, config_path=config_path) if dino_disc_ckpt_path else None
    if disc_ckpt is None:
        raise ValueError("GAN training requires gan.disc.arch.dino_ckpt_path.")

    return Stage1Paths(
        encoder_pretrained_path=encoder_pretrained_path,
        decoder_config_path=decoder_config_path,
        pretrained_decoder_path=str(pretrained_decoder_path) if pretrained_decoder_path else None,
        normalization_stat_path=str(normalization_stat_path) if normalization_stat_path else None,
        dino_disc_ckpt_path=str(disc_ckpt),
    )


def _torch_to_array(tensor: torch.Tensor, dtype: jnp.dtype = jnp.float32) -> Array:
    return jnp.asarray(tensor.detach().cpu().numpy(), dtype=dtype)


def _torch_conv2d_to_hwio(tensor: torch.Tensor, dtype: jnp.dtype = jnp.float32) -> Array:
    return jnp.asarray(tensor.detach().cpu().numpy().transpose(2, 3, 1, 0), dtype=dtype)


def _torch_linear_to_io(tensor: torch.Tensor, dtype: jnp.dtype = jnp.float32) -> Array:
    return jnp.asarray(tensor.detach().cpu().numpy().T, dtype=dtype)


def _torch_conv1d_to_wio(tensor: torch.Tensor, dtype: jnp.dtype = jnp.float32) -> Array:
    return jnp.asarray(tensor.detach().cpu().numpy().transpose(2, 1, 0), dtype=dtype)


def _load_backend_decoder_config(config_path: Path):
    activate_backend()
    from networks.decoders.vit import ViTMAEConfig

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return ViTMAEConfig(**payload)


def build_stage1_encoder(
    *,
    paths: Stage1Paths,
    stage1_params: dict[str, Any],
    precision: str,
) -> tuple[Any, EncoderSpec]:
    activate_backend()
    from networks.encoders.dino_w_register import DinoWithRegisters

    model_dtype = get_model_dtype(precision)
    encoder_input_size = int(stage1_params.get("encoder_input_size", 224))
    encoder = DinoWithRegisters(
        pretrained_path=paths.encoder_pretrained_path,
        resolution=encoder_input_size,
        dtype=model_dtype,
    )
    encoder.eval()

    processor = AutoImageProcessor.from_pretrained(paths.encoder_pretrained_path)
    mean = jnp.asarray(processor.image_mean, dtype=jnp.float32)
    std = jnp.asarray(processor.image_std, dtype=jnp.float32)
    encoder_mean_hwc = mean.reshape(1, 1, 1, 3)
    encoder_std_hwc = std.reshape(1, 1, 1, 3)
    encoder_mean_chw = mean.reshape(1, 3, 1, 1)
    encoder_std_chw = std.reshape(1, 3, 1, 1)

    latent_norm = None
    if paths.normalization_stat_path:
        stats = torch.load(paths.normalization_stat_path, map_location="cpu")
        mean_hwc = jnp.asarray(stats.get("mean").numpy().transpose(1, 2, 0), dtype=jnp.float32) if stats.get("mean") is not None else None
        var_hwc = jnp.asarray(stats.get("var").numpy().transpose(1, 2, 0), dtype=jnp.float32) if stats.get("var") is not None else None
        latent_norm = LatentNormalization(mean_hwc=mean_hwc, var_hwc=var_hwc, eps=float(stage1_params.get("eps", 1e-5)))

    patch_size = int(encoder.config.patch_size)
    latent_hw = encoder_input_size // patch_size
    spec = EncoderSpec(
        encoder_input_size=encoder_input_size,
        encoder_mean_hwc=encoder_mean_hwc,
        encoder_std_hwc=encoder_std_hwc,
        encoder_mean_chw=encoder_mean_chw,
        encoder_std_chw=encoder_std_chw,
        encoder_patch_size=patch_size,
        latent_dim=int(encoder.config.hidden_size),
        latent_hw=latent_hw,
        noise_tau=float(stage1_params.get("noise_tau", 0.0)),
        reshape_to_2d=bool(stage1_params.get("reshape_to_2d", True)),
        latent_norm=latent_norm,
    )
    return encoder, spec


def build_stage1_decoder(
    *,
    paths: Stage1Paths,
    encoder_spec: EncoderSpec,
    stage1_params: dict[str, Any],
    precision: str,
    seed: int,
) -> Any:
    activate_backend()
    from networks.decoders.vit import GeneralDecoder

    model_dtype = get_model_dtype(precision)
    decoder_config = _load_backend_decoder_config(paths.decoder_config_path)
    decoder_patch_size = int(stage1_params.get("decoder_patch_size", 16))
    decoder_config.hidden_size = encoder_spec.latent_dim
    decoder_config.patch_size = decoder_patch_size
    decoder_config.image_size = encoder_spec.latent_hw * decoder_patch_size
    decoder = GeneralDecoder(
        config=decoder_config,
        num_patches=encoder_spec.latent_hw * encoder_spec.latent_hw,
        dtype=model_dtype,
        rngs=nnx.Rngs(seed),
    )
    if paths.pretrained_decoder_path:
        decoder.load_pretrained(paths.pretrained_decoder_path)
    return decoder


def _resize_nhwc(images: Array, size: int) -> Array:
    return jax.image.resize(images, (images.shape[0], size, size, images.shape[-1]), method="bicubic", antialias=True)


def _maybe_apply_latent_normalization(latents: Array, latent_norm: LatentNormalization | None) -> Array:
    if latent_norm is None or latent_norm.mean_hwc is None or latent_norm.var_hwc is None:
        return latents
    return (latents - latent_norm.mean_hwc[None, ...]) / jnp.sqrt(latent_norm.var_hwc[None, ...] + latent_norm.eps)


def _maybe_invert_latent_normalization(latents: Array, latent_norm: LatentNormalization | None) -> Array:
    if latent_norm is None or latent_norm.mean_hwc is None or latent_norm.var_hwc is None:
        return latents
    return latents * jnp.sqrt(latent_norm.var_hwc[None, ...] + latent_norm.eps) + latent_norm.mean_hwc[None, ...]


def encode_images(
    encoder: Any,
    images_nhwc: Array,
    spec: EncoderSpec,
    *,
    train: bool,
    noise_key: Array | None,
) -> Array:
    if images_nhwc.shape[1] != spec.encoder_input_size or images_nhwc.shape[2] != spec.encoder_input_size:
        images_nhwc = _resize_nhwc(images_nhwc, spec.encoder_input_size)
    inputs = (images_nhwc - spec.encoder_mean_hwc) / spec.encoder_std_hwc
    latents = encoder.encode(inputs, deterministic=not train)
    if train and spec.noise_tau > 0.0 and noise_key is not None:
        sigma_key, sample_key = jax.random.split(noise_key)
        sigma = spec.noise_tau * jax.random.uniform(
            sigma_key,
            (latents.shape[0],) + (1,) * (latents.ndim - 1),
            dtype=jnp.float32,
        )
        latents = latents + sigma.astype(latents.dtype) * jax.random.normal(sample_key, latents.shape, dtype=latents.dtype)
    if spec.reshape_to_2d or spec.latent_norm is not None:
        batch, length, channels = latents.shape
        hw = int(math.sqrt(length))
        latents = jnp.reshape(latents, (batch, hw, hw, channels))
        latents = _maybe_apply_latent_normalization(latents, spec.latent_norm)
    return latents


def decode_latents(decoder: Any, latents: Array, spec: EncoderSpec) -> Array:
    if spec.reshape_to_2d or spec.latent_norm is not None:
        latents = _maybe_invert_latent_normalization(latents, spec.latent_norm)
        batch, height, width, channels = latents.shape
        latents = jnp.reshape(latents, (batch, height * width, channels))
    outputs = decoder(latents, drop_cls_token=False).logits
    recon = decoder.unpatchify(outputs).astype(jnp.float32)
    recon = recon * spec.encoder_std_chw + spec.encoder_mean_chw
    return jnp.clip(recon, 0.0, 1.0)


def decode_latents_to_nhwc(decoder: Any, latents: Array, spec: EncoderSpec) -> Array:
    recon_nchw = decode_latents(decoder, latents, spec)
    return jnp.transpose(recon_nchw, (0, 2, 3, 1))


def stack_real_and_reconstruction_grid(real_nhwc: np.ndarray, recon_nhwc: np.ndarray, *, nrow: int = 4) -> np.ndarray:
    samples = np.concatenate([real_nhwc, recon_nhwc], axis=0)
    samples = np.clip(samples, 0.0, 1.0)
    total = samples.shape[0]
    rows = int(math.ceil(total / nrow))
    height, width, channels = samples.shape[1:]
    grid = np.zeros((rows * height, nrow * width, channels), dtype=np.uint8)
    for idx, sample in enumerate(samples):
        row = idx // nrow
        col = idx % nrow
        tile = np.clip(sample * 255.0, 0.0, 255.0).astype(np.uint8)
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = tile
    return grid


def _conv2d_nhwc(x: Array, weights: Conv2DWeights, *, stride: int = 1, padding: int = 0) -> Array:
    if padding:
        x = jnp.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    y = jax.lax.conv_general_dilated(
        x,
        weights.kernel,
        window_strides=(stride, stride),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    if weights.bias is not None:
        y = y + weights.bias.reshape(1, 1, 1, -1)
    return y


def _conv1d_nlc(x: Array, kernel: Array, bias: Array | None = None) -> Array:
    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NLC", "WIO", "NLC"),
    )
    if bias is not None:
        y = y + bias.reshape(1, 1, -1)
    return y


def _max_pool2d_nhwc(x: Array) -> Array:
    return jax.lax.reduce_window(
        x,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding="VALID",
    )


def _feature_normalize(x: Array, eps: float = 1e-10) -> Array:
    norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True))
    return x / (norm + eps)


def load_lpips_weights(dtype: jnp.dtype = jnp.float32) -> LPIPSWeights:
    import torchvision.models as tv_models

    vgg_model = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).eval()
    conv_layers = [layer for layer in vgg_model.features if isinstance(layer, torch.nn.Conv2d)]
    if len(conv_layers) != 13:
        raise RuntimeError(f"Unexpected VGG16 conv count: {len(conv_layers)}")
    convs = tuple(
        Conv2DWeights(
            kernel=_torch_conv2d_to_hwio(layer.weight, dtype),
            bias=_torch_to_array(layer.bias, dtype) if layer.bias is not None else None,
        )
        for layer in conv_layers
    )

    lpips_state = torch.load(get_ckpt_path("vgg_lpips"), map_location="cpu")
    linear_kernels: list[Array] = []
    for idx in range(5):
        candidates = [
            f"lin{idx}.model.1.weight",
            f"lin{idx}.model.0.weight",
        ]
        for key in candidates:
            if key in lpips_state:
                linear_kernels.append(_torch_conv2d_to_hwio(lpips_state[key], dtype))
                break
        else:
            raise KeyError(f"Missing LPIPS head weight for lin{idx}")

    return LPIPSWeights(
        scaling_shift=jnp.asarray([-.030, -.088, -.188], dtype=dtype).reshape(1, 1, 1, 3),
        scaling_scale=jnp.asarray([.458, .448, .450], dtype=dtype).reshape(1, 1, 1, 3),
        convs=convs,
        linear_kernels=tuple(linear_kernels),
    )


def _vgg16_features(weights: LPIPSWeights, x_nhwc: Array) -> tuple[Array, ...]:
    convs = weights.convs
    x = jax.nn.relu(_conv2d_nhwc(x_nhwc, convs[0], padding=1))
    x = jax.nn.relu(_conv2d_nhwc(x, convs[1], padding=1))
    feat1 = x
    x = _max_pool2d_nhwc(x)

    x = jax.nn.relu(_conv2d_nhwc(x, convs[2], padding=1))
    x = jax.nn.relu(_conv2d_nhwc(x, convs[3], padding=1))
    feat2 = x
    x = _max_pool2d_nhwc(x)

    x = jax.nn.relu(_conv2d_nhwc(x, convs[4], padding=1))
    x = jax.nn.relu(_conv2d_nhwc(x, convs[5], padding=1))
    x = jax.nn.relu(_conv2d_nhwc(x, convs[6], padding=1))
    feat3 = x
    x = _max_pool2d_nhwc(x)

    x = jax.nn.relu(_conv2d_nhwc(x, convs[7], padding=1))
    x = jax.nn.relu(_conv2d_nhwc(x, convs[8], padding=1))
    x = jax.nn.relu(_conv2d_nhwc(x, convs[9], padding=1))
    feat4 = x
    x = _max_pool2d_nhwc(x)

    x = jax.nn.relu(_conv2d_nhwc(x, convs[10], padding=1))
    x = jax.nn.relu(_conv2d_nhwc(x, convs[11], padding=1))
    x = jax.nn.relu(_conv2d_nhwc(x, convs[12], padding=1))
    feat5 = x
    return feat1, feat2, feat3, feat4, feat5


def lpips_loss(weights: LPIPSWeights, input_pm1_nhwc: Array, target_pm1_nhwc: Array) -> Array:
    input_scaled = (input_pm1_nhwc - weights.scaling_shift) / weights.scaling_scale
    target_scaled = (target_pm1_nhwc - weights.scaling_shift) / weights.scaling_scale
    feats_input = _vgg16_features(weights, input_scaled)
    feats_target = _vgg16_features(weights, target_scaled)

    values = []
    for feat_in, feat_target, kernel in zip(feats_input, feats_target, weights.linear_kernels, strict=True):
        feat_in = _feature_normalize(feat_in)
        feat_target = _feature_normalize(feat_target)
        diff = jnp.square(feat_in - feat_target)
        diff = _conv2d_nhwc(diff, Conv2DWeights(kernel=kernel, bias=None))
        values.append(jnp.mean(diff, axis=(1, 2, 3)))
    stacked = sum(values)
    return jnp.mean(stacked)


def _load_layer_norm_weights(state: dict[str, torch.Tensor], prefix: str, dtype: jnp.dtype) -> LayerNormWeights:
    return LayerNormWeights(
        scale=_torch_to_array(state[f"{prefix}.weight"], dtype),
        bias=_torch_to_array(state[f"{prefix}.bias"], dtype),
    )


def _load_linear_weights(state: dict[str, torch.Tensor], prefix: str, dtype: jnp.dtype) -> LinearWeights:
    bias = state.get(f"{prefix}.bias")
    return LinearWeights(
        kernel=_torch_linear_to_io(state[f"{prefix}.weight"], dtype),
        bias=_torch_to_array(bias, dtype) if bias is not None else None,
    )


def load_dino_backbone_weights(
    ckpt_path: str,
    *,
    recipe: str,
    dtype: jnp.dtype = jnp.float32,
) -> DinoBackboneWeights:
    if recipe not in DINO_RECIPES:
        raise ValueError(f"Unsupported discriminator recipe: {recipe}")
    recipe_cfg = DINO_RECIPES[recipe]
    state = torch.load(ckpt_path, map_location="cpu")
    for key in list(state.keys()):
        if ".attn.qkv.bias" in key:
            bias = state[key]
            channels = bias.numel() // 3
            bias = bias.clone()
            bias[channels : 2 * channels].zero_()
            state[key] = bias

    blocks = []
    for idx in range(recipe_cfg["depth"]):
        blocks.append(
            DinoBlockWeights(
                norm1=_load_layer_norm_weights(state, f"blocks.{idx}.norm1", dtype),
                qkv=_load_linear_weights(state, f"blocks.{idx}.attn.qkv", dtype),
                proj=_load_linear_weights(state, f"blocks.{idx}.attn.proj", dtype),
                norm2=_load_layer_norm_weights(state, f"blocks.{idx}.norm2", dtype),
                fc1=_load_linear_weights(state, f"blocks.{idx}.mlp.fc1", dtype),
                fc2=_load_linear_weights(state, f"blocks.{idx}.mlp.fc2", dtype),
            )
        )

    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    x_scale = jnp.asarray((0.5 / std).reshape(1, 1, 1, 3), dtype=dtype)
    x_shift = jnp.asarray(((0.5 - mean) / std).reshape(1, 1, 1, 3), dtype=dtype)
    return DinoBackboneWeights(
        patch_embed=Conv2DWeights(
            kernel=_torch_conv2d_to_hwio(state["patch_embed.proj.weight"], dtype),
            bias=_torch_to_array(state["patch_embed.proj.bias"], dtype),
        ),
        cls_token=_torch_to_array(state["cls_token"], dtype),
        pos_embed=_torch_to_array(state["pos_embed"], dtype),
        blocks=tuple(blocks),
        x_scale=x_scale,
        x_shift=x_shift,
        num_heads=int(recipe_cfg["num_heads"]),
        patch_size=int(recipe_cfg["patch_size"]),
        key_depths=tuple(int(depth) for depth in recipe_cfg["key_depths"]),
        norm_eps=float(recipe_cfg["norm_eps"]),
        embed_dim=int(recipe_cfg["embed_dim"]),
    )


def _layer_norm(x: Array, weights: LayerNormWeights, eps: float) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * weights.scale.reshape(1, 1, -1) + weights.bias.reshape(1, 1, -1)


def _linear(x: Array, weights: LinearWeights) -> Array:
    y = jnp.einsum("...c,cd->...d", x, weights.kernel)
    if weights.bias is not None:
        y = y + weights.bias
    return y


def _gelu_tanh(x: Array) -> Array:
    return jax.nn.gelu(x, approximate=True)


def _self_attention(x: Array, qkv_weights: LinearWeights, proj_weights: LinearWeights, *, num_heads: int) -> Array:
    batch, length, channels = x.shape
    head_dim = channels // num_heads
    qkv = _linear(x, qkv_weights)
    qkv = jnp.reshape(qkv, (batch, length, 3, num_heads, head_dim))
    qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
    query, key, value = qkv[0], qkv[1], qkv[2]
    scale = 1.0 / math.sqrt(head_dim)
    attn = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) * scale
    attn = jax.nn.softmax(attn, axis=-1)
    context = jnp.matmul(attn, value)
    context = jnp.transpose(context, (0, 2, 1, 3))
    context = jnp.reshape(context, (batch, length, channels))
    return _linear(context, proj_weights)


def _dino_block_forward(x: Array, weights: DinoBlockWeights, *, num_heads: int, eps: float) -> Array:
    x = x + _self_attention(_layer_norm(x, weights.norm1, eps), weights.qkv, weights.proj, num_heads=num_heads)
    mlp_hidden = _gelu_tanh(_linear(_layer_norm(x, weights.norm2, eps), weights.fc1))
    x = x + _linear(mlp_hidden, weights.fc2)
    return x


def dino_backbone_forward(weights: DinoBackboneWeights, x_pm1_nhwc: Array) -> tuple[Array, ...]:
    if x_pm1_nhwc.shape[1] != 224 or x_pm1_nhwc.shape[2] != 224:
        x_pm1_nhwc = jax.image.resize(x_pm1_nhwc, (x_pm1_nhwc.shape[0], 224, 224, x_pm1_nhwc.shape[-1]), method="bilinear", antialias=True)
    x = x_pm1_nhwc * weights.x_scale + weights.x_shift
    x = _conv2d_nhwc(x, weights.patch_embed, stride=weights.patch_size)
    batch, height, width, channels = x.shape
    x = jnp.reshape(x, (batch, height * width, channels))
    cls_tokens = jnp.broadcast_to(weights.cls_token, (batch,) + weights.cls_token.shape[1:])
    x = jnp.concatenate([cls_tokens, x], axis=1)
    x = x + weights.pos_embed

    activations: list[Array] = []
    for idx, block_weights in enumerate(weights.blocks):
        x = _dino_block_forward(x, block_weights, num_heads=weights.num_heads, eps=weights.norm_eps)
        if idx in weights.key_depths:
            activations.append(x[:, 1:, :])
    activations.insert(0, x[:, 1:, :])
    return tuple(activations)


class BatchNormLocal1D(nnx.Module):
    def __init__(self, channels: int, *, dtype: jnp.dtype = jnp.float32):
        self.scale = nnx.Param(jnp.ones((channels,), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((channels,), dtype=dtype))
        self.eps = 1e-6

    def __call__(self, x: Array) -> Array:
        mean = jnp.mean(x.astype(jnp.float32), axis=1, keepdims=True)
        var = jnp.var(x.astype(jnp.float32), axis=1, keepdims=True)
        x = (x.astype(jnp.float32) - mean) / jnp.sqrt(var + self.eps)
        return x * self.scale.value.reshape(1, 1, -1) + self.bias.value.reshape(1, 1, -1)


class GroupNorm1D(nnx.Module):
    def __init__(self, channels: int, *, groups: int = 32, dtype: jnp.dtype = jnp.float32):
        self.groups = min(groups, channels)
        self.scale = nnx.Param(jnp.ones((channels,), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((channels,), dtype=dtype))
        self.eps = 1e-6

    def __call__(self, x: Array) -> Array:
        batch, length, channels = x.shape
        groups = max(1, min(self.groups, channels))
        if channels % groups != 0:
            groups = math.gcd(channels, groups)
        x_reshaped = jnp.reshape(x.astype(jnp.float32), (batch, length, groups, channels // groups))
        mean = jnp.mean(x_reshaped, axis=(1, 3), keepdims=True)
        var = jnp.var(x_reshaped, axis=(1, 3), keepdims=True)
        x_reshaped = (x_reshaped - mean) / jnp.sqrt(var + self.eps)
        x = jnp.reshape(x_reshaped, (batch, length, channels))
        return x * self.scale.value.reshape(1, 1, -1) + self.bias.value.reshape(1, 1, -1)


class TrainableConv1D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        use_bias: bool = True,
        using_spec_norm: bool = False,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        kernel_init = jax.nn.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform")
        self.kernel = nnx.Param(kernel_init(rngs.params(), (kernel_size, in_channels, out_channels), dtype))
        self.bias = nnx.Param(jnp.zeros((out_channels,), dtype=dtype)) if use_bias else None
        self.kernel_size = int(kernel_size)
        self.using_spec_norm = bool(using_spec_norm)

    def _normalized_kernel(self) -> Array:
        kernel = self.kernel.value
        if not self.using_spec_norm:
            return kernel
        matrix = jnp.reshape(jnp.transpose(kernel, (2, 0, 1)), (kernel.shape[-1], -1)).astype(jnp.float32)
        sigma = jnp.linalg.svd(matrix, compute_uv=False)[0]
        sigma = jnp.maximum(sigma, 1e-6)
        return kernel / sigma.astype(kernel.dtype)

    def __call__(self, x: Array) -> Array:
        pad = self.kernel_size // 2
        if pad > 0:
            x = jnp.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
        return _conv1d_nlc(x, self._normalized_kernel(), None if self.bias is None else self.bias.value)


class ConvNormAct1D(nnx.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        norm_type: str,
        using_spec_norm: bool,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.conv = TrainableConv1D(
            channels,
            channels,
            kernel_size=kernel_size,
            use_bias=True,
            using_spec_norm=using_spec_norm,
            dtype=dtype,
            rngs=rngs,
        )
        if norm_type == "bn":
            self.norm = BatchNormLocal1D(channels, dtype=dtype)
        elif norm_type == "gn":
            self.norm = GroupNorm1D(channels, dtype=dtype)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def __call__(self, x: Array) -> Array:
        x = self.conv(x)
        x = self.norm(x)
        return jax.nn.leaky_relu(x, negative_slope=0.2)


class ResidualHeadBlock(nnx.Module):
    def __init__(self, inner: ConvNormAct1D):
        self.inner = inner
        self.scale = 1.0 / math.sqrt(2.0)

    def __call__(self, x: Array) -> Array:
        return (self.inner(x) + x) * self.scale


class DinoDiscriminatorHead(nnx.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        ks: int,
        norm_type: str,
        using_spec_norm: bool,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.pre = ConvNormAct1D(
            embed_dim,
            kernel_size=1,
            norm_type=norm_type,
            using_spec_norm=using_spec_norm,
            dtype=dtype,
            rngs=rngs,
        )
        self.residual = ResidualHeadBlock(
            ConvNormAct1D(
                embed_dim,
                kernel_size=ks,
                norm_type=norm_type,
                using_spec_norm=using_spec_norm,
                dtype=dtype,
                rngs=rngs,
            )
        )
        self.out = TrainableConv1D(
            embed_dim,
            1,
            kernel_size=1,
            use_bias=True,
            using_spec_norm=using_spec_norm,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        x = self.pre(x)
        x = self.residual(x)
        return self.out(x)


class DinoDiscriminatorHeads(nnx.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        ks: int,
        norm_type: str,
        using_spec_norm: bool,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        num_streams = int(num_heads)
        self.heads = [
            DinoDiscriminatorHead(
                embed_dim=embed_dim,
                ks=ks,
                norm_type=norm_type,
                using_spec_norm=using_spec_norm,
                dtype=dtype,
                rngs=rngs,
            )
            for _ in range(num_streams)
        ]

    def __call__(self, activations: Sequence[Array]) -> Array:
        outputs = []
        for head, activation in zip(self.heads, activations, strict=True):
            logits = head(activation)
            outputs.append(jnp.reshape(logits, (logits.shape[0], -1)))
        return jnp.concatenate(outputs, axis=1)


def hinge_d_loss(logits_real: Array, logits_fake: Array) -> Array:
    loss_real = jnp.mean(jax.nn.relu(1.0 - logits_real))
    loss_fake = jnp.mean(jax.nn.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_g_loss(logits_fake: Array) -> Array:
    return -jnp.mean(logits_fake)


def tree_l2_norm(tree: Any) -> Array:
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(tree) if leaf is not None]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32))) for leaf in leaves))


def compute_adaptive_weight(recon_grads: Any, gan_grads: Any, max_value: float) -> Array:
    recon_norm = tree_l2_norm(recon_grads)
    gan_norm = tree_l2_norm(gan_grads)
    ratio = recon_norm / (gan_norm + 1e-6)
    return jnp.clip(ratio, 0.0, float(max_value))


def _translate_images(images: Array, offsets_y: Array, offsets_x: Array) -> Array:
    batch, height, width, channels = images.shape
    padded = jnp.pad(images, ((0, 0), (1, 1), (1, 1), (0, 0)))

    def _translate_single(image_pad: Array, dy: Array, dx: Array) -> Array:
        ys = jnp.clip(jnp.arange(height) + dy + 1, 0, height + 1)
        xs = jnp.clip(jnp.arange(width) + dx + 1, 0, width + 1)
        return image_pad[ys[:, None], xs[None, :], :]

    return jax.vmap(_translate_single)(padded, offsets_y, offsets_x)


def _cutout_images(images: Array, centers_y: Array, centers_x: Array, *, ratio: float) -> Array:
    batch, height, width, _channels = images.shape
    cut_h = max(1, round(height * ratio))
    cut_w = max(1, round(width * ratio))
    ys = jnp.arange(height)
    xs = jnp.arange(width)

    def _cutout_single(image: Array, center_y: Array, center_x: Array) -> Array:
        mask_y = (ys >= center_y - cut_h // 2) & (ys < center_y - cut_h // 2 + cut_h)
        mask_x = (xs >= center_x - cut_w // 2) & (xs < center_x - cut_w // 2 + cut_w)
        mask = ~(mask_y[:, None] & mask_x[None, :])
        return image * mask[..., None].astype(image.dtype)

    return jax.vmap(_cutout_single)(images, centers_y, centers_x)


def diffaug(images: Array, *, key: Array, prob: float, cutout: float) -> Array:
    if prob <= 1e-6:
        return images
    key_trans, key_color, key_cut, key_misc = jax.random.split(key, 4)
    apply_trans = jax.random.bernoulli(key_misc, p=min(max(prob, 0.0), 1.0))
    apply_color = jax.random.bernoulli(jax.random.fold_in(key_misc, 1), p=min(max(prob, 0.0), 1.0))
    apply_cut = jax.random.bernoulli(jax.random.fold_in(key_misc, 2), p=min(max(prob, 0.0), 1.0))

    batch, height, width, _channels = images.shape

    def _apply_translation(x: Array) -> Array:
        delta_h = round(height * 0.125)
        delta_w = round(width * 0.125)
        dy = jax.random.randint(key_trans, (batch,), minval=-delta_h, maxval=delta_h + 1)
        dx = jax.random.randint(jax.random.fold_in(key_trans, 1), (batch,), minval=-delta_w, maxval=delta_w + 1)
        return _translate_images(x, dy, dx)

    def _apply_color(x: Array) -> Array:
        brightness = jax.random.uniform(key_color, (batch, 1, 1, 1), minval=-0.5, maxval=0.5)
        saturation = jax.random.uniform(jax.random.fold_in(key_color, 1), (batch, 1, 1, 1), minval=0.0, maxval=2.0)
        contrast = jax.random.uniform(jax.random.fold_in(key_color, 2), (batch, 1, 1, 1), minval=0.5, maxval=1.5)
        x = x + brightness
        mean_channel = jnp.mean(x, axis=-1, keepdims=True)
        x = (x - mean_channel) * saturation + mean_channel
        mean_global = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
        x = (x - mean_global) * contrast + mean_global
        return x

    def _apply_cutout(x: Array) -> Array:
        center_y = jax.random.randint(key_cut, (batch,), minval=0, maxval=height)
        center_x = jax.random.randint(jax.random.fold_in(key_cut, 1), (batch,), minval=0, maxval=width)
        return _cutout_images(x, center_y, center_x, ratio=cutout)

    images = jax.lax.cond(apply_trans, _apply_translation, lambda x: x, images)
    images = jax.lax.cond(apply_color, _apply_color, lambda x: x, images)
    images = jax.lax.cond(apply_cut & (cutout > 0.0), _apply_cutout, lambda x: x, images)
    return images
