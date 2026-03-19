# Config Reference

## Overview

All main entrypoints are driven by OmegaConf YAML files. The central loader is
[src/utils/train_utils.py](../src/utils/train_utils.py), which
parses these top-level sections:

- `stage_1`
- `stage_2`
- `transport`
- `sampler`
- `guidance`
- `misc`
- `training`

`src/train.py` additionally reads the optional `eval` block directly from the
full config. The JAX adapter in `src_jax/` consumes the same top-level blocks
and translates them into the backend configuration expected by NNX.

## Section-to-Script Matrix

| Section | `train.py` | `sample.py` | `sample_ddp.py` | `stage1_sample.py` | `stage1_sample_ddp.py` |
| --- | --- | --- | --- | --- | --- |
| `stage_1` | yes | yes | yes | yes | yes |
| `stage_2` | yes | yes | yes | no | no |
| `transport` | yes | no | no | no | no |
| `sampler` | yes | yes | yes | no | no |
| `guidance` | yes | yes | yes | no | no |
| `misc` | yes | yes | yes | no | no |
| `training` | yes | no | no | no | no |
| `eval` | yes | no | no | no | no |

JAX adapter coverage:

| Section | `src_jax/train.py` | `src_jax/sample.py` | `src_jax/sample_ddp.py` | `src_jax/stage1_sample.py` |
| --- | --- | --- | --- | --- |
| `stage_1` | yes | yes | yes | yes |
| `stage_2` | yes | yes | yes | no |
| `transport` | yes | yes | yes | no |
| `sampler` | yes | yes | yes | no |
| `guidance` | yes | yes | yes | no |
| `misc` | yes | yes | yes | no |
| `training` | yes | no | no | no |
| `eval` | yes, partially | no | no | no |

## `stage_1`

Typical shape:

```yaml
stage_1:
  target: stage1.RAE
  ckpt: null
  params:
    encoder_cls: Dinov2withNorm
    encoder_config_path: facebook/dinov2-with-registers-base
    encoder_input_size: 224
    encoder_params: {...}
    decoder_config_path: configs/decoder/ViTXL
    pretrained_decoder_path: models/...
    noise_tau: 0.0
    reshape_to_2d: true
    normalization_stat_path: models/stats/...
```

Key fields:

- `target`: usually `stage1.RAE`
- `ckpt`: optional full RAE checkpoint
- `params.encoder_cls`: encoder implementation key from `src/stage1/encoders/`
- `params.encoder_config_path`: Hugging Face config path for image processor and config
- `params.encoder_input_size`: resolution expected by the representation encoder
- `params.decoder_config_path`: decoder config
- `params.pretrained_decoder_path`: decoder initialization weights
- `params.noise_tau`: latent noising strength during training
- `params.reshape_to_2d`: whether latent tokens become `(C, H, W)`
- `params.normalization_stat_path`: optional latent mean/variance stats

## `stage_2`

Typical shape:

```yaml
stage_2:
  target: stage2.models.DDT.DiTwDDTHead
  ckpt: null
  params:
    input_size: 16
    patch_size: 1
    in_channels: 768
    hidden_size: [1152, 2048]
    depth: [28, 2]
    num_heads: [16, 16]
    mlp_ratio: 4.0
    class_dropout_prob: 0.1
    num_classes: 1000
    use_qknorm: false
    use_swiglu: true
    use_rope: true
    use_rmsnorm: true
    wo_shift: false
    use_pos_embed: true
```

Common fields:

- `target`: model class path
- `ckpt`: checkpoint used for sampling or fine-tuning resume
- `params.input_size`: latent spatial resolution
- `params.in_channels`: latent channel count
- `params.hidden_size`: encoder/decoder hidden widths
- `params.depth`: number of encoder and decoder transformer blocks
- `params.num_heads`: attention heads per tower
- feature toggles such as `use_rope`, `use_rmsnorm`, and `use_swiglu`

For the JAX adapter:

- `stage_2.ckpt` may be either a PyTorch `.pt` checkpoint or a JAX Orbax
  directory
- `target` is used only to infer which NNX backbone should be instantiated
  (`lightning_ddt`, `lightning_dit`, or `dit`)

## `transport`

Typical shape:

```yaml
transport:
  params:
    path_type: Linear
    prediction: velocity
    loss_weight: null
    time_dist_type: uniform
```

Fields used by `src/train.py`:

- `path_type`
- `prediction`
- `loss_weight`
- `time_dist_type`

The training loop also injects `time_dist_shift` at runtime from the `misc`
section.

## `sampler`

Typical shape:

```yaml
sampler:
  mode: ODE
  params:
    sampling_method: euler
    num_steps: 50
    atol: 1.0e-6
    rtol: 1.0e-3
    reverse: false
```

On this branch:

- `mode` must be `ODE` for the provided sampling entrypoints
- `num_steps` controls the manual Euler schedule length

## `guidance`

Typical shape:

```yaml
guidance:
  method: cfg
  scale: 1.0
  t_min: 0.0
  t_max: 1.0
```

Supported methods:

- `cfg`
- `autoguidance`

If using `autoguidance`, add:

```yaml
guidance:
  method: autoguidance
  scale: 2.0
  guidance_model:
    target: ...
    ckpt: ...
    params: ...
```

On the JAX path:

- `cfg` maps to the same-model conditional/unconditional guidance flow
- `autoguidance` loads `guidance_model` as a second network and uses it as the
  guide model during sampling

## `misc`

Typical shape:

```yaml
misc:
  latent_size: [768, 16, 16]
  num_classes: 1000
  null_label: 1000
  time_dist_shift_dim: 196608
  time_dist_shift_base: 4096
```

Meaning:

- `latent_size`: `(C, H, W)` latent shape produced by Stage 1 and consumed by Stage 2
- `num_classes`: class vocabulary size
- `null_label`: null token for classifier-free guidance; defaults to `num_classes`
- `time_dist_shift_dim` and `time_dist_shift_base`: runtime scaling inputs for the shifted time distribution

## `training`

Common fields consumed by `src/train.py`:

```yaml
training:
  global_seed: 0
  epochs: 1400
  global_batch_size: 1024
  grad_accum_steps: 1
  ema_decay: 0.9995
  num_workers: 4
  log_every: 100
  ckpt_every: 5000
  sample_every: 10000
  base_lr: 0.0002
  final_lr: 0.00002
  beta: [0.9, 0.95]
  wd: 0.0
  schedule_type: linear
  decay_start_epoch: 40
  decay_end_epoch: 800
  clip_grad: 1.0
```

Notes:

- `global_batch_size` is the true batch across all TPU cores and accumulation steps
- `micro_batch_size` is derived internally as
  `global_batch_size / (world_size * grad_accum_steps)`
- optimizer defaults to AdamW
- scheduler supports `linear` and `cosine`
- nested `optimizer` and `scheduler` sub-blocks are also supported by
  `src/utils/optim_utils.py`

The JAX adapter also accepts CLI overrides in the form:

```bash
python3 src_jax/train.py \
  --config <config> \
  --set training.global_batch_size=256 \
  --set guidance.scale=1.5
```

## `eval`

This block is optional and only consumed by `src/train.py`.

On the JAX path, only the FID-related subset is mapped today. Validation-loss
parity with the XLA loop is not yet implemented in `src_jax/train.py`.

### Validation Loss Keys

```yaml
eval:
  data_path: data/imagenet/val/
  eval_every: 5000
  batch_size: 128
  num_workers: 4
  max_batches: 32
  eval_model: false
```

Meaning:

- `data_path`: validation `ImageFolder`
- `eval_every`: cadence in optimizer steps
- `batch_size`: per-device evaluation batch size
- `num_workers`: dataloader workers for eval
- `max_batches`: optional per-rank cap
- `eval_model`: score the non-EMA model in addition to EMA

### FID Keys

```yaml
eval:
  fid_ref: /path/to/reference_stats.npz
  fid_every: 25000
  fid_num_samples: 4096
  fid_per_proc_batch_size: 4
  fid_batch_size: 128
  fid_device: cpu
  fid_num_threads: 96
  fid_label_sampling: random
  fid_eval_model: false
```

Meaning:

- `fid_ref`: reference statistics built by `src/build_fid_stats.py`
- `fid_every`: cadence in optimizer steps
- `fid_num_samples`: number of generated images per FID measurement
- `fid_per_proc_batch_size`: generation batch per TPU core
- `fid_batch_size`: host-side Inception batch size
- `fid_device`: `cpu`, `cuda`, or `auto`
- `fid_num_threads`: optional CPU thread count for host-side scoring
- `fid_label_sampling`: `equal` or `random`
- `fid_eval_model`: also score the online model, not only EMA

## Example Stage 2 Training Config Skeleton

```yaml
stage_1:
  ...
stage_2:
  ...
transport:
  ...
sampler:
  ...
guidance:
  ...
misc:
  latent_size: [768, 16, 16]
  num_classes: 1000
training:
  global_batch_size: 1024
  grad_accum_steps: 1
  epochs: 1400
eval:
  data_path: data/imagenet/val/
  eval_every: 5000
  fid_ref: data/imagenet/reference_stats.npz
  fid_every: 25000
  fid_num_samples: 4096
  fid_device: cpu
  fid_num_threads: 96
```

## Practical Editing Guidance

- Start from an existing file under `configs/stage2/training/` or
  `configs/stage2/sampling/`.
- Keep `misc.latent_size` aligned with the Stage 1 encoder output.
- Keep `misc.num_classes`, `stage_2.params.num_classes`, and your label
  sampling assumptions consistent.
- If you enable `fid_label_sampling: equal`, ensure `fid_num_samples` is
  divisible by `num_classes`.
- For CPU-only FID, tune `fid_batch_size` and `fid_num_threads` together.
