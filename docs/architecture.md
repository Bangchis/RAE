# Architecture

## Goal

This repository implements a two-stage image generation pipeline based on
Representation Autoencoders (RAE):

1. Stage 1 maps images into a semantically rich latent space with a frozen
   representation encoder and a trainable decoder.
2. Stage 2 learns a diffusion transformer in that latent space and decodes the
   sampled latents back into images through the Stage 1 decoder.

The XLA branch focuses on TPU execution for Stage 2 training and sampling, with
optional host-side FID scoring. The `jax-sit-dh` branch also adds a thin JAX/NNX
compatibility layer under `src_jax/` that maps the repository's existing YAML
schema into a pinned `diffuse_nnx` backend, including backend-native FID
reference building, held-out validation loss, and a compatibility patch that
keeps backend EMA initialization aligned with the live model weights.

## End-to-End Data Flow

```text
image
  -> Stage 1 encoder
  -> latent tensor z
  -> Stage 2 transport objective / sampler
  -> predicted latent z'
  -> Stage 1 decoder
  -> reconstructed or generated image
```

For Stage 2 training, the model does not learn directly in pixel space. The
training loop first encodes images through the frozen RAE, then computes the
transport loss in latent space.

## Major Runtime Components

### Stage 1: RAE

Implemented in [src/stage1/rae.py](../src/stage1/rae.py).

- Encoder: loaded from `src/stage1/encoders/`
- Decoder: loaded from `src/stage1/decoders/`
- Optional latent normalization: driven by `normalization_stat_path`
- Optional latent noising during training: controlled by `noise_tau`

Key behavior:

- `encode(x)` resizes input to the encoder resolution, normalizes by the
  encoder processor statistics, runs the frozen encoder, optionally reshapes the
  token sequence into `(B, C, H, W)`, and applies latent normalization.
- `decode(z)` reverses normalization, converts latent maps back to token
  sequences if needed, and reconstructs pixels through the ViT decoder.

### Stage 2: SiT in Latent Space

The branch-default Stage 2 target is [src/stage2/models/SiT.py](../src/stage2/models/SiT.py),
which exposes `SiTDH` as a repo-facing alias over the two-tower
`DiTwDDTHead` implementation in [src/stage2/models/DDT.py](../src/stage2/models/DDT.py).
This keeps the Stage 2 architecture aligned with the DH variant while the JAX
adapter continues to use the `sit` transport interface.

Design highlights:

- latent input instead of RGB input
- timestep embedding through Gaussian Fourier features
- class conditioning through `LabelEmbedder`
- rotary embeddings, RMSNorm, SwiGLU, and qk-norm as configurable options
- support for classifier-free guidance and autoguidance during sampling

### Transport Objective

Implemented in [src/stage2/transport/transport.py](../src/stage2/transport/transport.py).

The transport layer is the contract between the latent diffusion model and the
training loop.

Main responsibilities:

- sample a time `t`
- generate the path interpolation target
- compute the supervision target for the configured model type
- provide the drift function used by ODE/SDE samplers

For this branch, Stage 2 training uses `transport.training_losses(...)` inside
the TPU loop.

## Entrypoints

### Training

- [src/train.py](../src/train.py): Stage 2 TPU training, EMA,
  checkpointing, wandb logging, validation loss, preview sampling, and optional
  host-side FID.
- [src_jax/train.py](../src_jax/train.py): Stage 2 JAX/NNX
  training, OmegaConf CLI overrides, wandb logging, held-out validation loss,
  optional online FID, and optional Hugging Face upload of the finished
  workdir.

### Stage 2 Sampling

- [src/sample.py](../src/sample.py): small single-process sample run
  for quick inspection
- [src/sample_ddp.py](../src/sample_ddp.py): distributed TPU
  sampling, PNG export, `.npz` creation, and optional post-sampling FID
- [src_jax/sample.py](../src_jax/sample.py): Stage 2 JAX/NNX
  single-run sampling with either CFG or autoguidance
- [src_jax/sample_ddp.py](../src_jax/sample_ddp.py): distributed
  JAX/NNX sampling with PNG export, optional per-rank `.npz`, and optional FID

### Stage 1 Reconstruction

- [src/stage1_sample.py](../src/stage1_sample.py): reconstruct a
  single image
- [src/stage1_sample_ddp.py](../src/stage1_sample_ddp.py):
  distributed reconstruction of an `ImageFolder`
- [src_jax/stage1_sample.py](../src_jax/stage1_sample.py):
  reconstruct a single image through the JAX RAE path
- [src_jax/reconstruct_folder.py](../src_jax/reconstruct_folder.py):
  reconstruct an image folder through the JAX RAE path
- [src_jax/build_stage1_stats.py](../src_jax/build_stage1_stats.py):
  compute dataset-specific Stage 1 latent normalization stats
- [src_jax/export_celebahq_hf.py](../src_jax/export_celebahq_hf.py):
  export the Hugging Face dataset `eurecom-ds/celeba-hq-256` into the `ImageFolder`
  layout still expected by the current JAX training notebooks
- [src_jax/export_celebahq_tfds.py](../src_jax/export_celebahq_tfds.py):
  export TFDS `celeb_a_hq/256` into the `ImageFolder` layout still expected by
  the current JAX training notebooks, while forcing the Python protobuf runtime
  before importing TFDS to avoid Kaggle descriptor crashes

### FID Utilities

- [src/build_fid_stats.py](../src/build_fid_stats.py): build
  reference `mu` and `sigma`
- [src_jax/build_fid_stats.py](../src_jax/build_fid_stats.py): build
  backend-native reference stats with the same Flax Inception detector used by
  JAX online FID
- [src/evaluate_fid.py](../src/evaluate_fid.py): evaluate a
  generated archive against reference stats
- [src/utils/fid_utils.py](../src/utils/fid_utils.py): reusable
  FID/statistics functions

### JAX Adapter Layer

The JAX path is intentionally kept thin:

- [src_jax/vendor.py](../src_jax/vendor.py): bootstraps
  `diffuse_nnx` into `~/.cache/rae_jax/diffuse_nnx` and pins commit
  `023afd23c7b62a8cdb00e840b36a4ab8fc970bba`
- [src_jax/config_adapter.py](../src_jax/config_adapter.py):
  translates the repository's OmegaConf YAML into the backend config expected
  by NNX, mapping `SiTDH` to the backend `lightning_ddt` network while keeping
  the `sit` training interface, forwarding `random_flip` plus prefetch knobs,
  and defaulting CelebA-HQ train configs to horizontal flips unless overridden
- [src_jax/stage2_runtime.py](../src_jax/stage2_runtime.py):
  training, checkpoint loading, sampling, guidance wiring, JAX validation-loss
  integration, optional horizontal flips on the raw-image path, and FID glue
  for both EMA and optional online-model diagnostics
- [src_jax/stage1_runtime.py](../src_jax/stage1_runtime.py):
  shared JAX Stage 1 encoder loading, single-image reconstruction, folder reconstruction, and latent-stat accumulation
- [src_jax/export_celebahq_hf.py](../src_jax/export_celebahq_hf.py):
  prepares the public Hugging Face CelebA-HQ source into a repo-compatible
  `ImageFolder` tree for Kaggle and local JAX workflows without manual tar
  files
- [src_jax/export_celebahq_tfds.py](../src_jax/export_celebahq_tfds.py):
  prepares the manual TFDS CelebA-HQ source into a repo-compatible
  `ImageFolder` tree for Kaggle and local JAX workflows
- [src_jax/hf_utils.py](../src_jax/hf_utils.py): Hugging Face upload
  helpers for finished workdirs or checkpoint folders

## XLA-Specific Execution Model

### Parallelism

The distributed entrypoints use `xmp.spawn(...)` and run one process per TPU
core. Data parallelism is handled manually through:

- `DistributedSampler`
- `torch_xla.distributed.parallel_loader.ParallelLoader`
- `xm.optimizer_step(...)`
- `xm.rendezvous(...)` and `xm.mesh_reduce(...)`

### Compilation Cache

[src/utils/train_utils.py](../src/utils/train_utils.py) initializes
an XLA compilation cache under fixed paths for training vs sampling.

### Manual Euler Sampling

The XLA branch uses a manual Euler sampler in
[src/utils/sample_utils.py](../src/utils/sample_utils.py) because
`torchdiffeq` is not used in the TPU path here. Each sampling step explicitly
calls `xm.mark_step()` to keep XLA execution progressing.

### Eval and FID Split

There are now two evaluation layers in Stage 2 training:

- validation loss on TPU, using a held-out `ImageFolder`
- optional FID, where sample generation still happens on TPU, but Inception
  feature extraction runs on the host CPU or GPU

That means train-time FID is available on CPU-only TPU VMs as long as the host
has enough RAM and CPU throughput.

On the JAX path, online FID uses the backend Flax Inception detector instead of
the host-side `torch-fidelity` path, so matching `fid_ref` files should be
built with [src_jax/build_fid_stats.py](../src_jax/build_fid_stats.py).

## Code Map

```text
configs/
  stage1/
  stage2/
src/
  stage1/
    encoders/
    decoders/
    rae.py
  stage2/
    models/
    transport/
  utils/
    train_utils.py
    optim_utils.py
    sample_utils.py
    wandb_utils.py
    fid_utils.py
  train.py
  sample.py
  sample_ddp.py
  stage1_sample.py
  stage1_sample_ddp.py
  build_fid_stats.py
  evaluate_fid.py
src_jax/
  vendor.py
  config_adapter.py
  stage2_runtime.py
  stage1_runtime.py
  build_stage1_stats.py
  export_celebahq_hf.py
  export_celebahq_tfds.py
  reconstruct_folder.py
  hf_utils.py
  train.py
  sample.py
  sample_ddp.py
  stage1_sample.py
  push_hf.py
raes-jax-celebahq-kaggle.ipynb
raes-jax-celebahq-kaggle-tpuv5e8-sitdh-s.ipynb
raes-jax-celebahq-kaggle-tpuv5e8-sitdh-b.ipynb
raes-jax-celebahq-kaggle-tpuv5e8-sitdh-b-resume.ipynb
```

## Checkpoint Compatibility

On this branch, `stage_2.ckpt` must already be compatible with the two-tower
SiTDH/DiTwDDTHead shape.

- PyTorch `.pt` checkpoints are supported when they come from a SiTDH-compatible
  model definition.
- Orbax directories are supported when they come from previous SiTDH JAX runs.
- Legacy pre-DH SiTDH checkpoints are not auto-converted on this branch.
- Stage 1 decoder checkpoints remain reusable on both the PyTorch and JAX paths.

## Experiment Artifacts

Stage 2 training writes artifacts under `results/<experiment_name>/`:

- `log.txt`
- `checkpoints/<step>.pt`
- `fid_eval/step_<step>/...` when online FID is enabled

Distributed sampling writes:

- per-image PNG files into the chosen sample directory
- a packed `.npz` archive next to that directory
- optional `.fid.json` result when `--fid-ref` is supplied

JAX runs typically write:

- `results_jax/<experiment_name>/jax_adapter_config.json`
- Orbax checkpoint directories managed by the backend
- optional wandb media and FID outputs
- optional Hugging Face upload of the full workdir

## What To Read Next

- Use [Workflows](./workflows.md) for run commands and operational guidance.
- Use [Config Reference](./config-reference.md) while editing YAML files.
