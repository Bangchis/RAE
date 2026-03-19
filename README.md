## Diffusion Transformers with Representation Autoencoders (RAE)<br><sub>Official PyTorch Implementation</sub>

### [Paper](https://arxiv.org/abs/2510.11690) | [Project Page](https://rae-dit.github.io/) 


This repository contains **PyTorch/GPU** and **TorchXLA/TPU** implementations of our paper: 
Diffusion Transformers with Representation Autoencoders. For JAX/TPU implementation, please refer to [diffuse_nnx](https://github.com/willisma/diffuse_nnx)

> [**Diffusion Transformers with Representation Autoencoders**](https://arxiv.org/abs/2510.11690)<br>
> [Boyang Zheng](https://bytetriper.github.io/), [Nanye Ma](https://willisma.github.io), [Shengbang Tong](https://tsb0601.github.io/),  [Saining Xie](https://www.sainingxie.com)
> <br>New York University<br>


We present Representation Autoencoders (RAE), a class of autoencoders that utilize  pretrained, frozen representation encoders such as [DINOv2](https://arxiv.org/abs/2304.07193) and [SigLIP2](https://arxiv.org/abs/2502.14786) as encoders with trained ViT decoders. RAE can be used in a two-stage training pipeline for high-fidelity image synthesis, where a Stage 2 diffusion model is trained on the latent space of a pretrained RAE to generate images.

This branch contains:

TorchXLA/TPU:
* A TPU implementation of RAE and pretrained weights.
* Sampling of RAE and DiT<sup>DH</sup> on TPU.

JAX/NNX:
* A lightweight JAX/NNX compatibility layer under `src_jax/`.
* Stage 2 train/sample entrypoints that reuse the existing YAML schema.
* Weights & Biases logging, Hugging Face upload, and optional FID scoring without duplicating the full PyTorch codebase.

## Documentation

Use the docs folder as the detailed guide for this branch:

- [docs/README.md](docs/README.md): documentation index
- [docs/architecture.md](docs/architecture.md): architecture and code map
- [docs/workflows.md](docs/workflows.md): practical runbooks for XLA and JAX/NNX training, sampling, and FID
- [docs/config-reference.md](docs/config-reference.md): YAML schema reference
- [pdf/main.pdf](pdf/main.pdf): detailed Vietnamese PDF for architecture, workflow, config, and operations
- [raes-jax-celeba-kaggle.ipynb](raes-jax-celeba-kaggle.ipynb): Kaggle notebook for the standard CelebA JAX flow, using a dedicated `uv` virtualenv plus `uv run`
- [raes-jax-celeba-kaggle-tpuv5e8.ipynb](raes-jax-celeba-kaggle-tpuv5e8.ipynb): Kaggle notebook tuned for `TPU v5e-8` with a dedicated `uv` virtualenv, host-side CPU FID, and a fresh-process TPU sanity check

## Environment

### Dependency Setup
1. Create environment and install via `uv`:
   ```bash
   conda create -n rae python=3.10 -y
   conda activate rae
   pip install uv
   
   # Install PyTorch 2.2.0 with CUDA 12.1
   uv pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 torchvision==0.20.1 -f https://storage.googleapis.com/libtpu-releases/index.html
   
   # Install other dependencies
   uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb scipy torch-fidelity
   uv pip install "numpy<2" transformers einops
   ```

2. If you want to use the JAX/NNX adapter in `src_jax/`, also install:
   ```bash
   uv pip install "jax[cuda12]==0.5.1" flax==0.10.4 optax==0.2.4 orbax-checkpoint==0.11.16
   uv pip install ml-collections clu absl-py etils huggingface_hub
   ```

   Notes:
   - On TPU, replace `jax[cuda12]` with the TPU wheel flow you already use in your environment.
   - `src_jax/` pins `diffuse_nnx` at commit `023afd23c7b62a8cdb00e840b36a4ab8fc970bba` and bootstraps it into `~/.cache/rae_jax/diffuse_nnx` on first run.

## Data & Model Preparation

### Download Pre-trained Models

We release three kind of models: RAE decoders, DiT<sup>DH</sup> diffusion transformers and stats for latent normalization. To download all models at once:


```bash

cd RAE
pip install huggingface_hub
hf download nyu-visionx/RAE-collections \
  --local-dir models 
```


To download specific models, run:
```bash
hf download nyu-visionx/RAE-collections \
  <remote_model_path> \
  --local-dir models 
```

### Prepare Dataset

1. Download ImageNet-1k.
2. Point Stage 1 and Stage 2 scripts to the training split via `--data-path`.


## Config-based Initialization

All training and sampling entrypoints are driven by OmegaConf YAML files. A
single config describes the Stage 1 autoencoder, the Stage 2 diffusion model,
and the solver used during training or inference. A minimal example looks like:

```yaml
stage_1:
   target: stage1.RAE
   params: { ... }
   ckpt: <path_to_ckpt>  

stage_2:
   target: stage2.models.DDT.DiTwDDTHead
   params: { ... }
   ckpt: <path_to_ckpt>  

transport:
   params:
      path_type: Linear
      prediction: velocity
      ...
sampler:
   mode: ODE
   params:
      num_steps: 50
      ...
guidance:
   method: cfg/autoguidance
   scale: 1.0
   ...
misc:
   latent_size: [768, 16, 16]
   num_classes: 1000
training:
   ...

# Optional online validation during Stage 2 training.
eval:
   ...
```

- `stage_1` instantiates the frozen encoder and trainable decoder. For Stage 1
  training you can point to an existing checkpoint via `stage_1.ckpt` or start
  from `pretrained_decoder_path`.
- `stage_2` defines the diffusion transformer. During sampling you must provide
  `ckpt`; during training you typically omit it so weights initialise randomly.
- `transport`, `sampler`, and `guidance` select the forward/backward SDE/ODE
  integrator and optional classifier-free or autoguidance schedule.
- `misc` collects shapes, class counts, and scaling constants used by both
  stages.
- `training` contains defaults that the training scripts consume (epochs,
  learning rate, EMA decay, gradient accumulation, etc.).
- `eval` is optional and enables TPU-native validation loss during Stage 2
  training without running FID online.

Stage 1 training configs additionally include a top-level `gan` block that
configures the discriminator architecture and the LPIPS/GAN loss schedule.


### Provided Configs:

#### Stage1

We release decoders for DINOv2-B, SigLIP-B, MAE-B, at `configs/stage1/pretrained/`.

There is also a training script for training a ViT-XL decoder on DINOv2-B: `configs/stage1/training/DINOv2-B_decXL.yaml`

#### Stage2

We release our best model, DiT<sup>DH</sup>-XL and it's guidance model on both $256\times 256$ and $512\times 512$, at `configs/stage2/sampling/`.

We also provide training configs for DiT<sup>DH</sup> at `configs/stage2/training/`.

## Stage 1: Representation Autoencoder

### Sampling/Reconstruction

Use `src/stage1_sample.py` to encode/decode a single image:

```bash
python src/stage1_sample.py \
  --config <config> \
  --image assets/pixabay_cat.png \
```

For batched reconstructions and `.npz` export, run the XLA native DDP variant:

```bash
  python src/stage1_sample_ddp.py \
  --config <config> \
  --data-path <imagenet_val_split> \
  --sample-dir recon_samples \
  --image-size 256
```

The script writes per-image PNGs as well as a packed `.npz` suitable for FID.

## Stage 2: Latent Diffusion Transformer


For sampling, XLA branch only supports a manually implemented Euler sampler as `torchdiffeq` is not compatible with TPU.

### Training

Train Stage 2 on TPU with:

```bash
python src/train.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml \
  --data-path <imagenet_train_split> \
  --results-dir results \
  --image-size 256 \
  --precision bf16
```

To enable Weights & Biases logging, set:

```bash
export ENTITY=<wandb_entity>
export PROJECT=<wandb_project>
export WANDB_KEY=<wandb_api_key>
```

and add `--wandb` to the training command.

Stage 2 training now logs the following namespaces:

- `train/*`: loss, learning rate, optimizer steps/sec, images/sec, epoch, and gradient norm when clipping is enabled.
- `eval/*`: periodic validation loss on a held-out ImageFolder split (`eval/ema_loss` by default, plus `eval/model_loss` when enabled), and optional FID metrics (`eval/ema_fid`, `eval/model_fid`) when configured.
- `checkpoint/*`: checkpoint save step.
- `samples/ema`: EMA preview images logged at the training step.

To enable online validation loss, add an optional `eval` block to the Stage 2 training config:

```yaml
eval:
  data_path: data/imagenet/val/
  eval_every: 5000
  batch_size: 128        # per TPU core; defaults to the train micro batch size
  num_workers: 4         # defaults to training.num_workers
  max_batches: 32        # optional cap per rank to limit eval cost
  eval_model: false      # set true to also evaluate the non-EMA model
  fid_ref: data/imagenet/VIRTUAL_imagenet256_labeled.npz
  fid_every: 25000       # defaults to eval_every when omitted
  fid_num_samples: 4096  # trade off speed vs stability (e.g. 1024 / 4096 / 50000)
  fid_per_proc_batch_size: 4
  fid_batch_size: 64     # Inception batch size on the host CPU/GPU
  fid_device: cpu
  fid_num_threads: 96    # optional torch CPU thread count for host-side FID
  fid_label_sampling: equal
  fid_eval_model: false  # set true to also compute FID for the non-EMA model
```

This XLA branch runs validation loss on TPU inside the training loop. Optional FID
evaluation is also available at eval checkpoints by sampling on TPU and computing
Inception features on the host CPU or GPU.

### JAX / NNX Compatibility Layer

The `jax` branch also ships a thin adapter in `src_jax/` that keeps the current
OmegaConf YAML files, but runs Stage 2 through a JAX/NNX backend instead of
duplicating the entire PyTorch codebase.

Main entrypoints:

```bash
python3 src_jax/train.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml \
  --data-path <imagenet_train_root> \
  --results-dir results_jax \
  --precision bf16 \
  --wandb \
  --set training.global_batch_size=256
```

```bash
python3 src_jax/sample.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B_AG.yaml \
  --output sample_jax.png \
  --class-labels 207,360
```

```bash
python3 src_jax/sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --sample-dir samples_jax \
  --num-samples 50000 \
  --label-sampling equal \
  --fid-ref /path/to/reference_stats.npz
```

```bash
python3 src_jax/stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B_512.yaml \
  --image assets/pixabay_cat.png \
  --output recon_jax.png
```

```bash
python3 src_jax/build_stage1_stats.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --input /path/to/train_imagefolder \
  --output /path/to/stage1_stat.pt \
  --set stage_1.params.normalization_stat_path=/path/to/bootstrap_identity_stat.pt
```

```bash
python3 src_jax/reconstruct_folder.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --input /path/to/val_imagefolder \
  --output-dir recon_jax_dir \
  --batch-size 8
```

```bash
python3 src_jax/push_hf.py \
  --path results_jax/<run_name> \
  --repo-id <hf_user_or_org>/<repo_name>
```

Key behavior:

- `src_jax/` accepts the same top-level YAML blocks: `stage_1`, `stage_2`, `transport`, `sampler`, `guidance`, `misc`, `training`, and `eval`.
- `stage_2.ckpt` can point to the original PyTorch `.pt` checkpoints for inference, or to a JAX Orbax directory for resumed JAX runs.
- `--set key=value` applies OmegaConf CLI overrides without adding a second config format.
- `src_jax/build_stage1_stats.py` writes a PyTorch-compatible `stat.pt` file, so the same Stage 1 normalization stats can be reused by both the original repo code and the JAX adapter.
- for dataset-specific Stage 1 stats, start from a bootstrap identity stats file (`mean=0`, `var=1`) and override `stage_1.params.normalization_stat_path` during the stats pass.
- `ENTITY` / `PROJECT` / `WANDB_KEY` are bridged to the `WANDB_*` variables expected by the JAX backend.
- `--hf-repo-id` on `src_jax/train.py` uploads the finished workdir directly to Hugging Face.
- `raes-jax-celeba-kaggle.ipynb` mirrors the standard Kaggle workflow end to end for CelebA, but keeps package-backed steps inside a dedicated `uv` virtualenv via `uv run` instead of relying on the notebook kernel interpreter.
- `raes-jax-celeba-kaggle-tpuv5e8.ipynb` copies that flow for `TPU v5e-8`, switches the install path to `jax[tpu]`, keeps the heavy steps inside a dedicated `uv` virtualenv via `uv run`, verifies TPU visibility in a fresh Python process, and moves FID/stat-heavy host work onto the `96 vCPU` side.

Current limitation:

- The JAX path intentionally focuses on Stage 2 training/sampling and Stage 1 reconstruction. A dedicated JAX port of the adversarial Stage 1 decoder training loop is not shipped yet, because porting LPIPS + GAN + discriminator into JAX would make the branch substantially larger and harder to maintain.

### Sampling

`src/sample.py` uses the same config schema to draw a small batch of images on a
single device and saves them to `sample.png`:

```bash
python src/sample.py \
  --config <sample_config> \
  --seed 42
```


### Distributed sampling for evaluation

`src/sample_ddp.py` parallelises sampling across TPU cores, producing PNGs and an
FID-ready `.npz`:

```bash
python src/sample_ddp.py \
  --config <sample_config> \
  --sample-dir samples \
  --precision bf16 \
  --label-sampling equal
```
`--label-sampling {equal,random}`: `equal` uses exactly 50 images per class for FID-50k; `random` uniformly samples labels. Using `equal` brings consistently lower FID than `random` by around 0.1. We use `equal` by default.

Autoguidance and classifier-free guidance are controlled via the config’s
`guidance` block.

To compute FID immediately after sampling on the same TPU VM, point `sample_ddp.py`
to reference statistics:

```bash
python src/sample_ddp.py \
  --config <sample_config> \
  --sample-dir samples \
  --precision bf16 \
  --label-sampling equal \
  --fid-ref /path/to/VIRTUAL_imagenet256_labeled.npz \
  --fid-device cpu
```

This still samples on TPU, but the Inception feature extraction for FID runs on
the host CPU or GPU (`--fid-device auto|cpu|cuda`). You can also set
`--fid-num-threads` to control CPU parallelism explicitly. Rank 0 writes a
`<sample_dir>.fid.json` file with the result.

## Evaluation

### Online evaluation during training (TPU + host CPU/GPU)

The `eval` block above runs held-out validation loss inside `src/train.py` and
sends the resulting scalars to wandb when `--wandb` is enabled. If `fid_ref` is
set, the same block can also trigger periodic FID evaluation. Sampling still
runs on TPU; the Inception feature extraction runs on the host CPU or GPU.

### Local FID from generated `.npz`

For custom datasets, first build reference statistics once:

```bash
python src/build_fid_stats.py \
  --input /path/to/reference_images \
  --output /path/to/reference_stats.npz \
  --device cpu \
  --num-workers 32 \
  --num-threads 96
```

`--input` accepts either an image folder (recursively scanned) or an existing
`.npy`/`.npz` image archive.

If you already have a generated `.npz`, you can score it directly with:

```bash
python src/evaluate_fid.py \
  --samples /path/to/samples.npz \
  --ref /path/to/VIRTUAL_imagenet256_labeled.npz \
  --device cpu \
  --num-threads 96 \
  --output-json /path/to/samples.fid.json
```

This path uses `torch-fidelity`'s InceptionV3-compatible feature extractor and
works on CPU or CUDA. It avoids moving samples to another machine, but it is not
the ADM TensorFlow evaluator.

### ADM Suite FID setup (only available on GPU)

Use GPU with the ADM evaluation suite to score generated samples. You need to port the npz file generated from TPU to a GPU machine.

1. Clone the repo:

   ```bash
   git clone https://github.com/openai/guided-diffusion.git
   cd guided-diffusion/evaluation
   ```

2. Create an environment and install dependencies:

   ```bash
   conda create -n adm-fid python=3.10
   conda activate adm-fid
   pip install 'tensorflow[and-cuda]'==2.19 scipy requests tqdm
   ```

3. Download ImageNet statistics (256×256 shown here):

   ```bash
   wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
   ```

4. Evaluate:

   ```bash
   python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples.npz
   ```
