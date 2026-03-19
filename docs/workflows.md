# Workflows

## 1. Environment Setup

The repository supports two practical runtime stacks:

- `torch_xla` for the existing TPU branch under `src/`
- `jax` / `flax.nnx` for the compatibility layer under `src_jax/`

Baseline XLA dependencies:

Baseline install flow:

```bash
conda create -n rae python=3.10 -y
conda activate rae
pip install uv
uv pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 torchvision==0.20.1 -f https://storage.googleapis.com/libtpu-releases/index.html
uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb scipy torch-fidelity
uv pip install "numpy<2" transformers einops
```

JAX / NNX additions for `src_jax/`:

```bash
uv pip install "jax[cuda12]==0.5.1" flax==0.10.4 optax==0.2.4 orbax-checkpoint==0.11.16
uv pip install ml-collections clu absl-py etils huggingface_hub
```

The first JAX run automatically bootstraps `diffuse_nnx` into
`~/.cache/rae_jax/diffuse_nnx` and pins it to commit
`023afd23c7b62a8cdb00e840b36a4ab8fc970bba`.

## 2. Prepare Models and Data

### Model Weights

Download weights into `models/`:

```bash
hf download nyu-visionx/RAE-collections --local-dir models
```

### Dataset Format

All current dataset-consuming scripts assume an `ImageFolder` layout:

```text
dataset_root/
  class_a/
    0001.png
  class_b/
    0002.png
```

For unlabeled one-class datasets, you still need one subdirectory, for example:

```text
celeba256_imgfolder/
  train/
    face/
      ...
  val/
    face/
      ...
```

## 3. Reconstruct Images with Stage 1

### Single Image

```bash
python src/stage1_sample.py \
  --config <config> \
  --image assets/pixabay_cat.png \
  --output recon.png
```

Use this when you want to verify the Stage 1 checkpoint, latent shape, or input
preprocessing.

### Distributed Reconstruction

```bash
python src/stage1_sample_ddp.py \
  --config <config> \
  --data-path <imagefolder_root> \
  --sample-dir recon_samples \
  --image-size 256 \
  --per-proc-batch-size 4 \
  --save-npz
```

Outputs:

- PNG reconstructions
- optional `.npz` archive for later scoring

## 4. Train Stage 2 on TPU

Main entrypoint:

```bash
python src/train.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml \
  --data-path <imagenet_train_split> \
  --results-dir results \
  --image-size 256 \
  --precision bf16
```

What happens inside the loop:

1. images are center-cropped and converted to tensors
2. the RAE encodes them to latent tensors
3. the transport loss is computed in latent space
4. the optimizer and scheduler step on TPU
5. the EMA model is updated
6. logging, checkpointing, preview sampling, validation, and FID run on their
   configured cadences

### Important Training Outputs

- `results/<exp>/log.txt`
- `results/<exp>/checkpoints/*.pt`
- `results/<exp>/fid_eval/step_*/` when FID is enabled

### Resume Training

```bash
python src/train.py \
  --config <config> \
  --data-path <train_split> \
  --results-dir results \
  --image-size 256 \
  --precision bf16 \
  --ckpt results/<exp>/checkpoints/0050000.pt
```

The loop restores:

- model
- EMA
- optimizer
- scheduler
- `epoch`
- `train_steps`

Cadence checks such as `log_every`, `eval_every`, `fid_every`, `sample_every`,
and `ckpt_every` continue from the restored step count.

## 4b. Train Stage 2 with JAX / NNX

Main JAX entrypoint:

```bash
python3 src_jax/train.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml \
  --data-path <imagenet_train_root> \
  --results-dir results_jax \
  --precision bf16 \
  --wandb
```

Useful additions:

- `--set training.global_batch_size=256`: override YAML values from the CLI
- `--hf-repo-id <user>/<repo>`: upload the finished workdir to Hugging Face
- `--workdir <path>`: force an explicit output directory instead of letting the
  adapter derive one from `--results-dir`

Resume / initialize behavior:

- If `stage_2.ckpt` points to a PyTorch `.pt`, the adapter ports those weights
  into the NNX model for initialization.
- If `stage_2.ckpt` points to an Orbax directory from a previous JAX run, the
  adapter restores that JAX checkpoint instead.

## 5. Enable wandb Logging

Set:

```bash
export ENTITY=<wandb_entity>
export PROJECT=<wandb_project>
export WANDB_KEY=<wandb_api_key>
```

Then add `--wandb` to `src/train.py`.

The same environment variables are also honored by `src_jax/train.py`. The JAX
adapter bridges the legacy `ENTITY`, `PROJECT`, and `WANDB_KEY` names into the
`WANDB_*` names expected by the NNX backend.

Current Stage 2 namespaces:

- `train/*`
- `eval/*`
- `checkpoint/*`
- `samples/ema`
- `sample/duration_sec`

Only the master rank initializes and logs to wandb.

## 6. Validation Loss During Training

Add an `eval` block:

```yaml
eval:
  data_path: data/imagenet/val/
  eval_every: 5000
  batch_size: 128
  num_workers: 4
  max_batches: 32
  eval_model: false
```

Behavior:

- runs a deterministic center-crop validation loader
- evaluates EMA by default
- optionally evaluates the non-EMA model as well
- uses distributed weighting so the global mean stays correct even when the
  validation set is not divisible by world size

## 7. FID During Training

Train-time FID is optional and uses two execution domains:

- TPU: latent sampling and image decoding
- host CPU or GPU: Inception feature extraction

Config:

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

### Meaning of the Main Knobs

- `fid_every`: FID cadence in optimizer steps
- `fid_num_samples`: sample budget per FID measurement
- `fid_per_proc_batch_size`: generation batch per TPU core
- `fid_batch_size`: host-side Inception batch size
- `fid_device`: `cpu`, `cuda`, or `auto`
- `fid_num_threads`: only matters on CPU, maps to `torch.set_num_threads`
- `fid_label_sampling`: `equal` or `random`
- `fid_eval_model`: also score the non-EMA model

### Recommended CPU-Only Starting Point

For a TPU VM with a strong CPU host and no GPU:

```yaml
eval:
  fid_ref: /path/to/reference_stats.npz
  fid_every: 25000
  fid_num_samples: 4096
  fid_per_proc_batch_size: 4
  fid_batch_size: 128
  fid_device: cpu
  fid_num_threads: 96
```

### Cost vs Stability

- `fid_num_samples: 1024` is fast, but noisy
- `fid_num_samples: 4096` is a practical online-monitoring compromise
- `fid_num_samples: 50000` is the closest to standard FID-50k, but expensive

Do not compare FID across runs unless the sample count and reference stats are
the same.

## 8. Sample Stage 2 Outputs

### Quick Single-Process Sample

```bash
python src/sample.py \
  --config <sample_config> \
  --seed 42 \
  --class-labels 207,360 \
  --output sample.png
```

### Distributed Sampling

```bash
python src/sample_ddp.py \
  --config <sample_config> \
  --sample-dir samples \
  --per-proc-batch-size 4 \
  --num-fid-samples 50000 \
  --precision bf16 \
  --label-sampling equal
```

### JAX Single-Run Sampling

```bash
python3 src_jax/sample.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B_AG.yaml \
  --class-labels 207,360 \
  --output sample_jax.png
```

Notes:

- `guidance.method=cfg` uses the same model for conditional and unconditional
  passes.
- `guidance.method=autoguidance` loads `guidance.guidance_model` as the guide
  network.

### JAX Distributed Sampling

```bash
python3 src_jax/sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --sample-dir samples_jax \
  --num-samples 50000 \
  --label-sampling equal \
  --fid-ref /path/to/reference_stats.npz
```

Outputs:

- per-image PNG files under `samples_jax/`
- optional per-rank `.npz` archives when `--save-npz` is enabled
- optional printed FID score when `--fid-ref` is supplied

### Stage 1 Reconstruction on the JAX Path

```bash
python3 src_jax/stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B_512.yaml \
  --image assets/pixabay_cat.png \
  --output recon_jax.png
```

This is intentionally scoped to inference and checkpoint verification. The
adversarial Stage 1 training loop has not been ported to JAX in this branch.

### Stage 1 Latent Stats on the JAX Path

For a new dataset, first bootstrap normalization with an identity stats file,
then run the JAX stats pass:

```bash
python3 src_jax/build_stage1_stats.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --input /path/to/train_imagefolder \
  --output /path/to/stage1_stat.pt \
  --batch-size 16 \
  --set stage_1.params.normalization_stat_path=/path/to/bootstrap_identity_stat.pt
```

The output `stat.pt` matches the original repo format (`mean` / `var` tensors),
so it can be consumed by both `src/` and `src_jax/`.

### JAX Folder Reconstruction

```bash
python3 src_jax/reconstruct_folder.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --input /path/to/val_imagefolder \
  --output-dir recon_jax_dir \
  --batch-size 8
```

This is the simplest way to export a validation reconstruction set before
building FID references or comparing Stage 1 decoder changes.

### Kaggle CelebA Notebook

Use [../raes-jax-celeba-kaggle.ipynb](../raes-jax-celeba-kaggle.ipynb) when you
want the standard Kaggle-style workflow end to end:

- clone the repo and checkout `jax`
- create a dedicated `.venv` with `uv`
- run package-backed data/stat/reconstruction/train steps through `uv run`
- convert CelebA into a real `256x256` `ImageFolder`
- create the bootstrap identity stats file
- compute Stage 1 latent stats for CelebA
- export Stage 1 reconstructions and build validation FID stats
- write a CelebA Stage 2 config and launch `src_jax/train.py`

For Kaggle `TPU v5e-8`, use
[../raes-jax-celeba-kaggle-tpuv5e8.ipynb](../raes-jax-celeba-kaggle-tpuv5e8.ipynb).
That copy switches installation to `jax[tpu]`, keeps the package-backed steps
inside a dedicated `uv` virtualenv via `uv run`, clears the `jaxlib`
executable-stack flag that Kaggle can reject before each JAX import, runs the
TPU device check in a fresh Python process, keeps Stage 1 and Stage 2 on TPU,
and moves FID/stat-heavy work to the host CPU side with `96` threads.

## 9. Upload a JAX Run to Hugging Face

Standalone upload:

```bash
python3 src_jax/push_hf.py \
  --path results_jax/<run_name> \
  --repo-id <hf_user_or_org>/<repo_name>
```

Or upload automatically at the end of training:

```bash
python3 src_jax/train.py \
  --config <config> \
  --data-path <train_root> \
  --hf-repo-id <hf_user_or_org>/<repo_name>
```

This produces:

- PNG files
- a packed `.npz` archive

### Distributed Sampling with Immediate FID

```bash
python src/sample_ddp.py \
  --config <sample_config> \
  --sample-dir samples \
  --per-proc-batch-size 4 \
  --num-fid-samples 4096 \
  --precision bf16 \
  --label-sampling random \
  --fid-ref /path/to/reference_stats.npz \
  --fid-device cpu \
  --fid-batch-size 128 \
  --fid-num-threads 96
```

## 9. Build FID Reference Statistics

From an image folder:

```bash
python src/build_fid_stats.py \
  --input /path/to/reference_images \
  --output /path/to/reference_stats.npz \
  --device cpu \
  --batch-size 64 \
  --num-workers 32 \
  --num-threads 96
```

From an existing `.npz` or `.npy` archive:

```bash
python src/build_fid_stats.py \
  --input /path/to/reference_images.npz \
  --output /path/to/reference_stats.npz \
  --device cpu \
  --batch-size 64 \
  --num-threads 96
```

## 10. Evaluate an Existing Archive

```bash
python src/evaluate_fid.py \
  --samples /path/to/samples.npz \
  --ref /path/to/reference_stats.npz \
  --device cpu \
  --batch-size 128 \
  --num-threads 96 \
  --output-json /path/to/samples.fid.json
```

## 11. Common Operational Notes

- `src/train.py` expects the Stage 2 training config to use `training.global_batch_size`,
  not `batch_size`.
- `src/train.py` reads `full_cfg.get("eval")` directly, so the eval block does
  not need to be part of `parse_configs(...)`.
- `src/sample.py` and `src/sample_ddp.py` only support the manual ODE path on
  this branch.
- Stage 1 reconstruction scripts are for inference and inspection; this branch
  does not include a local Stage 1 trainer entrypoint.
- For CPU-only FID, increase `fid_batch_size` cautiously. Throughput improves
  until memory bandwidth becomes the bottleneck.
