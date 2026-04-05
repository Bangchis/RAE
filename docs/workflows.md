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
uv pip install "numpy<2" "transformers==4.57.1" einops
```

JAX / NNX additions for `src_jax/`:

```bash
uv pip install "jax[cuda12]==0.5.1" flax==0.10.4 optax==0.2.4 orbax-checkpoint==0.11.16
uv pip install ml-collections clu absl-py etils datasets huggingface_hub tensorflow-datasets
```

The first JAX run automatically bootstraps `diffuse_nnx` into
`~/.cache/rae_jax/diffuse_nnx` and pins it to commit
`023afd23c7b62a8cdb00e840b36a4ab8fc970bba`.
The Kaggle JAX notebooks do not replay those package lists manually anymore:
they set `UV_PROJECT_ENVIRONMENT=/tmp/.venv`, `UV_CACHE_DIR=/tmp/uv-cache`, and
run `uv sync -q` against the repo `pyproject.toml` before the package-backed
cells.
This repo patches the pinned `diffuse_nnx` checkout to import the Dinov2 models
from `transformers` subpackages, so use `transformers==4.57.1` on this path.
The same patch also lazy-loads `google-cloud-storage`, so the RAE/DINO Stage 1
path does not need that package unless you actually use backend code that pulls
assets from GCS.
The bootstrap patch also fixes the backend EMA initialization so the EMA starts
from a copy of the live model instead of an all-zero parameter tree. Existing
Orbax checkpoints keep the EMA state they already saved, so use a fresh run if
you need the corrected EMA trajectory end to end.
The adapter also derives the Stage-1 latent downsample factor from
`misc.latent_size`, so backend preview sampling and FID stay in latent space
instead of allocating image-resolution latent noise.

## 2. Prepare Models and Data

### Model Weights

Download weights into `models/`:

```bash
hf download nyu-visionx/RAE-collections --local-dir models
```

### Dataset Format

All current dataset-consuming scripts still assume an `ImageFolder` layout:

```text
dataset_root/
  class_a/
    0001.png
  class_b/
    0002.png
```

For unlabeled one-class datasets, you still need one subdirectory, for example
after exporting CelebA-HQ into `ImageFolder`:

```text
celebahq256_imgfolder/
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
  --config configs/stage2/training/ImageNet256/SiTDH-XL_DINOv2-B.yaml \
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
  --config configs/stage2/training/ImageNet256/SiTDH-XL_DINOv2-B.yaml \
  --data-path <imagenet_train_root> \
  --results-dir results_jax \
  --precision bf16 \
  --wandb
```

On Kaggle TPU, start from `--set training.num_workers=1` and then raise
`--set training.prefetch_factor=<n>` before jumping straight to a large worker
pool. This keeps the backend PyTorch loader compatible with
`persistent_workers=True` after JAX has already initialized multithreaded
runtime state, while still letting the host queue several ready batches ahead
of the TPU. The adapter also disables the backend TensorBoard summary writer on
Kaggle and keeps metric logging on stdout plus wandb.

Useful additions:

- `--set training.global_batch_size=256`: override YAML values from the CLI
- `--set training.prefetch_factor=8`: deepen the host-side train prefetch queue
- `--set eval.prefetch_factor=4`: do the same for the validation loader
- `--wandb-run-id <existing_run_id>`: bind a legacy JAX resume workdir to the exact historical W&B run once, then persist that binding inside the workdir
- `--hf-repo-id <user>/<repo>`: upload the finished workdir to Hugging Face
- `--workdir <path>`: force an explicit output directory instead of letting the
  adapter derive one from `--results-dir`

Resume / initialize behavior:

- If `stage_2.ckpt` points to a PyTorch `.pt`, the adapter ports those weights
  into the NNX model for initialization.
- If `stage_2.ckpt` points to an Orbax directory from a previous JAX run, the
  adapter restores that JAX checkpoint instead.
- When `--wandb` is enabled, a fresh JAX workdir writes `wandb_run.json` and
  starts with a fresh W&B run ID plus `resume="never"`.
- Later resumes of the same JAX workdir read `wandb_run.json`, keep that exact
  W&B run ID, and auto-rewind W&B history to the latest `checkpoint_*` step via
  `resume_from`, so post-checkpoint logs from a crashed session do not survive
  into the resumed history.
- If a legacy Orbax workdir already has checkpoints but predates
  `wandb_run.json`, pass `--wandb-run-id <existing_run_id>` once so the
  adapter can persist the exact historical run binding before training
  continues.

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
On the JAX path, `wandb_run.json` inside the workdir is now the source of truth
for exact resume behavior. A later launch must match that stored binding, and
the adapter rewinds the W&B run to the latest checkpoint step before logging
continues.

Current Stage 2 namespaces:

- `train/*`
- `eval/*`
- `checkpoint/*`
- `network_samples`
- `ema_network_samples`
- `sample/duration_sec`

If you enable `training.log_rae_latent_stats: true`, the JAX path also logs
`train_rae_latent_rms` and `train_rae_latent_var`.
If you enable `training.log_activation_stats: true`, it additionally logs
`train_sitdh_output_rms`, `train_sitdh_output_var`, and one RMS/variance pair
per encoder/decoder block such as `train_sitdh_act_enc_00_rms` and
`train_sitdh_act_dec_01_var`.

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
- logs `eval/ema_loss`, `eval/ema_batches`, `eval/ema_samples`, and
  `eval/ema_duration_sec`
- logs the matching `eval/model_*` metrics when `eval_model: true`

## 7. FID During Training

Train-time FID is optional and uses two execution domains:

- TPU: latent sampling and image decoding
- host CPU or GPU: Inception feature extraction

Config:

```yaml
eval:
  fid_ref: /path/to/reference_stats.pkl
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

Behavior on the JAX path:

- the default `FID-4K (cfg=...)` series measures EMA
- `fid_eval_model: true` adds `FID-4K/model (cfg=...)` for the live model
- `network_samples` and `ema_network_samples` therefore no longer refer to the
  same weights once EMA starts diverging from the online model

### Recommended CPU-Only Starting Point

For a TPU VM with a strong CPU host and no GPU:

```yaml
eval:
  fid_ref: /path/to/reference_stats.pkl
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
  --config configs/stage2/sampling/ImageNet256/SiTDHXL-DINOv2-B_AG.yaml \
  --class-labels 207,360 \
  --output sample_jax.png
```

Notes:

- On this branch, the checked-in JAX sampling configs are templates; set
  `stage_2.ckpt` and `guidance.guidance_model.ckpt` to SiTDH-compatible weights
  before running sampling.
- `guidance.method=cfg` uses the same model for conditional and unconditional
  passes.
- `guidance.method=autoguidance` loads `guidance.guidance_model` as the guide
  network.

### JAX Distributed Sampling

```bash
python3 src_jax/sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/SiTDHXL-DINOv2-B.yaml \
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
These Stage 1-only JAX utilities can run from a YAML that only defines
`stage_1`; they do not require `stage_2.target`.

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

### Kaggle CelebA-HQ Notebook

Use [../raes-jax-celebahq-kaggle.ipynb](../raes-jax-celebahq-kaggle.ipynb) when
you want the standard Kaggle-style workflow end to end:

- clone the repo and checkout `jax-sit-dh-celebahq256`
- set `UV_PROJECT_ENVIRONMENT=/tmp/.venv` and `UV_CACHE_DIR=/tmp/uv-cache`
- run `uv sync -q` from the repo root
- run package-backed data/stat/reconstruction/train steps through `uv run`
- download `eurecom-ds/celeba-hq-256` from Hugging Face and export it into a real `256x256` `ImageFolder`
- create the bootstrap identity stats file
- compute Stage 1 latent stats for CelebA-HQ
- export Stage 1 reconstructions and build validation FID stats
- write a CelebA-HQ Stage 2 config and launch `src_jax/train.py`

The default helper script behind that step is
[`src_jax/export_celebahq_hf.py`](../src_jax/export_celebahq_hf.py). It
downloads the public Hugging Face dataset `eurecom-ds/celeba-hq-256` into a cache
directory, then materializes it into `ImageFolder`, because the current JAX
training/runtime path has not been refactored to consume a Hub dataset
directly. The older
[`src_jax/export_celebahq_tfds.py`](../src_jax/export_celebahq_tfds.py) helper
is still available as a fallback if you specifically want the TFDS
`celeb_a_hq/256` route with manual tar files; that fallback still forces the
Python protobuf runtime before importing TFDS to avoid the common Kaggle
descriptor crash.

The final Stage 2 train cells in the CelebA-HQ notebooks pass
`--data-path /kaggle/working/celebahq256_imgfolder`, while the generated
config keeps `eval.data_path` on `/kaggle/working/celebahq256_imgfolder/val`.
Those notebook train/resume cells also enable
`training.random_flip=true`, keep `eval.random_flip=false`, enable
`--set training.log_rae_latent_stats=true` and
`--set training.log_activation_stats=true` by default so wandb exposes latent
and activation RMS/variance during the run. The TPU notebooks now also set
`--set training.prefetch_factor=8` and `--set eval.prefetch_factor=4` so the
host can queue batches more aggressively without immediately increasing
`training.num_workers`. Expect a measurable throughput and memory cost when
activation logging is enabled, because the backend now returns intermediate
SiTDH features on every train step.

For Kaggle `TPU v5e-8`, use
[../raes-jax-celebahq-kaggle-tpuv5e8-sitdh-s.ipynb](../raes-jax-celebahq-kaggle-tpuv5e8-sitdh-s.ipynb).
That copy fixes the Stage 2 CelebA-HQ variant to `SiTDH-S`, syncs the repo
dependencies into `/tmp/.venv`, clears the `jaxlib` executable-stack flag that
Kaggle can reject before each JAX import, runs the TPU device check in a fresh
Python process, keeps Stage 1 and Stage 2 on TPU, builds the JAX `fid_ref` with
the same backend detector used by online FID, enables
`training.random_flip=true`, and keeps the default checkpoint cadence at
`210000` steps.

If you want the same Kaggle TPU flow but with the `SiTDH-B` DH Stage 2 setup, use
[../raes-jax-celebahq-kaggle-tpuv5e8-sitdh-b.ipynb](../raes-jax-celebahq-kaggle-tpuv5e8-sitdh-b.ipynb).
That notebook keeps the same JAX/TPU workarounds and data prep, but writes the
CelebA-HQ Stage 2 config as `CelebAHQ256_SiTDH-B_DINOv2-B_jax_tpuv5e8.yaml`,
using the DH two-tower layout `hidden_size=[768, 2048]`, `depth=[12, 2]`,
`num_heads=[12, 16]`, enabling `use_pos_embed`, disabling label dropout with
`class_dropout_prob=0.0`, enabling `training.random_flip=true`, and keeping the
same `210000`-step checkpoint cadence.

If you already have an Orbax run directory for `SiTDH-B` and want to continue
training from its latest checkpoint, use
[../raes-jax-celebahq-kaggle-tpuv5e8-sitdh-b-resume.ipynb](../raes-jax-celebahq-kaggle-tpuv5e8-sitdh-b-resume.ipynb).
That notebook is intentionally stripped down for the common Kaggle resume case
where you start from the archived output of the previous notebook. Its first
cell runs `unzip -o /kaggle/input/notebooks/kieuhongquan/rae-jax/_output_.zip
-d /kaggle/working`, then it only checks that `/kaggle/working/RAE` is present,
runs `uv sync`, applies the `jaxlib` executable-stack fix, loads the Kaggle
secret, runs a path sanity-check for the restored dataset/FID/workdir files,
and finishes with `src_jax/train.py --workdir ...`. The notebook first locates
the newest `CelebAHQ256_SiTDH-B_DINOv2-B_jax_tpuv5e8-*` run directory under
`/kaggle/working/results_jax_tpu/`, then both the shell pre-check and the
runtime work from the newest `checkpoint_<step>` directory available under that
workdir, so the notebook no longer hardcodes either the timestamped run folder
or `checkpoint_100000`.

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

For JAX online FID, build the reference with the backend-native detector:

```bash
python3 src_jax/build_fid_stats.py \
  --input /path/to/reference_images \
  --output /path/to/reference_stats.pkl \
  --batch-size 64 \
  --num-workers 8
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

`src_jax/build_fid_stats.py` only accepts an `ImageFolder`, because it must run
the same Flax Inception detector used by the backend online FID path.

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
and finishes with `src_jax/train.py --workdir ...`. If the target workdir was
created before `wandb_run.json` existed, pass
`--wandb-run-id <existing_run_id>` once so the resumed Kaggle session binds to
the exact old W&B run instead of aborting.
