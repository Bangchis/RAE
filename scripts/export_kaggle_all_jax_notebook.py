from __future__ import annotations

import json
from pathlib import Path


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    title = markdown_cell(
        """# RAE CelebA-HQ256 Kaggle TPU v5e-8 Runbook (All-JAX + LightningDiT-B)

Notebook này bám hướng bạn đã chốt:

- Stage 1: config giống `main` nhất có thể (`reconstruction + LPIPS + GAN`) nhưng đi theo all-JAX roadmap
- Stage 2: `LightningDiT-B` trên TFDS CelebA-HQ256, chạy bằng JAX trên Kaggle TPU

Lưu ý:

- Notebook dùng split TFDS có sẵn là `train` và `validation`, không còn tự cắt `train[:95%]`.
- Cell Stage 2 chạy được với branch hiện tại.
- Cell Stage 1 all-JAX hiện là launch contract cho trainer JAX đang được port. Notebook vẫn xuất đủ config và lệnh để bạn giữ một pipeline Kaggle thống nhất.
- Nếu TPU compile hoặc OOM, giảm `STAGE1_BATCH_SIZE` từ `512` xuống `256/128/64/32`, và giảm `STAGE2_BATCH_SIZE` từ `64` xuống `32`.
"""
    )

    env_setup = code_cell(
        """import json
import os
from pathlib import Path

try:
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    wandb_key = secrets.get_secret("WANDB_KEY")
    os.environ["WANDB_API_KEY"] = wandb_key
    os.environ["WANDB_KEY"] = wandb_key

    netrc = Path.home() / ".netrc"
    netrc.write_text(f"machine api.wandb.ai login user password {wandb_key}\\n")
    os.chmod(netrc, 0o600)
    print("Loaded WANDB_KEY from Kaggle Secrets.")
except Exception as exc:
    print(f"Skipping Kaggle secret bootstrap: {exc}")

RAE_REPO_URL = "https://github.com/Bangchis/RAE.git"
RAE_BRANCH = "jax-sit-dh-celebahq256"

INPUT_ROOT = Path("/kaggle/input/shortcut-celebahq-256")
TFDS_SOURCE_DIR = INPUT_ROOT / "tensorflow_datasets"
TFDS_BUILDERS_DIR = INPUT_ROOT / "tfds_builders"

WORK_ROOT = Path("/kaggle/working")
REPO_ROOT = WORK_ROOT / "RAE"
TFDS_WORK_ROOT = WORK_ROOT / "shortcut_celebahq256"
TFDS_DATA_DIR = TFDS_WORK_ROOT / "tensorflow_datasets"
RESULTS_ROOT = WORK_ROOT / "results_all_jax"
ARTIFACTS_ROOT = WORK_ROOT / "artifacts"

STAGE1_RESULTS_DIR = RESULTS_ROOT / "stage1_jax"
STAGE2_RESULTS_DIR = RESULTS_ROOT / "stage2_jax"
STAGE1_KAGGLE_CFG = REPO_ROOT / "configs" / "stage1" / "training" / "CelebAHQ256_DINOv2-B_decB_tfds_kaggle.yaml"
STAGE2_KAGGLE_CFG = REPO_ROOT / "configs" / "stage2" / "training" / "CelebAHQ256" / "LightningDiT-B_DINOv2-B_tfds_kaggle.yaml"

STAGE1_DECODER_EXPORT = ARTIFACTS_ROOT / "celebahq256_stage1_decoder.pt"
STAGE1_STATS_PATH = ARTIFACTS_ROOT / "celebahq256_stage1_stat.pt"
FALLBACK_DECODER_PATH = REPO_ROOT / "models" / "decoders" / "dinov2" / "wReg_base" / "decB_ganv3" / "dinov2_decoder.pt"
DINO_DISC_CKPT = REPO_ROOT / "models" / "discs" / "dino_vit_small_patch8_224.pth"

STAGE1_BATCH_SIZE = 512
STAGE2_BATCH_SIZE = 64
STAGE2_NUM_WORKERS = 1
STAGE2_PREFETCH_FACTOR = 8
STAGE2_CKPT_EVERY = 210000
STAGE2_SAMPLE_EVERY = 5000

os.environ["RAE_REPO_URL"] = RAE_REPO_URL
os.environ["RAE_BRANCH"] = RAE_BRANCH
os.environ["TFDS_SOURCE_DIR"] = str(TFDS_SOURCE_DIR)
os.environ["TFDS_BUILDERS_DIR"] = str(TFDS_BUILDERS_DIR)
os.environ["TFDS_WORK_ROOT"] = str(TFDS_WORK_ROOT)
os.environ["TFDS_DATA_DIR"] = str(TFDS_DATA_DIR)
os.environ["REPO_ROOT"] = str(REPO_ROOT)
os.environ["RESULTS_ROOT"] = str(RESULTS_ROOT)
os.environ["ARTIFACTS_ROOT"] = str(ARTIFACTS_ROOT)
os.environ["STAGE1_RESULTS_DIR"] = str(STAGE1_RESULTS_DIR)
os.environ["STAGE2_RESULTS_DIR"] = str(STAGE2_RESULTS_DIR)
os.environ["STAGE1_KAGGLE_CFG"] = str(STAGE1_KAGGLE_CFG)
os.environ["STAGE2_KAGGLE_CFG"] = str(STAGE2_KAGGLE_CFG)
os.environ["STAGE1_DECODER_EXPORT"] = str(STAGE1_DECODER_EXPORT)
os.environ["STAGE1_STATS_PATH"] = str(STAGE1_STATS_PATH)
os.environ["FALLBACK_DECODER_PATH"] = str(FALLBACK_DECODER_PATH)
os.environ["DINO_DISC_CKPT"] = str(DINO_DISC_CKPT)
os.environ["STAGE1_BATCH_SIZE"] = str(STAGE1_BATCH_SIZE)
os.environ["STAGE2_BATCH_SIZE"] = str(STAGE2_BATCH_SIZE)
os.environ["STAGE2_NUM_WORKERS"] = str(STAGE2_NUM_WORKERS)
os.environ["STAGE2_PREFETCH_FACTOR"] = str(STAGE2_PREFETCH_FACTOR)
os.environ["STAGE2_CKPT_EVERY"] = str(STAGE2_CKPT_EVERY)
os.environ["STAGE2_SAMPLE_EVERY"] = str(STAGE2_SAMPLE_EVERY)

os.environ["JAX_PLATFORMS"] = "tpu,cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["UV_PROJECT_ENVIRONMENT"] = "/tmp/.venv"
os.environ["UV_CACHE_DIR"] = "/tmp/uv-cache"

print("Repo URL:", RAE_REPO_URL)
print("Branch:", RAE_BRANCH)
print("TFDS input root:", TFDS_SOURCE_DIR)
print("Planned Stage 1 cfg:", STAGE1_KAGGLE_CFG)
print("Planned Stage 2 cfg:", STAGE2_KAGGLE_CFG)
print("Stage 1 batch (main-like):", STAGE1_BATCH_SIZE)
print("Stage 2 Kaggle TPU settings:", json.dumps({
    "batch": STAGE2_BATCH_SIZE,
    "num_workers": STAGE2_NUM_WORKERS,
    "prefetch_factor": STAGE2_PREFETCH_FACTOR,
    "ckpt_every": STAGE2_CKPT_EVERY,
    "sample_every": STAGE2_SAMPLE_EVERY,
}, indent=2))
"""
    )

    clone_repo = code_cell(
        """%cd /kaggle/working
!rm -rf RAE
!git clone "${RAE_REPO_URL}" RAE
%cd /kaggle/working/RAE
!git checkout "${RAE_BRANCH}"
!curl -LsSf https://astral.sh/uv/install.sh | sh
!ln -sf /root/.local/bin/uv /usr/local/bin/uv
"""
    )

    sync_env = code_cell(
        """%cd /kaggle/working/RAE
!uv sync -q
!uv run python scripts/clear_elf_execstack.py --package jaxlib --quiet-unchanged
!env JAX_PLATFORMS=tpu,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -c "import os,sys,jax,jaxlib; print('python:', sys.executable); print('jax:', jax.__version__); print('jaxlib:', jaxlib.__version__); print('default backend:', jax.default_backend()); print('local device count:', jax.local_device_count()); print('devices:', jax.devices()); print('host cpu cores:', os.cpu_count())"
"""
    )

    prepare_tfds = code_cell(
        """%cd /kaggle/working
!rm -rf "${TFDS_WORK_ROOT}"
!mkdir -p "${TFDS_WORK_ROOT}"
!cp -r "${TFDS_SOURCE_DIR}" "${TFDS_WORK_ROOT}/"
!cp -r "${TFDS_BUILDERS_DIR}" "${TFDS_WORK_ROOT}/"
!echo "TFDS work root: ${TFDS_WORK_ROOT}"
!find "${TFDS_WORK_ROOT}" -maxdepth 2 -type d | sort | head -n 50
"""
    )

    download_assets = code_cell(
        """%cd /kaggle/working/RAE
!mkdir -p models
!uv run hf download nyu-visionx/RAE-collections \
  decoders/dinov2/wReg_base/decB_ganv3/dinov2_decoder.pt \
  discs/dino_vit_small_patch8_224.pth \
  --local-dir models
!ls -lah "${FALLBACK_DECODER_PATH}"
!ls -lah "${DINO_DISC_CKPT}"
"""
    )

    inspect_splits = code_cell(
        """import json
from pathlib import Path

candidate_infos = sorted(TFDS_DATA_DIR.glob("celebahq256/**/dataset_info.json"))
dataset_info_path = candidate_infos[-1] if candidate_infos else None
print("dataset_info_path =", dataset_info_path)

def _extract_split_counts(payload):
    counts = {}
    raw_splits = payload.get("splits", {})
    if isinstance(raw_splits, dict):
        for name, value in raw_splits.items():
            if isinstance(value, dict):
                counts[name] = value.get("numExamples") or value.get("shardLengths")
    elif isinstance(raw_splits, list):
        for value in raw_splits:
            if isinstance(value, dict):
                name = value.get("name")
                if name:
                    counts[name] = value.get("numExamples") or value.get("shardLengths")
    normalized = {}
    for name, value in counts.items():
        if isinstance(value, list):
            normalized[name] = sum(int(x) for x in value)
        elif value is not None:
            normalized[name] = int(value)
    return normalized

split_counts = {}
if dataset_info_path is not None:
    payload = json.loads(dataset_info_path.read_text())
    split_counts = _extract_split_counts(payload)

print("split_counts =", split_counts)
stage2_num_train_samples = int(split_counts.get("train", 28500))
os.environ["STAGE2_NUM_TRAIN_SAMPLES"] = str(stage2_num_train_samples)
print("STAGE2_NUM_TRAIN_SAMPLES =", stage2_num_train_samples)
"""
    )

    write_configs = code_cell(
        """from pathlib import Path
from textwrap import dedent

REPO_ROOT.mkdir(parents=True, exist_ok=True)
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
STAGE1_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STAGE2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

stage1_yaml = dedent(f\"\"\"\
data:
  format: tfds
  dataset_name: celebahq256
  train_split: train
  eval_split: validation
  shuffle_buffer: 20000

stage_1:
  target: stage1.RAE
  params:
    encoder_cls: 'Dinov2withNorm'
    encoder_config_path: 'facebook/dinov2-with-registers-base'
    encoder_input_size: 224
    encoder_params:
      dinov2_path: 'facebook/dinov2-with-registers-base'
      normalize: true
    decoder_config_path: 'configs/decoder/ViTB'
    noise_tau: 0.8
    reshape_to_2d: true

training:
  epochs: 16
  ema_decay: 0.9978
  global_batch_size: {STAGE1_BATCH_SIZE}
  num_workers: 8
  clip_grad: 0.0
  log_interval: 100
  checkpoint_interval: 1
  sample_every: 2500
  optimizer:
    lr: 2.0e-4
    betas: [0.9, 0.95]
    weight_decay: 0.0
  scheduler:
    type: cosine
    warmup_epochs: 1
    decay_end_epoch: 16
    base_lr: 2.0e-4
    final_lr: 2.0e-5
    warmup_from_zero: true

gan:
  disc:
    arch:
      dino_ckpt_path: '{DINO_DISC_CKPT}'
      ks: 9
      norm_type: 'bn'
      using_spec_norm: true
      recipe: 'S_8'
    optimizer:
      lr: 2.0e-4
      betas: [0.9, 0.95]
      weight_decay: 0.0
    scheduler:
      type: cosine
      warmup_epochs: 1
      decay_end_epoch: 16
      base_lr: 2.0e-4
      final_lr: 2.0e-5
      warmup_from_zero: true
    augment:
      prob: 1.0
      cutout: 0.0
  loss:
    disc_loss: hinge
    gen_loss: vanilla
    disc_weight: 0.75
    perceptual_weight: 1.0
    disc_start: 8
    disc_upd_start: 6
    lpips_start: 0
    max_d_weight: 10000.0
    disc_updates: 1
\"\"\")

active_decoder = STAGE1_DECODER_EXPORT if STAGE1_DECODER_EXPORT.exists() else FALLBACK_DECODER_PATH
active_stats_yaml = f\"'{STAGE1_STATS_PATH}'\" if STAGE1_STATS_PATH.exists() else "null"

stage2_yaml = dedent(f\"\"\"\
data:
  format: tfds
  dataset_name: celebahq256
  train_split: train
  eval_split: validation
  shuffle_buffer: 20000

stage_1:
  target: stage1.RAE
  params:
    encoder_cls: 'Dinov2withNorm'
    encoder_config_path: 'facebook/dinov2-with-registers-base'
    encoder_input_size: 224
    encoder_params:
      dinov2_path: 'facebook/dinov2-with-registers-base'
      normalize: true
    decoder_config_path: 'configs/decoder/ViTB'
    pretrained_decoder_path: '{active_decoder}'
    noise_tau: 0.0
    reshape_to_2d: true
    normalization_stat_path: {active_stats_yaml}

stage_2:
  target: stage2.models.lightningDiT.LightningDiT
  params:
    input_size: 16
    patch_size: 1
    in_channels: 768
    hidden_size: 768
    depth: 12
    num_heads: 12
    mlp_ratio: 4.0
    class_dropout_prob: 0.0
    num_classes: 1
    use_qknorm: false
    use_swiglu: true
    use_rope: true
    use_rmsnorm: true
    wo_shift: false

transport:
  params:
    path_type: 'Linear'
    prediction: 'velocity'
    loss_weight: null
    time_dist_type: 'uniform'

sampler:
  mode: ODE
  params:
    sampling_method: 'euler'
    num_steps: 50
    atol: 1.0e-6
    rtol: 1.0e-3
    reverse: false

guidance:
  method: 'cfg'
  scale: 1.0
  t_min: 0.0
  t_max: 1.0

misc:
  latent_size: [768, 16, 16]
  num_classes: 1
  time_dist_shift_dim: 196608
  time_dist_shift_base: 4096

training:
  global_seed: 0
  epochs: 1400
  global_batch_size: {STAGE2_BATCH_SIZE}
  ema_decay: 0.9995
  num_workers: {STAGE2_NUM_WORKERS}
  prefetch_factor: {STAGE2_PREFETCH_FACTOR}
  log_every: 100
  ckpt_every: {STAGE2_CKPT_EVERY}
  sample_every: {STAGE2_SAMPLE_EVERY}
  base_lr: 0.0002
  final_lr: 0.00002
  beta: [0.9, 0.95]
  wd: 0.0
  schedule_type: 'linear'
  decay_start_epoch: 40
  decay_end_epoch: 800
  clip_grad: 1.0
  random_flip: true
\"\"\")

STAGE1_KAGGLE_CFG.write_text(stage1_yaml, encoding="utf-8")
STAGE2_KAGGLE_CFG.write_text(stage2_yaml, encoding="utf-8")

print("Stage 1 Kaggle cfg:", STAGE1_KAGGLE_CFG)
print(STAGE1_KAGGLE_CFG.read_text())
print("Stage 2 Kaggle cfg:", STAGE2_KAGGLE_CFG)
print(STAGE2_KAGGLE_CFG.read_text())
"""
    )

    stage1_dry_run = code_cell(
        """%cd /kaggle/working/RAE
!uv run python src_jax/train_stage1.py \
  --config "${STAGE1_KAGGLE_CFG}" \
  --data-path "${TFDS_DATA_DIR}" \
  --data-format tfds \
  --dataset-name celebahq256 \
  --train-split train \
  --eval-split validation \
  --results-dir "${STAGE1_RESULTS_DIR}" \
  --precision bf16 \
  --print-config \
  --dry-run
"""
    )

    stage1_train_cmd = markdown_cell(
        """## Stage 1 all-JAX launch command

Cell bên dưới là đúng command contract cho Stage 1 main-like JAX trên Kaggle TPU.  
Hiện tại branch này mới có `config/state scaffold` cho `src_jax/train_stage1.py`, nên bạn giữ command này để chạy ngay khi phần trainer `RAE + LPIPS + GAN + DINO-disc` được port xong.

```bash
cd /kaggle/working/RAE

uv run python src_jax/train_stage1.py \
  --config /kaggle/working/RAE/configs/stage1/training/CelebAHQ256_DINOv2-B_decB_tfds_kaggle.yaml \
  --data-path /kaggle/working/shortcut_celebahq256/tensorflow_datasets \
  --data-format tfds \
  --dataset-name celebahq256 \
  --train-split train \
  --eval-split validation \
  --results-dir /kaggle/working/results_all_jax/stage1_jax \
  --precision bf16
```
"""
    )

    handoff = code_cell(
        """from pathlib import Path

active_decoder = STAGE1_DECODER_EXPORT if STAGE1_DECODER_EXPORT.exists() else FALLBACK_DECODER_PATH
active_stats_value = str(STAGE1_STATS_PATH) if STAGE1_STATS_PATH.exists() else "null"

os.environ["ACTIVE_DECODER_PATH"] = str(active_decoder)
os.environ["ACTIVE_STATS_VALUE"] = active_stats_value

print("ACTIVE_DECODER_PATH =", active_decoder)
print("ACTIVE_STATS_VALUE =", active_stats_value)
print("If Stage 1 export is not ready yet, the notebook falls back to the shared decoder for Stage 2 smoke tests.")
"""
    )

    stats_guidance = markdown_cell(
        """## Decoder handoff và latent stats

Đường đi mục tiêu sau khi Stage 1 all-JAX hoàn chỉnh:

1. export decoder mới từ checkpoint Stage 1 vào `/kaggle/working/artifacts/celebahq256_stage1_decoder.pt`
2. build latent stats mới vào `/kaggle/working/artifacts/celebahq256_stage1_stat.pt`
3. train Stage 2 với đúng decoder và stats đó

Hiện tại notebook tự fall back sang decoder shared từ `RAE-collections` nếu artifact Stage 1 chưa tồn tại.  
`normalization_stat_path` cũng tự để `null` nếu file latent stat chưa có, nên bạn vẫn smoke-test được Stage 2 trước.
"""
    )

    stage2_train = code_cell(
        """%cd /kaggle/working/RAE
!uv run python src_jax/train.py \
  --config "${STAGE2_KAGGLE_CFG}" \
  --data-path "${TFDS_DATA_DIR}" \
  --data-format tfds \
  --dataset-name celebahq256 \
  --train-split train \
  --eval-split validation \
  --results-dir "${STAGE2_RESULTS_DIR}" \
  --precision bf16 \
  --num-train-samples "${STAGE2_NUM_TRAIN_SAMPLES}" \
  --wandb \
  --set "stage_1.params.pretrained_decoder_path=${ACTIVE_DECODER_PATH}" \
  --set "stage_1.params.normalization_stat_path=${ACTIVE_STATS_VALUE}" \
  --set "training.global_batch_size=${STAGE2_BATCH_SIZE}" \
  --set "training.num_workers=${STAGE2_NUM_WORKERS}" \
  --set "training.prefetch_factor=${STAGE2_PREFETCH_FACTOR}" \
  --set "training.ckpt_every=${STAGE2_CKPT_EVERY}" \
  --set "training.sample_every=${STAGE2_SAMPLE_EVERY}"
"""
    )

    recommendations = markdown_cell(
        """## Kaggle TPU v5e-8 recommendations

- `STAGE1_BATCH_SIZE=512` là giá trị bám config `main`; nếu Kaggle TPU không kham nổi khi trainer Stage 1 JAX hoàn chỉnh, hạ dần xuống `256`, `128`, `64`, rồi `32`.
- `STAGE2_BATCH_SIZE=64`, `STAGE2_NUM_WORKERS=1`, `STAGE2_PREFETCH_FACTOR=8`, `STAGE2_CKPT_EVERY=210000`, `STAGE2_SAMPLE_EVERY=5000` là các giá trị đang khớp notebook TPU CelebA-HQ hiện có trong branch này.
- Nếu gặp lỗi `cannot enable executable stack` từ `jaxlib`, chạy lại cell `clear_elf_execstack`.
- Nếu muốn bám `main` hơn nữa ở Stage 2 sau khi Stage 1 xong, chỉ việc thay `ACTIVE_DECODER_PATH` và `ACTIVE_STATS_VALUE` bằng artifact mới; phần transport và optimizer hiện tại đã giữ đúng recipe `Linear + velocity + Euler`.
"""
    )

    return {
        "cells": [
            title,
            env_setup,
            clone_repo,
            sync_env,
            prepare_tfds,
            download_assets,
            inspect_splits,
            write_configs,
            stage1_dry_run,
            stage1_train_cmd,
            handoff,
            stats_guidance,
            stage2_train,
            recommendations,
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "raes-jax-celebahq-kaggle-tpuv5e8-alljax-lightningdit.ipynb"
    output_path.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
