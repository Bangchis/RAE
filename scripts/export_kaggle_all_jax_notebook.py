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
        """# RAE CelebA-HQ256 Kaggle TPU v5e-8 Minimal Runbook

Notebook này chỉ giữ 3 lệnh chính như bạn yêu cầu:

1. train Stage 1
2. extract decoder từ Stage 1
3. train Stage 2

Thiết lập hiện tại:

- dataset root cố định: `/kaggle/input/shortcut-celebahq256`
- split dùng thẳng `train` và `validation`
- Stage 1 bám config `main` hơn: `global_batch_size=512`
- Stage 2 giữ override Kaggle TPU đang dùng trong branch này: `batch=64`, `num_workers=1`, `prefetch_factor=8`, `ckpt_every=210000`, `sample_every=5000`
"""
    )

    setup = code_cell(
        """import json
import os
from pathlib import Path
from textwrap import dedent

try:
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    wandb_key = secrets.get_secret("WANDB_KEY")
    os.environ["WANDB_API_KEY"] = wandb_key
    os.environ["WANDB_KEY"] = wandb_key
except Exception as exc:
    print(f"Skipping Kaggle secret bootstrap: {exc}")

REPO_URL = "https://github.com/Bangchis/RAE.git"
REPO_BRANCH = "jax-sit-dh-celebahq256"

DATASET_ROOT = Path("/kaggle/input/shortcut-celebahq256")
TFDS_DATA_DIR = DATASET_ROOT / "tensorflow_datasets"
TFDS_BUILDERS_DIR = DATASET_ROOT / "tfds_builders"
WORK_ROOT = Path("/kaggle/working")
REPO_ROOT = WORK_ROOT / "RAE"
RESULTS_ROOT = WORK_ROOT / "results_all_jax"
ARTIFACTS_ROOT = WORK_ROOT / "artifacts"

STAGE1_RESULTS_DIR = RESULTS_ROOT / "stage1_jax"
STAGE2_RESULTS_DIR = RESULTS_ROOT / "stage2_jax"
STAGE1_CFG = REPO_ROOT / "configs" / "stage1" / "training" / "CelebAHQ256_DINOv2-B_decB_tfds_kaggle.yaml"
STAGE2_CFG = REPO_ROOT / "configs" / "stage2" / "training" / "CelebAHQ256" / "LightningDiT-B_DINOv2-B_tfds_kaggle.yaml"

STAGE1_DECODER_EXPORT = ARTIFACTS_ROOT / "celebahq256_stage1_decoder.pt"
STAGE1_CKPT_PATH = STAGE1_RESULTS_DIR / "checkpoints" / "ep-last.pt"
FALLBACK_DECODER_PATH = REPO_ROOT / "models" / "decoders" / "dinov2" / "wReg_base" / "decB_ganv3" / "dinov2_decoder.pt"
DINO_DISC_CKPT = REPO_ROOT / "models" / "discs" / "dino_vit_small_patch8_224.pth"

STAGE1_BATCH_SIZE = 512
STAGE2_BATCH_SIZE = 64
STAGE2_NUM_WORKERS = 1
STAGE2_PREFETCH_FACTOR = 8
STAGE2_CKPT_EVERY = 210000
STAGE2_SAMPLE_EVERY = 5000

dataset_info_candidates = sorted((TFDS_DATA_DIR / "celebahq256").glob("*/dataset_info.json"))
dataset_info_path = dataset_info_candidates[0] if dataset_info_candidates else None
stage2_num_train_samples = 30000
if dataset_info_path is not None:
    payload = json.loads(dataset_info_path.read_text())
    raw_splits = payload.get("splits", {})
    if isinstance(raw_splits, dict):
        train_info = raw_splits.get("train", {})
        if isinstance(train_info, dict):
            stage2_num_train_samples = int(train_info.get("numExamples", stage2_num_train_samples))

os.environ["JAX_PLATFORMS"] = "tpu,cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["UV_PROJECT_ENVIRONMENT"] = "/tmp/.venv"
os.environ["UV_CACHE_DIR"] = "/tmp/uv-cache"

os.environ["REPO_URL"] = REPO_URL
os.environ["REPO_BRANCH"] = REPO_BRANCH
os.environ["TFDS_DATA_DIR"] = str(TFDS_DATA_DIR)
os.environ["TFDS_BUILDERS_DIR"] = str(TFDS_BUILDERS_DIR)
os.environ["REPO_ROOT"] = str(REPO_ROOT)
os.environ["RESULTS_ROOT"] = str(RESULTS_ROOT)
os.environ["ARTIFACTS_ROOT"] = str(ARTIFACTS_ROOT)
os.environ["STAGE1_RESULTS_DIR"] = str(STAGE1_RESULTS_DIR)
os.environ["STAGE2_RESULTS_DIR"] = str(STAGE2_RESULTS_DIR)
os.environ["STAGE1_CFG"] = str(STAGE1_CFG)
os.environ["STAGE2_CFG"] = str(STAGE2_CFG)
os.environ["STAGE1_DECODER_EXPORT"] = str(STAGE1_DECODER_EXPORT)
os.environ["STAGE1_CKPT_PATH"] = str(STAGE1_CKPT_PATH)
os.environ["FALLBACK_DECODER_PATH"] = str(FALLBACK_DECODER_PATH)
os.environ["DINO_DISC_CKPT"] = str(DINO_DISC_CKPT)
os.environ["STAGE2_NUM_TRAIN_SAMPLES"] = str(stage2_num_train_samples)

print("repo:", REPO_URL, REPO_BRANCH)
print("dataset:", DATASET_ROOT)
print("dataset_info:", dataset_info_path)
print("stage2_num_train_samples:", stage2_num_train_samples)

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
    pretrained_decoder_path: '{STAGE1_DECODER_EXPORT}'
    noise_tau: 0.0
    reshape_to_2d: true
    normalization_stat_path: null

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

%cd /kaggle/working
!rm -rf RAE
!git clone "{REPO_URL}" RAE
%cd /kaggle/working/RAE
!git checkout "{REPO_BRANCH}"
!curl -LsSf https://astral.sh/uv/install.sh | sh
!ln -sf /root/.local/bin/uv /usr/local/bin/uv
!uv sync -q
!uv run python scripts/clear_elf_execstack.py --package jaxlib --quiet-unchanged
!mkdir -p models "{RESULTS_ROOT}" "{ARTIFACTS_ROOT}"
!uv run hf download nyu-visionx/RAE-collections decoders/dinov2/wReg_base/decB_ganv3/dinov2_decoder.pt discs/dino_vit_small_patch8_224.pth --local-dir models

STAGE1_CFG.write_text(stage1_yaml, encoding="utf-8")
STAGE2_CFG.write_text(stage2_yaml, encoding="utf-8")
print(STAGE1_CFG)
print(STAGE2_CFG)
"""
    )

    stage1_train = code_cell(
        """%cd /kaggle/working/RAE
!uv run python src_jax/train_stage1.py \
  --config "${STAGE1_CFG}" \
  --data-path "${TFDS_DATA_DIR}" \
  --data-format tfds \
  --dataset-name celebahq256 \
  --results-dir "${STAGE1_RESULTS_DIR}" \
  --precision bf16
"""
    )

    stage1_extract = code_cell(
        """%cd /kaggle/working/RAE
!uv run python src/extract_decoder.py \
  --config "${STAGE1_CFG}" \
  --ckpt "${STAGE1_CKPT_PATH}" \
  --use-ema \
  --out "${STAGE1_DECODER_EXPORT}"
"""
    )

    stage2_train = code_cell(
        """%cd /kaggle/working/RAE
!uv run python src_jax/train.py \
  --config "${STAGE2_CFG}" \
  --data-path "${TFDS_DATA_DIR}" \
  --data-format tfds \
  --dataset-name celebahq256 \
  --results-dir "${STAGE2_RESULTS_DIR}" \
  --precision bf16 \
  --num-train-samples "${STAGE2_NUM_TRAIN_SAMPLES}" \
  --wandb
"""
    )

    notes = markdown_cell(
        """Ghi chú ngắn:

- `STAGE1_BATCH_SIZE=512` là giá trị bám `main`; nếu Kaggle TPU không kham nổi khi trainer Stage 1 JAX hoàn chỉnh, hạ dần xuống `256`, `128`, `64`, `32`.
- `STAGE2_BATCH_SIZE=64`, `num_workers=1`, `prefetch_factor=8`, `ckpt_every=210000`, `sample_every=5000` là các giá trị mình giữ theo notebook TPU CelebA-HQ hiện có trong branch này.
- Lệnh extract hiện dùng `src/extract_decoder.py` để tạo file decoder mà Stage 2 JAX có thể dùng trực tiếp.
"""
    )

    return {
        "cells": [
            title,
            setup,
            stage1_train,
            stage1_extract,
            stage2_train,
            notes,
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
