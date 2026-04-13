#!/usr/bin/env bash
# RAE Training Pipeline — CelebA-HQ 256
# Stage 1 (PyTorch) → latent stats → extract decoder → Stage 2 (JAX LightningDiT-XL)
#
# Chỉnh các biến ở mục CONFIGURATION trước khi chạy.
# Chạy từng bước riêng lẻ bằng cách comment/uncomment.

set -euo pipefail

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
TFDS_DATA_DIR="data/tfds"                          # thư mục chứa TFDS dataset
NUM_GPUS=8                                         # số GPU cho Stage 1
STAGE1_CKPT="ckpts/stage1_celebahq256/<run_name>/checkpoints/best.pt"
DECODER_OUT="models/decoders/celebahq256_decoder.pt"
STATS_OUT="models/stats/celebahq256_stage1_stat.pt"
NUM_TRAIN_SAMPLES=27000                            # ~CelebA-HQ full = 30000
STAGE2_BATCH_SIZE=256                              # giảm nếu OOM

# Configs
CFG_STAGE1="configs/stage1/training/CelebAHQ256_DINOv2-B_decXL_tfds.yaml"
CFG_STAGE2="configs/stage2/training/CelebAHQ256/LightningDiT-XL_DINOv2-B_tfds.yaml"


# ────────────────────────────────────────────────
# BƯỚC 0  Chuẩn bị dataset CelebA-HQ256
# ────────────────────────────────────────────────
# Cách A: tải từ Hugging Face (eurecom-ds/celeba-hq-256)
step0a() {
    python src_jax/export_celebahq_hf.py \
        --output-dir data/celebahq256_imgfolder
}

# Cách B: build TFDS từ tar files gốc (cần trỏ --manual-dir)
step0b() {
    python src_jax/export_celebahq_tfds.py \
        --data-dir "${TFDS_DATA_DIR}" \
        --manual-dir /path/to/celeba-hq-tars
}


# ────────────────────────────────────────────────
# BƯỚC 1  Train Stage 1 (PyTorch multi-GPU)
# ────────────────────────────────────────────────
step1() {
    torchrun --nproc_per_node "${NUM_GPUS}" src/train_stage1.py \
        --config "${CFG_STAGE1}" \
        --data-path "${TFDS_DATA_DIR}" \
        --data-format tfds \
        --dataset-name celebahq256 \
        --train-split "train[:95%]" \
        --eval-split "train[95%:]" \
        --image-size 256 \
        --precision bf16 \
        --results-dir ckpts/stage1_celebahq256 \
        --wandb
}

# Single-GPU (debug)
step1_single() {
    python src/train_stage1.py \
        --config "${CFG_STAGE1}" \
        --data-path "${TFDS_DATA_DIR}" \
        --data-format tfds \
        --dataset-name celebahq256 \
        --image-size 256 \
        --precision bf16 \
        --results-dir ckpts/stage1_celebahq256
}


# ────────────────────────────────────────────────
# BƯỚC 2  Build Stage 1 latent normalization stats
# ────────────────────────────────────────────────
step2() {
    python src_jax/build_stage1_stats.py \
        --config "${CFG_STAGE1}" \
        --input "${TFDS_DATA_DIR}" \
        --output "${STATS_OUT}" \
        --batch-size 32 \
        --num-workers 4 \
        --precision bf16
}


# ────────────────────────────────────────────────
# BƯỚC 3  Extract decoder từ Stage 1 checkpoint
# ────────────────────────────────────────────────
step3() {
    python src/extract_decoder.py \
        --config "${CFG_STAGE1}" \
        --ckpt "${STAGE1_CKPT}" \
        --use-ema \
        --out "${DECODER_OUT}" \
        --dtype bf16
}


# ────────────────────────────────────────────────
# BƯỚC 4  Train Stage 2 JAX LightningDiT-XL
# ────────────────────────────────────────────────
step4() {
    python src_jax/train.py \
        --config "${CFG_STAGE2}" \
        --data-path "${TFDS_DATA_DIR}" \
        --data-format tfds \
        --dataset-name celebahq256 \
        --train-split "train[:95%]" \
        --eval-split "train[95%:]" \
        --results-dir results/stage2_celebahq256_lightningdit \
        --precision bf16 \
        --num-train-samples "${NUM_TRAIN_SAMPLES}" \
        --wandb \
        --set \
            "stage_1.params.pretrained_decoder_path=${DECODER_OUT}" \
            "stage_1.params.normalization_stat_path=${STATS_OUT}" \
            "training.global_batch_size=${STAGE2_BATCH_SIZE}"
}


# ────────────────────────────────────────────────
# Chạy pipeline đầy đủ (comment bước không cần)
# ────────────────────────────────────────────────
# step0b    # chuẩn bị TFDS dataset
# step1     # train Stage 1
# step2     # build latent stats
# step3     # extract decoder
# step4     # train Stage 2
