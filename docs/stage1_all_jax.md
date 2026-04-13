# Stage 1 All-JAX Roadmap

This branch is moving Stage 1 toward the same training behavior as `main`, but with a JAX-first stack that fits TPU training better than a `torch_xla` fork.

## Target stack

- Model/runtime: JAX + Flax/NNX
- Optimization: Optax
- Checkpointing: Orbax
- Data: TFDS, with the same CelebA-HQ256 builder flow already used in `shortcut-models`
- Precision/sharding: regular JAX TPU flow, keeping the code Pallas-free unless profiling proves a hotspot needs a custom kernel

## What must match `main`

- Frozen Stage 1 encoder + trainable decoder
- Reconstruction objective
- LPIPS perceptual loss
- GAN loss with adaptive discriminator weighting
- DINO discriminator
- DiffAug
- EMA checkpoint for the generator
- Evaluation/export path that still hands a decoder checkpoint into Stage 2

## Port order

1. Config/runtime scaffold
   - Parse existing Stage 1 configs into a JAX-native runtime object.
   - Keep the current YAML surface so training recipes stay familiar.

2. Train-state scaffold
   - Introduce separate generator/discriminator train states with Optax and EMA.
   - Mirror the dual-optimizer structure from `main`.

3. TFDS-native Stage 1 datapipe
   - Use the existing TFDS helpers under `src_jax/`.
   - Avoid reintroducing torch `DataLoader` on the JAX path.

4. Reconstruction-only generator path
   - Port the trainable decoder path first.
   - Verify loss parity and checkpoint semantics before adding adversarial pieces.

5. LPIPS port
   - Reproduce the perceptual objective in JAX.
   - Validate scale and input normalization against `main`.

6. DINO discriminator + DiffAug
   - Port discriminator blocks and augmentation behavior.
   - Recreate adaptive GAN weighting and alternating optimizer updates.

7. Stage 1 handoff to Stage 2
   - Export decoder weights/statistics in a format the JAX Stage 2 path can consume directly.

## Current scaffold in this branch

- `src_jax/stage1_jax_config.py`
  Parses existing Stage 1 YAML into a JAX runtime config.
- `src_jax/stage1_jax_state.py`
  Provides Optax schedule/optimizer helpers and EMA train-state utilities.

These pieces are meant to stabilize the structure before the model/loss port lands.
