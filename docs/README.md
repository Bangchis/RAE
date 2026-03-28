# Documentation

This folder is the detailed documentation entrypoint for the XLA branch of RAE
and the lightweight JAX/NNX adapter shipped in `src_jax/`.

## Documents

- [Architecture](./architecture.md): system overview, module boundaries, and code map.
- [Workflows](./workflows.md): practical runbooks for setup, Stage 2 training, sampling, FID, wandb, and Hugging Face upload.
- [Config Reference](./config-reference.md): YAML schema and which scripts consume each block.
- [PDF Manual](../pdf/main.pdf): detailed Vietnamese PDF for architecture, workflows, config, and operations.
- [Kaggle Notebook](../raes-jax-celeba-kaggle.ipynb): end-to-end CelebA notebook for the standard JAX branch flow, syncing repo dependencies into `/tmp/.venv` via `uv sync` and running package-backed steps through `uv run`.
- [Kaggle TPU Notebook (DiTDH-S)](../raes-jax-celeba-kaggle-tpuv5e8-ditdh-s.ipynb): CelebA notebook tuned for Kaggle `TPU v5e-8` for the `DiTDH-S` variant, also syncing repo dependencies into `/tmp/.venv` via `uv sync`, with an automatic `jaxlib` executable-stack fix, host-side CPU FID/stat work, and a default `210000`-step checkpoint cadence.
- [Kaggle TPU Notebook (DiTDH-B)](../raes-jax-celeba-kaggle-tpuv5e8-ditdh-b.ipynb): sibling `TPU v5e-8` notebook that keeps the same `/tmp/.venv` + `uv sync` JAX/Kaggle flow but swaps the Stage 2 CelebA config to `DiTDH-B`, also using the `210000`-step checkpoint cadence.
- [Kaggle TPU Notebook (DiTDH-B Resume)](../raes-jax-celeba-kaggle-tpuv5e8-ditdh-b-resume.ipynb): minimal `TPU v5e-8` resume-only notebook that starts by unzipping the previous notebook `_output_.zip` back into `/kaggle/working`, then rebuilds only the `uv` environment and auto-detects the latest `CelebA256_DiTDH-B_DINOv2-B_jax_tpuv5e8-*` run plus its latest `checkpoint_<step>`.

## Recommended Reading Order

1. Read [Architecture](./architecture.md) to understand the Stage 1 and Stage 2 split.
2. Read [Workflows](./workflows.md) before running experiments.
3. Use [Config Reference](./config-reference.md) while editing YAML files.

## Branch Scope

This branch is centered on two runtime paths:

- `src/`: the existing `torch_xla` TPU path for Stage 2 training, Stage 2 sampling, Stage 1 reconstruction, and host-side FID utilities
- `src_jax/`: a thin JAX/NNX adapter that keeps the current YAML schema, logs to wandb, can upload workdirs to Hugging Face, and supports Stage 2 train/sample plus Stage 1 reconstruction, folder reconstruction, and latent-stat building without forking the whole codebase

Limitations:

- `src/` still does not ship a dedicated Stage 1 training entrypoint on the XLA side.
- `src_jax/` intentionally does not port the full adversarial Stage 1 training loop yet.
