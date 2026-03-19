from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


BACKEND_REPO_URL = "https://github.com/willisma/diffuse_nnx"
BACKEND_COMMIT = "023afd23c7b62a8cdb00e840b36a4ab8fc970bba"
BACKEND_ENV_VAR = "RAE_JAX_BACKEND_DIR"
DEFAULT_BACKEND_DIR = Path.home() / ".cache" / "rae_jax" / "diffuse_nnx"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _git_head(path: Path) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return output.strip()


def _patch_backend_file(path: Path, old: str, new: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Backend compatibility patch target not found: {path}")
    content = path.read_text(encoding="utf-8")
    if new in content:
        return
    if old not in content:
        raise RuntimeError(f"Unexpected backend source layout while patching {path}")
    path.write_text(content.replace(old, new, 1), encoding="utf-8")


def _apply_backend_compat_patches(backend_dir: Path) -> None:
    dino_path = backend_dir / "networks" / "encoders" / "dino.py"
    _patch_backend_file(
        dino_path,
        "from transformers import FlaxDinov2Model, AutoImageProcessor\n",
        textwrap.dedent(
            """\
            from transformers import AutoImageProcessor
            from transformers.models.dinov2.modeling_flax_dinov2 import FlaxDinov2Model
            """
        ),
    )

    dino_w_register_path = backend_dir / "networks" / "encoders" / "dino_w_register.py"
    _patch_backend_file(
        dino_w_register_path,
        "from transformers import Dinov2WithRegistersModel\n",
        "from transformers.models.dinov2_with_registers import Dinov2WithRegistersModel\n",
    )

    encoder_utils_path = backend_dir / "networks" / "encoders" / "utils.py"
    _patch_backend_file(
        encoder_utils_path,
        textwrap.dedent(
            """\
            \"\"\"File containing utility functions for the encoder.\"\"\"

            # built-in libs
            import math

            # external libs
            from google.cloud import storage

            def download_blob(bucket_name, source_blob_name, destination_file_name):
                \"\"\"Downloads a blob from the bucket.\"\"\"
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(source_blob_name)
                blob.download_to_filename(destination_file_name)
            """
        ),
        textwrap.dedent(
            """\
            \"\"\"File containing utility functions for the encoder.\"\"\"

            # built-in libs
            import math


            def _load_storage():
                try:
                    from google.cloud import storage
                except ImportError as exc:
                    raise ImportError(
                        \"google-cloud-storage is only required when diffuse_nnx needs to download \"
                        \"encoder assets from GCS. Install it with `uv pip install google-cloud-storage` \"
                        \"or provide the expected local checkpoint files.\"
                    ) from exc
                return storage


            def download_blob(bucket_name, source_blob_name, destination_file_name):
                \"\"\"Downloads a blob from the bucket.\"\"\"
                storage = _load_storage()
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(source_blob_name)
                blob.download_to_filename(destination_file_name)
            """
        ),
    )


def resolve_backend_dir(explicit_dir: str | None = None) -> Path:
    raw = explicit_dir or os.environ.get(BACKEND_ENV_VAR)
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_BACKEND_DIR


def ensure_backend(explicit_dir: str | None = None) -> Path:
    backend_dir = resolve_backend_dir(explicit_dir)
    backend_dir.parent.mkdir(parents=True, exist_ok=True)

    if not (backend_dir / ".git").exists():
        _run(["git", "clone", "--depth", "1", BACKEND_REPO_URL, str(backend_dir)])

    current_head = _git_head(backend_dir)
    if current_head != BACKEND_COMMIT:
        _run(["git", "-C", str(backend_dir), "fetch", "--depth", "1", "origin", BACKEND_COMMIT])
        _run(["git", "-C", str(backend_dir), "checkout", BACKEND_COMMIT])

    _apply_backend_compat_patches(backend_dir)

    return backend_dir


def activate_backend(explicit_dir: str | None = None) -> Path:
    backend_dir = ensure_backend(explicit_dir)
    backend_path = str(backend_dir)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    return backend_dir
