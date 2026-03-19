from __future__ import annotations

import os
import subprocess
import sys
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

    return backend_dir


def activate_backend(explicit_dir: str | None = None) -> Path:
    backend_dir = ensure_backend(explicit_dir)
    backend_path = str(backend_dir)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    return backend_dir
