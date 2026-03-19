from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def upload_path(
    local_path: str,
    repo_id: str,
    *,
    repo_type: str = "model",
    revision: str = "main",
    private: bool = False,
    token_env: str = "HF_TOKEN",
    path_in_repo: str | None = None,
    commit_message: str | None = None,
) -> None:
    token = os.environ.get(token_env)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

    source = Path(local_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Hugging Face upload source not found: {source}")

    if source.is_dir():
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(source),
            path_in_repo=path_in_repo,
            revision=revision,
            commit_message=commit_message or f"upload {source.name}",
        )
        return

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=str(source),
        path_in_repo=path_in_repo or source.name,
        revision=revision,
        commit_message=commit_message or f"upload {source.name}",
    )
