from __future__ import annotations

import argparse

try:
    from .hf_utils import upload_path
except ImportError:
    from hf_utils import upload_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload a file or directory from the JAX branch to Hugging Face.")
    parser.add_argument("--path", required=True, help="Local file or directory to upload.")
    parser.add_argument("--repo-id", required=True, help="Destination Hugging Face repo id.")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--path-in-repo", default=None)
    parser.add_argument("--commit-message", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    upload_path(
        args.path,
        args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        private=args.private,
        token_env=args.token_env,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
