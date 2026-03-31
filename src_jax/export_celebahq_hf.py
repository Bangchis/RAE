from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

HF_DATASET_DEFAULT = "eurecom-ds/celeba-hq-256"
HF_DATASET_URL = f"https://huggingface.co/datasets/{HF_DATASET_DEFAULT}"


def build_split_specs(train_percent: int = 90, val_percent: int = 5) -> dict[str, str]:
    if train_percent <= 0 or val_percent <= 0:
        raise ValueError("train_percent and val_percent must both be greater than 0.")
    test_percent = 100 - train_percent - val_percent
    if test_percent <= 0:
        raise ValueError("train_percent + val_percent must be less than 100.")
    return {
        "train": f"train[:{train_percent}%]",
        "val": f"train[{train_percent}%:{train_percent + val_percent}%]",
        "test": f"train[{train_percent + val_percent}%:]",
    }


def normalize_example_filename(raw_filename: Any, *, index: int) -> str:
    if hasattr(raw_filename, "item"):
        raw_filename = raw_filename.item()
    if isinstance(raw_filename, bytes):
        raw_filename = raw_filename.decode("utf-8")
    filename = Path(str(raw_filename or f"{index:06d}.png")).name
    if not Path(filename).suffix:
        filename = f"{filename}.png"
    return filename


def _load_hf_dataset(
    dataset: str,
    *,
    split: str,
    cache_dir: Path | None,
    revision: str | None,
):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `datasets` package is required for Hugging Face CelebA-HQ export. "
            "Install repo dependencies with `uv sync` first."
        ) from exc

    load_kwargs: dict[str, Any] = {"split": split}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    if revision is not None:
        load_kwargs["revision"] = revision

    try:
        return load_dataset(dataset, **load_kwargs)
    except Exception as exc:  # pragma: no cover - depends on remote dataset state
        raise RuntimeError(
            "Failed to load the Hugging Face CelebA-HQ dataset. "
            f"Check internet access and dataset availability for {dataset}. "
            f"The default public source is {HF_DATASET_URL}."
        ) from exc


def _coerce_image_to_pil(image_value: Any) -> Image.Image:
    if isinstance(image_value, Image.Image):
        pil_image = image_value
    elif isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            pil_image = Image.open(io.BytesIO(image_value["bytes"]))
        elif image_value.get("path"):
            pil_image = Image.open(image_value["path"])
        else:
            raise TypeError("Image dictionaries must provide `bytes` or `path`.")
    elif hasattr(image_value, "__array__"):
        pil_image = Image.fromarray(image_value.__array__())
    else:
        raise TypeError(f"Unsupported image payload type: {type(image_value)!r}")
    return pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image


def _resolve_example_filename(example: dict[str, Any], *, index: int, filename_key: str | None) -> str:
    if filename_key is not None:
        return normalize_example_filename(example.get(filename_key), index=index)

    for key in ("file_name", "filename", "image/filename", "path", "name"):
        if example.get(key):
            return normalize_example_filename(example[key], index=index)

    image_value = example.get("image")
    if isinstance(image_value, dict):
        for key in ("path", "filename"):
            if image_value.get(key):
                return normalize_example_filename(image_value[key], index=index)

    return normalize_example_filename(None, index=index)


def _export_examples(
    *,
    examples: Iterable[dict[str, Any]],
    split_name: str,
    output_root: Path,
    class_name: str,
    overwrite: bool,
    image_key: str,
    filename_key: str | None,
) -> int:
    split_dir = output_root / split_name / class_name
    split_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for index, example in enumerate(examples):
        filename = _resolve_example_filename(example, index=index, filename_key=filename_key)
        output_path = split_dir / filename
        if output_path.exists() and not overwrite:
            written += 1
            continue
        image = _coerce_image_to_pil(example[image_key])
        image.save(output_path)
        written += 1
    return written


def export_celebahq_hf_to_imagefolder(
    *,
    dataset: str,
    output_root: Path,
    cache_dir: Path | None,
    revision: str | None,
    train_percent: int,
    val_percent: int,
    class_name: str,
    image_key: str,
    filename_key: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    split_specs = build_split_specs(train_percent=train_percent, val_percent=val_percent)

    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split_name, split_spec in split_specs.items():
        dataset_split = _load_hf_dataset(
            dataset,
            split=split_spec,
            cache_dir=cache_dir,
            revision=revision,
        )
        counts[split_name] = _export_examples(
            examples=dataset_split,
            split_name=split_name,
            output_root=output_root,
            class_name=class_name,
            overwrite=overwrite,
            image_key=image_key,
            filename_key=filename_key,
        )

    summary = {
        "dataset": dataset,
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "output_root": str(output_root),
        "split_specs": split_specs,
        "class_name": class_name,
        "image_key": image_key,
        "filename_key": filename_key,
        "counts": counts,
    }
    (output_root / "hf_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face CelebA-HQ dataset into the ImageFolder layout expected by the JAX pipeline."
    )
    parser.add_argument(
        "--dataset",
        default=HF_DATASET_DEFAULT,
        help="Hugging Face dataset id, e.g. eurecom-ds/celeba-hq-256.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination ImageFolder root.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face datasets cache directory.")
    parser.add_argument("--revision", default=None, help="Optional dataset revision or commit hash for reproducibility.")
    parser.add_argument("--train-percent", type=int, default=90, help="Percentage assigned to the train split.")
    parser.add_argument("--val-percent", type=int, default=5, help="Percentage assigned to the val split.")
    parser.add_argument("--class-name", default="face", help="Single class folder name inside each split.")
    parser.add_argument("--image-key", default="image", help="Column name holding the image payload.")
    parser.add_argument(
        "--filename-key",
        default=None,
        help="Optional column name to use as the exported file name. Defaults to auto-detection with numeric fallback.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output tree before exporting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = export_celebahq_hf_to_imagefolder(
        dataset=args.dataset,
        output_root=args.output.expanduser().resolve(),
        cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir is not None else None,
        revision=args.revision,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        class_name=args.class_name,
        image_key=args.image_key,
        filename_key=args.filename_key,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
