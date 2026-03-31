from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


DATASET_DOCS_URL = "https://www.tensorflow.org/datasets/catalog/celeb_a_hq"
MANUAL_PREP_URL = "https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training"


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


def _load_tfds_builder(dataset: str, data_dir: Path, manual_dir: Path | None):
    try:
        import tensorflow_datasets as tfds
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorflow-datasets is required for CelebA-HQ export. "
            "Install repo dependencies with `uv sync` first."
        ) from exc

    builder = tfds.builder(dataset, data_dir=str(data_dir))
    download_kwargs: dict[str, Any] = {}
    if manual_dir is not None:
        download_kwargs["download_config"] = tfds.download.DownloadConfig(manual_dir=str(manual_dir))

    try:
        builder.download_and_prepare(**download_kwargs)
    except Exception as exc:  # pragma: no cover - depends on local TFDS state
        manual_hint = f"manual_dir={manual_dir}" if manual_dir is not None else "manual_dir is unset"
        raise RuntimeError(
            "Failed to prepare TFDS CelebA-HQ dataset. "
            f"Provide the official Progressive GAN tar files in {manual_hint}. "
            f"See {DATASET_DOCS_URL} and {MANUAL_PREP_URL}."
        ) from exc
    return builder, tfds


def _export_split(
    *,
    builder: Any,
    tfds: Any,
    split_name: str,
    split_spec: str,
    output_root: Path,
    class_name: str,
    overwrite: bool,
) -> int:
    split_dir = output_root / split_name / class_name
    split_dir.mkdir(parents=True, exist_ok=True)
    dataset = builder.as_dataset(split=split_spec, shuffle_files=False)

    written = 0
    for index, example in enumerate(tfds.as_numpy(dataset)):
        filename = normalize_example_filename(example.get("image/filename"), index=index)
        output_path = split_dir / filename
        if output_path.exists() and not overwrite:
            written += 1
            continue
        image = Image.fromarray(example["image"])
        image.save(output_path)
        written += 1
    return written


def export_celebahq_to_imagefolder(
    *,
    dataset: str,
    output_root: Path,
    data_dir: Path,
    manual_dir: Path | None,
    train_percent: int,
    val_percent: int,
    class_name: str,
    overwrite: bool,
) -> dict[str, Any]:
    split_specs = build_split_specs(train_percent=train_percent, val_percent=val_percent)
    builder, tfds = _load_tfds_builder(dataset, data_dir=data_dir, manual_dir=manual_dir)

    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split_name, split_spec in split_specs.items():
        counts[split_name] = _export_split(
            builder=builder,
            tfds=tfds,
            split_name=split_name,
            split_spec=split_spec,
            output_root=output_root,
            class_name=class_name,
            overwrite=overwrite,
        )

    summary = {
        "dataset": dataset,
        "data_dir": str(data_dir),
        "manual_dir": str(manual_dir) if manual_dir is not None else None,
        "output_root": str(output_root),
        "split_specs": split_specs,
        "class_name": class_name,
        "counts": counts,
    }
    (output_root / "tfds_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export TFDS celeb_a_hq/<resolution> into the ImageFolder layout expected by the JAX pipeline."
    )
    parser.add_argument("--dataset", default="celeb_a_hq/256", help="TFDS dataset name, e.g. celeb_a_hq/256.")
    parser.add_argument("--output", type=Path, required=True, help="Destination ImageFolder root.")
    parser.add_argument("--data-dir", type=Path, required=True, help="TFDS prepared cache directory.")
    parser.add_argument(
        "--manual-dir",
        type=Path,
        default=None,
        help="Directory containing the official manual celeb_a_hq tar files required by TFDS.",
    )
    parser.add_argument("--train-percent", type=int, default=90, help="Percentage assigned to the train split.")
    parser.add_argument("--val-percent", type=int, default=5, help="Percentage assigned to the val split.")
    parser.add_argument("--class-name", default="face", help="Single class folder name inside each split.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output tree before exporting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = export_celebahq_to_imagefolder(
        dataset=args.dataset,
        output_root=args.output.expanduser().resolve(),
        data_dir=args.data_dir.expanduser().resolve(),
        manual_dir=args.manual_dir.expanduser().resolve() if args.manual_dir is not None else None,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        class_name=args.class_name,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
