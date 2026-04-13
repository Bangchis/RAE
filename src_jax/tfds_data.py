from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np


def configure_tfds_runtime() -> None:
    # TFDS can pick up incompatible protobuf wheels on notebook stacks.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _maybe_import_custom_builder(data_dir: str | Path | None, dataset_name: str) -> None:
    if data_dir is None:
        return

    builders_root = Path(data_dir).expanduser().resolve().parent / "tfds_builders"
    if not builders_root.is_dir():
        return

    builders_root_str = str(builders_root)
    if builders_root_str not in sys.path:
        sys.path.insert(0, builders_root_str)

    module_candidates = [dataset_name]
    for dirpath, _dirnames, filenames in os.walk(builders_root):
        rel_dir = Path(dirpath).relative_to(builders_root)
        for filename in sorted(filenames):
            if not filename.endswith(".py") or filename == "__init__.py":
                continue
            module_stem = filename[:-3]
            if str(rel_dir) == ".":
                module_name = module_stem
            else:
                module_name = ".".join(rel_dir.parts + (module_stem,))
            module_candidates.append(module_name)

    for module_name in dict.fromkeys(module_candidates):
        try:
            importlib.import_module(module_name)
        except Exception:
            continue


def _find_built_dataset_dir(data_dir: str | Path | None, dataset_name: str) -> str | None:
    if data_dir is None:
        return None

    dataset_root = Path(data_dir).expanduser().resolve() / dataset_name
    if not dataset_root.exists():
        return None

    candidates: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(dataset_root):
        if "dataset_info.json" in filenames:
            candidates.append(Path(dirpath))
    if candidates:
        candidates.sort(key=lambda path: (len(path.parts), str(path)))
        return str(candidates[-1])

    version_dirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    if version_dirs:
        return str(version_dirs[-1])
    return str(dataset_root)


def _load_tfds_dataset(tfds_name: str, split: str, data_dir: str | Path | None):
    configure_tfds_runtime()
    _maybe_import_custom_builder(data_dir, tfds_name)

    import tensorflow_datasets as tfds

    try:
        return tfds.load(tfds_name, split=split, data_dir=str(data_dir) if data_dir is not None else None, try_gcs=False)
    except Exception:
        built_dir = _find_built_dataset_dir(data_dir, tfds_name)
        if built_dir is None:
            raise
        builder = tfds.builder_from_directory(built_dir)
        return builder.as_dataset(split=split)


def resolve_tfds_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower()
    if normalized == "celebahq256":
        return "celebahq256"
    if normalized in {"celeb_a_hq/256", "celeb_a_hq_256"}:
        return "celeb_a_hq/256"
    if normalized.startswith("imagenet"):
        return "imagenet2012"
    if normalized == "lsunchurch":
        return "lsunc"
    return dataset_name


def infer_default_split(dataset_name: str, *, is_train: bool) -> str:
    normalized = dataset_name.strip().lower()
    if normalized == "celebahq256":
        return "train"
    if normalized in {"celeb_a_hq/256", "celeb_a_hq_256"}:
        return "train"
    if normalized.startswith("imagenet"):
        return "train" if is_train else "validation"
    if normalized == "lsunchurch":
        return "church-train" if is_train else "church-test"
    return "train"


def infer_constant_label(dataset_name: str) -> int | None:
    normalized = dataset_name.strip().lower()
    if normalized in {"celebahq256", "celeb_a_hq/256", "celeb_a_hq_256", "lsunchurch"}:
        return 0
    return None


def _image_preprocess_fn(
    *,
    image_size: int,
    random_flip: bool,
    channels_first: bool,
    pixel_range: str,
) -> Callable[[Any], tuple[Any, Any]]:
    import tensorflow as tf

    def _inner(example: Any) -> tuple[Any, Any]:
        image = example["image"]
        image = tf.convert_to_tensor(image)
        if image.shape.rank != 3:
            raise ValueError(f"TFDS image must be rank-3, got {image.shape!r}")

        shape = tf.shape(image)
        min_side = tf.minimum(shape[0], shape[1])
        image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
        if image_size > 0:
            image = tf.image.resize(image, (image_size, image_size), antialias=True)
        if random_flip:
            image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float32)
        if pixel_range == "minus_one_one":
            image = image / 127.5 - 1.0
        elif pixel_range == "zero_one":
            image = image / 255.0
        else:
            raise ValueError(f"Unsupported pixel_range: {pixel_range}")
        if channels_first:
            image = tf.transpose(image, perm=(2, 0, 1))

        label = example["label"] if "label" in example else tf.zeros((), dtype=tf.int32)
        label = tf.cast(label, tf.int32)
        return image, label

    return _inner


def _dataset_cardinality(dataset: Any) -> int:
    cardinality = int(dataset.cardinality().numpy())
    if cardinality < 0:
        raise ValueError(
            "Unable to infer TFDS dataset cardinality. Use a concrete split or provide a cached TFDS dataset."
        )
    return cardinality


@dataclass(slots=True)
class TorchStyleBatchLoader:
    iterator_factory: Callable[[], Iterator[tuple[np.ndarray, np.ndarray]]]
    num_examples: int
    num_batches: int
    split: str

    def __iter__(self):
        import torch

        for images, labels in self.iterator_factory():
            yield torch.from_numpy(images), torch.from_numpy(labels)

    def __len__(self) -> int:
        return self.num_batches


@dataclass(slots=True)
class NumpyBatchLoader:
    iterator_factory: Callable[[], Iterator[tuple[np.ndarray, np.ndarray]]]
    num_examples: int
    num_batches: int
    split: str

    def __iter__(self):
        yield from self.iterator_factory()

    def __len__(self) -> int:
        return self.num_batches


class LabelProxyDataset:
    def __init__(
        self,
        *,
        num_examples: int,
        image_size: int,
        num_classes: int,
        constant_label: int | None = None,
    ) -> None:
        if num_examples <= 0:
            raise ValueError("num_examples must be positive.")
        self.num_examples = int(num_examples)
        self.image_size = int(image_size)
        self.num_classes = max(1, int(num_classes))
        self.constant_label = constant_label

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int):
        import torch

        if index < 0 or index >= self.num_examples:
            raise IndexError(index)
        label = self.constant_label if self.constant_label is not None else int(index % self.num_classes)
        image = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        return image, label


def build_label_proxy_dataset(
    *,
    num_examples: int,
    image_size: int,
    dataset_name: str,
    num_classes: int,
) -> LabelProxyDataset:
    return LabelProxyDataset(
        num_examples=num_examples,
        image_size=image_size,
        num_classes=num_classes,
        constant_label=infer_constant_label(dataset_name),
    )


def build_torch_style_tfds_loader(
    *,
    dataset_name: str,
    data_dir: str | Path | None,
    split: str,
    batch_size: int,
    image_size: int,
    random_flip: bool,
    repeat: bool,
    shuffle: bool,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
    shuffle_buffer: int = 20_000,
    drop_remainder: bool = True,
    channels_first: bool = True,
    pixel_range: str = "minus_one_one",
) -> TorchStyleBatchLoader:
    configure_tfds_runtime()

    import tensorflow as tf
    import tensorflow_datasets as tfds

    tfds_name = resolve_tfds_dataset_name(dataset_name)
    dataset = _load_tfds_dataset(tfds_name, split=split, data_dir=data_dir)
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)
    num_examples = _dataset_cardinality(dataset)

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(max(1, num_examples), int(shuffle_buffer)),
            seed=int(seed),
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(
        _image_preprocess_fn(
            image_size=image_size,
            random_flip=random_flip,
            channels_first=channels_first,
            pixel_range=pixel_range,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(int(batch_size), drop_remainder=bool(drop_remainder))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    num_batches = (
        max(1, num_examples // max(1, int(batch_size)))
        if drop_remainder
        else max(1, int(np.ceil(num_examples / max(1, int(batch_size)))))
    )
    return TorchStyleBatchLoader(
        iterator_factory=lambda: iter(tfds.as_numpy(dataset)),
        num_examples=num_examples,
        num_batches=num_batches,
        split=split,
    )


def build_numpy_tfds_loader(
    *,
    dataset_name: str,
    data_dir: str | Path | None,
    split: str,
    batch_size: int,
    image_size: int,
    random_flip: bool,
    repeat: bool,
    shuffle: bool,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
    shuffle_buffer: int = 20_000,
    drop_remainder: bool = True,
    channels_first: bool = False,
    pixel_range: str = "minus_one_one",
) -> NumpyBatchLoader:
    configure_tfds_runtime()

    import tensorflow as tf
    import tensorflow_datasets as tfds

    tfds_name = resolve_tfds_dataset_name(dataset_name)
    dataset = _load_tfds_dataset(tfds_name, split=split, data_dir=data_dir)
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)
    num_examples = _dataset_cardinality(dataset)

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(max(1, num_examples), int(shuffle_buffer)),
            seed=int(seed),
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(
        _image_preprocess_fn(
            image_size=image_size,
            random_flip=random_flip,
            channels_first=channels_first,
            pixel_range=pixel_range,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(int(batch_size), drop_remainder=bool(drop_remainder))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    num_batches = (
        max(1, num_examples // max(1, int(batch_size)))
        if drop_remainder
        else max(1, int(np.ceil(num_examples / max(1, int(batch_size)))))
    )
    return NumpyBatchLoader(
        iterator_factory=lambda: iter(tfds.as_numpy(dataset)),
        num_examples=num_examples,
        num_batches=num_batches,
        split=split,
    )
