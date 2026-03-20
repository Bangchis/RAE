from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from .vendor import activate_backend
except ImportError:
    from vendor import activate_backend


IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
    )

    arr = np.asarray(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def list_image_files(root: str | Path) -> list[Path]:
    base_path = Path(root).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Image directory not found: {base_path}")
    if not base_path.is_dir():
        raise ValueError(f"Expected an image directory, got: {base_path}")

    image_paths: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(base_path.rglob(pattern))
        image_paths.extend(base_path.rglob(pattern.upper()))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {base_path}")
    return image_paths


class RawImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, image_paths: list[Path], image_size: int | None) -> None:
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")
            if self.image_size is not None:
                image = center_crop_arr(image, self.image_size)
            arr = np.asarray(image, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1), 0


def _build_detector() -> tuple[object, object]:
    activate_backend()

    from ml_collections import ConfigDict
    from eval import utils as eval_utils

    config = ConfigDict({"eval": ConfigDict({"detector": "inception"})})
    return eval_utils.get_detector(config)


def _compute_moments_from_loader(loader: DataLoader[tuple[torch.Tensor, int]]) -> tuple[np.ndarray, np.ndarray, int]:
    import jax

    detector_params, detector = _build_detector()
    local_device_count = max(1, jax.local_device_count())

    total_count = 0
    feature_sum: np.ndarray | None = None
    feature_outer: np.ndarray | None = None

    for batch_images, _labels in loader:
        images = batch_images.permute(0, 2, 3, 1).numpy()
        real_count = int(images.shape[0])
        if real_count == 0:
            continue

        padded_count = int(math.ceil(real_count / local_device_count) * local_device_count)
        if padded_count != real_count:
            pad = padded_count - real_count
            images = np.concatenate([images, np.repeat(images[-1:], pad, axis=0)], axis=0)

        sharded = images.reshape(local_device_count, -1, *images.shape[1:])
        features = np.asarray(jax.device_get(detector(detector_params, sharded)[0]), dtype=np.float64)[:real_count]

        if feature_sum is None:
            feature_sum = features.sum(axis=0)
            feature_outer = features.T @ features
        else:
            feature_sum += features.sum(axis=0)
            feature_outer += features.T @ features
        total_count += real_count

    if total_count == 0 or feature_sum is None or feature_outer is None:
        raise RuntimeError("No images were processed while building backend FID statistics.")

    mu = feature_sum / total_count
    denom = max(total_count - 1, 1)
    sigma = (feature_outer - total_count * np.outer(mu, mu)) / denom
    return mu, sigma, total_count


def calculate_backend_reference_stats(
    image_root: str | Path,
    *,
    image_size: int | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    image_paths = list_image_files(image_root)
    dataset = RawImageDataset(image_paths, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return _compute_moments_from_loader(loader)


def write_backend_reference_stats(
    output_path: str | Path,
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    num_samples: int,
    source: str | Path,
) -> Path:
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix in {".pkl", ".pickle"}:
        payload = {
            "fid": {
                "mu": np.asarray(mu, dtype=np.float64),
                "sigma": np.asarray(sigma, dtype=np.float64),
            },
            "meta": {
                "source": str(Path(source).expanduser().resolve()),
                "num_samples": int(num_samples),
                "detector": "diffuse_nnx.eval.inception.InceptionV3",
            },
        }
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle)
        return out_path

    if out_path.suffix == ".npz":
        np.savez_compressed(
            out_path,
            mu=np.asarray(mu, dtype=np.float64),
            sigma=np.asarray(sigma, dtype=np.float64),
            num_samples=np.asarray([int(num_samples)], dtype=np.int64),
        )
        return out_path

    raise ValueError("Backend JAX FID stats must be saved as .pkl, .pickle, or .npz.")
