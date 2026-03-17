from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def load_image_array(path: str | Path) -> np.ndarray:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Image array file not found: {input_path}")
    if input_path.suffix == ".npy":
        return np.load(input_path)
    if input_path.suffix == ".npz":
        archive = np.load(input_path)
        if "arr_0" in archive.files:
            return archive["arr_0"]
        if "images" in archive.files:
            return archive["images"]
        raise KeyError(f"{input_path} does not contain 'arr_0' or 'images'. Keys={archive.files}")
    raise ValueError(f"Unsupported image array format: {input_path.suffix}")


def load_reference_stats(
    reference_path: str | Path,
    ref_mu_key: str = "mu",
    ref_sigma_key: str = "sigma",
) -> tuple[np.ndarray, np.ndarray]:
    stats_path = Path(reference_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Reference statistics file not found: {stats_path}")
    if stats_path.suffix != ".npz":
        raise ValueError("Reference statistics must be a .npz file containing mu/sigma arrays.")

    reference_stats = np.load(stats_path)
    if ref_mu_key not in reference_stats.files or ref_sigma_key not in reference_stats.files:
        raise KeyError(
            f"Missing '{ref_mu_key}'/'{ref_sigma_key}' in {stats_path}. "
            f"Available keys: {list(reference_stats.files)}"
        )
    return reference_stats[ref_mu_key], reference_stats[ref_sigma_key]


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA FID evaluation, but no CUDA device is available.")
    return device


@contextmanager
def torch_num_threads(num_threads: Optional[int], device: str) -> Iterator[None]:
    if device != "cpu" or num_threads is None:
        yield
        return
    if num_threads <= 0:
        raise ValueError("num_threads must be greater than 0 when provided.")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


class ImagePathDataset(Dataset[torch.Tensor]):
    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1)


def list_image_files(root: str | Path) -> list[Path]:
    base_path = Path(root)
    if not base_path.exists():
        raise FileNotFoundError(f"Image directory not found: {base_path}")
    if not base_path.is_dir():
        raise ValueError(f"Expected an image directory, got: {base_path}")

    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    image_paths = []
    for pattern in patterns:
        image_paths.extend(base_path.rglob(pattern))
        image_paths.extend(base_path.rglob(pattern.upper()))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {base_path}")
    return image_paths


def _fid_from_moments(
    mu_gen: np.ndarray,
    sigma_gen: np.ndarray,
    mu_ref: np.ndarray,
    sigma_ref: np.ndarray,
) -> float:
    scipy_linalg = __import__("scipy.linalg", fromlist=["sqrtm"])

    mu_gen = np.asarray(mu_gen, dtype=np.float64)
    mu_ref = np.asarray(mu_ref, dtype=np.float64)
    sigma_gen = np.asarray(sigma_gen, dtype=np.float64)
    sigma_ref = np.asarray(sigma_ref, dtype=np.float64)

    diff = mu_gen - mu_ref
    covmean = scipy_linalg.sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_gen + sigma_ref - 2.0 * covmean)
    return float(max(fid, 0.0))


@torch.no_grad()
def _compute_inception_moments_from_batches(
    batches: Any,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    extractor = FeatureExtractorInceptionV3(
        name="inception-v3-compat",
        features_list=["2048"],
    ).to(device).eval()

    total_count = 0
    feature_sum: Optional[torch.Tensor] = None
    feature_outer: Optional[torch.Tensor] = None

    for batch in batches:
        if isinstance(batch, np.ndarray):
            batch_tensor = torch.from_numpy(batch)
        else:
            batch_tensor = batch
        if batch_tensor.ndim != 4:
            raise ValueError(f"Expected a 4D image tensor, got shape {tuple(batch_tensor.shape)}.")
        if batch_tensor.shape[-1] == 3:
            batch_tensor = batch_tensor.permute(0, 3, 1, 2)
        if batch_tensor.dtype != torch.uint8:
            batch_tensor = batch_tensor.to(torch.float32)
            if float(batch_tensor.max()) <= 1.5:
                batch_tensor = batch_tensor * 255.0
            batch_tensor = batch_tensor.clamp(0, 255).to(torch.uint8)

        batch_tensor = batch_tensor.to(device=device, dtype=torch.uint8)
        feature = extractor(batch_tensor)[0].detach().cpu().to(torch.float64)

        if feature_sum is None:
            feature_sum = feature.sum(dim=0)
            feature_outer = feature.T @ feature
        else:
            feature_sum += feature.sum(dim=0)
            feature_outer += feature.T @ feature
        total_count += feature.shape[0]

    if total_count == 0 or feature_sum is None or feature_outer is None:
        raise RuntimeError("No samples were provided for FID moment computation.")

    mu = (feature_sum / total_count).numpy()
    denom = max(total_count - 1, 1)
    sigma = ((feature_outer - total_count * torch.outer(feature_sum / total_count, feature_sum / total_count)) / denom).numpy()
    return mu, sigma


def _compute_inception_moments_from_arr(
    arr: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D image array, got shape {arr.shape}.")

    batches = []
    for start_idx in range(0, arr.shape[0], batch_size):
        batches.append(arr[start_idx : start_idx + batch_size])
    return _compute_inception_moments_from_batches(batches, device=device)


def calculate_stats_from_image_folder(
    image_root: str | Path,
    batch_size: int = 64,
    device: str = "auto",
    num_workers: int = 0,
    num_threads: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    resolved_device = _resolve_device(device)
    image_paths = list_image_files(image_root)
    loader = DataLoader(
        ImagePathDataset(image_paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(resolved_device == "cuda"),
        drop_last=False,
    )
    with torch_num_threads(num_threads, resolved_device):
        mu, sigma = _compute_inception_moments_from_batches(loader, device=resolved_device)
    return mu, sigma, len(image_paths)


def calculate_stats_from_array(
    arr: np.ndarray,
    batch_size: int = 64,
    device: str = "auto",
    num_threads: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    resolved_device = _resolve_device(device)
    with torch_num_threads(num_threads, resolved_device):
        mu, sigma = _compute_inception_moments_from_arr(arr, batch_size=batch_size, device=resolved_device)
    return mu, sigma, int(arr.shape[0])


def calculate_fid_from_array(
    arr: np.ndarray,
    reference_path: str | Path,
    batch_size: int = 64,
    device: str = "auto",
    ref_mu_key: str = "mu",
    ref_sigma_key: str = "sigma",
    num_threads: Optional[int] = None,
) -> float:
    resolved_device = _resolve_device(device)
    mu_ref, sigma_ref = load_reference_stats(reference_path, ref_mu_key=ref_mu_key, ref_sigma_key=ref_sigma_key)
    with torch_num_threads(num_threads, resolved_device):
        mu_gen, sigma_gen = _compute_inception_moments_from_arr(arr, batch_size=batch_size, device=resolved_device)
    return _fid_from_moments(mu_gen, sigma_gen, mu_ref, sigma_ref)


def calculate_fid_from_path(
    sample_path: str | Path,
    reference_path: str | Path,
    batch_size: int = 64,
    device: str = "auto",
    ref_mu_key: str = "mu",
    ref_sigma_key: str = "sigma",
    num_threads: Optional[int] = None,
) -> float:
    images = load_image_array(sample_path)
    return calculate_fid_from_array(
        images,
        reference_path=reference_path,
        batch_size=batch_size,
        device=device,
        ref_mu_key=ref_mu_key,
        ref_sigma_key=ref_sigma_key,
        num_threads=num_threads,
    )


def iter_archive_batches(sample_paths: Sequence[str | Path], batch_size: int) -> Iterator[np.ndarray]:
    for sample_path in sample_paths:
        arr = load_image_array(sample_path)
        if arr.ndim != 4:
            raise ValueError(f"Expected a 4D image array, got shape {arr.shape} for {sample_path}.")
        for start_idx in range(0, arr.shape[0], batch_size):
            yield arr[start_idx : start_idx + batch_size]


def calculate_fid_from_archives(
    sample_paths: Sequence[str | Path],
    reference_path: str | Path,
    batch_size: int = 64,
    device: str = "auto",
    ref_mu_key: str = "mu",
    ref_sigma_key: str = "sigma",
    num_threads: Optional[int] = None,
) -> float:
    resolved_device = _resolve_device(device)
    mu_ref, sigma_ref = load_reference_stats(reference_path, ref_mu_key=ref_mu_key, ref_sigma_key=ref_sigma_key)
    with torch_num_threads(num_threads, resolved_device):
        mu_gen, sigma_gen = _compute_inception_moments_from_batches(
            iter_archive_batches(sample_paths, batch_size=batch_size),
            device=resolved_device,
        )
    return _fid_from_moments(mu_gen, sigma_gen, mu_ref, sigma_ref)


def count_images_in_archives(sample_paths: Sequence[str | Path]) -> int:
    total = 0
    for sample_path in sample_paths:
        total += int(load_image_array(sample_path).shape[0])
    return total


def compute_fid_with_metadata(
    sample_paths: Sequence[str | Path],
    reference_path: str | Path,
    batch_size: int = 64,
    device: str = "auto",
    ref_mu_key: str = "mu",
    ref_sigma_key: str = "sigma",
    num_threads: Optional[int] = None,
) -> tuple[float, int]:
    fid = calculate_fid_from_archives(
        sample_paths,
        reference_path=reference_path,
        batch_size=batch_size,
        device=device,
        ref_mu_key=ref_mu_key,
        ref_sigma_key=ref_sigma_key,
        num_threads=num_threads,
    )
    return fid, count_images_in_archives(sample_paths)


def write_fid_result(
    output_path: str | Path,
    *,
    fid: float,
    sample_path: str | Path,
    reference_path: str | Path,
    batch_size: int,
    device: str,
    num_samples: Optional[int] = None,
    num_threads: Optional[int] = None,
) -> Path:
    destination = Path(output_path)
    payload: dict[str, Any] = {
        "fid": fid,
        "sample_path": str(sample_path),
        "reference_path": str(reference_path),
        "batch_size": batch_size,
        "device": device,
    }
    if num_samples is not None:
        payload["num_samples"] = num_samples
    if num_threads is not None:
        payload["num_threads"] = num_threads
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


def write_reference_stats(
    output_path: str | Path,
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    num_samples: int,
    source: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, mu=mu, sigma=sigma, num_samples=np.array(num_samples), source=np.array(str(source)))
    return destination
