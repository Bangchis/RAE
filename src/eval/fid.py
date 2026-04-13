from __future__ import annotations

import numpy as np
import scipy.linalg
import torch
from torch_fidelity import calculate_metrics
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

from .utils import ImgArrDataset


def _fid_from_moments(mu1, sigma1, mu2, sigma2) -> float:
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(max(fid, 0.0))


@torch.no_grad()
def _compute_inception_moments_from_arr(arr: np.ndarray, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D array, got {arr.shape}")
    x = arr
    if x.shape[-1] == 3:
        x = np.transpose(x, (0, 3, 1, 2))
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        if x.max() <= 1.5:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)

    features = []
    extractor = FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    tensor = torch.from_numpy(x).to(device=device, dtype=torch.uint8)
    for start in range(0, tensor.shape[0], batch_size):
        batch = tensor[start : start + batch_size]
        features.append(extractor(batch)[0].detach().cpu())
    feats = torch.cat(features, dim=0).double().numpy()
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def calculate_gfid(
    arr1: np.ndarray,
    ref_arr: dict,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    mu_gen, sigma_gen = _compute_inception_moments_from_arr(arr1, batch_size=batch_size, device=device)
    return _fid_from_moments(mu_gen, sigma_gen, ref_arr["mu"], ref_arr["sigma"])


def calculate_rfid(
    arr1: np.ndarray,
    arr2: np.ndarray | None = None,
    bs: int = 64,
    device: str = "cuda",
    fid_statistics_file: str | None = None,
) -> float:
    arr1_ds = ImgArrDataset(arr1)
    if fid_statistics_file is not None:
        metrics = calculate_metrics(
            input1=arr1_ds,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            batch_size=bs,
            fid=True,
            cuda=(device == "cuda"),
        )
    else:
        if arr2 is None:
            raise ValueError("Either arr2 or fid_statistics_file must be provided.")
        arr2_ds = ImgArrDataset(arr2)
        metrics = calculate_metrics(
            input1=arr1_ds,
            input2=arr2_ds,
            batch_size=bs,
            fid=True,
            cuda=(device == "cuda"),
        )
    return float(metrics["frechet_inception_distance"])
