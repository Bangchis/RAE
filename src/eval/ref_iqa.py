from __future__ import annotations

import math
import sys

import torch
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from tqdm import trange

from .utils import LPIPS, to_torch_tensor


@torch.no_grad()
def calculate_psnr(arr1, arr2, bs, device: str = "cuda", disable_bar: bool = True) -> float:
    batch_count = arr1.shape[0]
    n_batches = math.ceil(batch_count / bs)
    psnr = torch.zeros(batch_count, device=device)

    for index in trange(n_batches, desc="Calculating PSNR", file=sys.stdout, disable=disable_bar):
        start = index * bs
        end = min((index + 1) * bs, batch_count)
        batch_arr1 = to_torch_tensor(arr1[start:end]).to(device)
        batch_arr2 = to_torch_tensor(arr2[start:end]).to(device)
        batch_mse = torch.mean((batch_arr1 - batch_arr2) ** 2, dim=[1, 2, 3]).clamp_min(1e-10)
        psnr[start:end] = 20.0 * torch.log10(1.0 / torch.sqrt(batch_mse))
    return float(psnr.mean().item())


@torch.no_grad()
def calculate_lpips(arr1, arr2, bs, device: str = "cuda", disable_bar: bool = True) -> float:
    batch_count = arr1.shape[0]
    n_batches = math.ceil(batch_count / bs)
    loss_fn = LPIPS().eval().to(device)
    values = torch.zeros(batch_count, device=device)

    for index in trange(n_batches, desc="Calculating LPIPS", file=sys.stdout, disable=disable_bar):
        start = index * bs
        end = min((index + 1) * bs, batch_count)
        batch_arr1 = (to_torch_tensor(arr1[start:end]).to(device) - 0.5) * 2.0
        batch_arr2 = (to_torch_tensor(arr2[start:end]).to(device) - 0.5) * 2.0
        values[start:end] = loss_fn(batch_arr1, batch_arr2).squeeze()
    return float(values.mean().item())


@torch.no_grad()
def calculate_ssim(arr1, arr2, bs, device: str = "cuda", disable_bar: bool = True) -> float:
    batch_count = arr1.shape[0]
    n_batches = math.ceil(batch_count / bs)
    ssim = torch.zeros(batch_count, device=device)

    for index in trange(n_batches, desc="Calculating SSIM", file=sys.stdout, disable=disable_bar):
        start = index * bs
        end = min((index + 1) * bs, batch_count)
        batch_arr1 = to_torch_tensor(arr1[start:end]).to(device)
        batch_arr2 = to_torch_tensor(arr2[start:end]).to(device)
        ssim[start:end] = structural_similarity_index_measure(
            target=batch_arr1,
            preds=batch_arr2,
            data_range=1.0,
            reduction="none",
        )
    return float(ssim.mean().item())
