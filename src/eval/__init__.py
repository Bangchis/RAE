from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .fid import calculate_gfid, calculate_rfid
from .ref_iqa import calculate_lpips, calculate_psnr, calculate_ssim


def compute_reconstruction_metrics(
    ref_arr: np.ndarray,
    rec_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    metrics_to_compute=("psnr", "ssim", "rfid"),
    disable_bar: bool = True,
) -> Dict[str, float]:
    device_str = "cuda" if device.type == "cuda" else "cpu"
    results: dict[str, float] = {}
    if "psnr" in metrics_to_compute:
        results["psnr"] = calculate_psnr(ref_arr, rec_arr, batch_size, device_str, disable_bar=disable_bar)
    if "ssim" in metrics_to_compute:
        results["ssim"] = calculate_ssim(ref_arr, rec_arr, batch_size, device_str, disable_bar=disable_bar)
    if "lpips" in metrics_to_compute:
        results["lpips"] = calculate_lpips(ref_arr, rec_arr, batch_size, device_str, disable_bar=disable_bar)
    if "rfid" in metrics_to_compute:
        results["rfid"] = calculate_rfid(ref_arr, rec_arr, batch_size, device_str)
    if not results:
        raise ValueError("No evaluation metrics were requested.")
    return results


def compute_generation_metrics(
    ref_arr: np.ndarray,
    rec_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, float]:
    device_str = "cuda" if device.type == "cuda" else "cpu"
    return {"fid": calculate_gfid(rec_arr, ref_arr, batch_size, device_str)}


@torch.no_grad()
def evaluate_reconstruction_distributed(
    model,
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    experiment_dir: str,
    global_step: int,
    autocast_kwargs: dict,
    metric_batch_size: int = 128,
    reference_npz_path: Optional[str] = None,
    metrics_to_compute: Optional[list] = ("psnr", "ssim", "rfid"),
) -> Optional[Dict[str, float]]:
    temp_dir = os.path.join(experiment_dir, "eval_npzs")
    if rank == 0:
        os.makedirs(temp_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    total = min(len(val_dataset), num_samples)
    chunk = total // world_size
    start = rank * chunk
    end = total if rank == world_size - 1 else (rank + 1) * chunk
    subset = Subset(val_dataset, list(range(start, end)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    reconstructions = []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Reconstructing", file=sys.stdout) if rank == 0 else loader
    with torch.inference_mode():
        for images, _labels in iterator:
            images = images.to(device, non_blocking=True)
            with autocast(**autocast_kwargs):
                recon = model(images)
            recon_np = recon.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            reconstructions.extend(list(recon_np))

    shard_array = np.stack(reconstructions) if reconstructions else np.zeros((0, 1, 1, 3), dtype=np.uint8)
    shard_path = os.path.join(temp_dir, f"recon_{global_step:07d}_{rank:02d}.npz")
    np.savez(shard_path, arr_0=shard_array)
    if dist.is_initialized():
        dist.barrier()

    metrics = None
    if rank == 0:
        all_recons = [
            np.load(os.path.join(temp_dir, f"recon_{global_step:07d}_{worker_rank:02d}.npz"))["arr_0"]
            for worker_rank in range(world_size)
        ]
        combined_recons = np.concatenate(all_recons, axis=0)[:num_samples]
        if reference_npz_path is None or not os.path.exists(reference_npz_path):
            raise FileNotFoundError(f"Reference NPZ not found at {reference_npz_path}")
        ref_images = np.load(reference_npz_path)["arr_0"]
        metrics = compute_reconstruction_metrics(
            ref_images,
            combined_recons,
            device,
            batch_size=metric_batch_size,
            metrics_to_compute=metrics_to_compute,
            disable_bar=True,
        )
        for worker_rank in range(world_size):
            os.remove(os.path.join(temp_dir, f"recon_{global_step:07d}_{worker_rank:02d}.npz"))

    if dist.is_initialized():
        dist.barrier()
    return metrics
