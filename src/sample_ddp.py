# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Distributed sampler for stage-2 models using torch-xla devices.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from PIL import Image
from omegaconf import OmegaConf
from torch_xla.amp import autocast
from tqdm import tqdm

from stage1 import RAE  # noqa: E402
from stage2.models import Stage2ModelProtocol  # noqa: E402
from utils.train_utils import initialize_cache, set_random_seed  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import parse_configs  # noqa: E402
from utils.sample_utils import build_label_sampler, manual_sample, make_timesteps  # noqa: E402
from utils.fid_utils import calculate_fid_from_path, write_fid_result  # noqa: E402
from tqdm import tqdm  # noqa: E402


def create_npz_from_sample_folder(sample_dir: str, num: int = 50_000) -> str:
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def parse_guidance_value(cfg: Dict[str, Any], key: str, default: float) -> float:
    if key in cfg:
        return cfg[key]
    dashed_key = key.replace("_", "-")
    return cfg.get(dashed_key, default)

@torch.no_grad()
def run_sampling(args: argparse.Namespace) -> None:
    initialize_cache(is_sample=True)

    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    set_random_seed(args.global_seed)
    torch.set_grad_enabled(False)
    if rank == 0:
        print(f"Starting rank={rank}, seed={args.global_seed}, world_size={world_size}, device={device}.")
    autocast_kwargs = dict(device=device, dtype=torch.bfloat16, enabled=args.precision == "bf16")
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        _,
    ) = parse_configs(args.config)
    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 entries.")

    def to_dict(cfg_section: Optional[OmegaConf]) -> Dict[str, Any]:
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)  # type: ignore[return-value]

    sampler_cfg = to_dict(sampler_config)
    guidance_cfg = to_dict(guidance_config)
    misc_cfg = to_dict(misc_config)

    latent_size = tuple(int(dim) for dim in misc_cfg.get("latent_size", (768, 16, 16)))
    shift_dim = misc_cfg.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc_cfg.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    if rank == 0:
        print(f"Using time_dist_shift={time_dist_shift:.4f}.")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    rae.eval()
    model.eval()

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    if sampler_mode != "ODE":
        raise ValueError(f"Manual sampling only supports ODE mode, but got {sampler_mode}.")
    sampler_params = dict(sampler_cfg.get("params", {}))
    num_steps = int(sampler_params.get("num_steps", sampler_params.get("steps", 50)))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    guidance_method = guidance_cfg.get("method", "cfg")
    t_min = parse_guidance_value(guidance_cfg, "t_min", 0.0)
    t_max = parse_guidance_value(guidance_cfg, "t_max", 1.0)
    schedule = make_timesteps(num_steps, 1/1000, 1, time_dist_shift)

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guid_model_cfg = guidance_cfg.get("guidance_model")
        if guid_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guid_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    num_classes = int(misc_cfg.get("num_classes", 1000))
    null_label = int(misc_cfg.get("null_label", num_classes))

    model_target = model_config.get("target", "stage2")
    model_string_name = str(model_target).split(".")[-1]
    ckpt_path = model_config.get("ckpt")
    ckpt_string_name = "pretrained" if not ckpt_path else os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    sampling_method = sampler_params.get("sampling_method", "na")
    num_steps_str = sampler_params.get("num_steps", sampler_params.get("steps", "na"))
    guidance_tag = f"cfg-{guidance_scale:.2f}"
    base_components = [model_string_name, ckpt_string_name, guidance_tag, f"bs{args.per_proc_batch_size}"]
    if sampler_mode == "ODE":
        detail_components = [sampler_mode, str(num_steps_str), str(sampling_method), args.precision]
    else:
        diffusion_form = sampler_params.get("diffusion_form", "na")
        last_step = sampler_params.get("last_step", "na")
        last_step_size = sampler_params.get("last_step_size", "na")
        detail_components = [
            sampler_mode,
            str(num_steps_str),
            str(sampling_method),
            str(diffusion_form),
            str(last_step),
            str(last_step_size),
            args.precision,
        ]
    folder_name = "-".join(component.replace(os.sep, "-") for component in base_components + detail_components)
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    xm.rendezvous("mkdir")

    per_rank_batch = args.per_proc_batch_size
    global_batch_size = per_rank_batch * world_size
    existing = [
        name
        for name in os.listdir(sample_folder_dir)
        if os.path.isfile(os.path.join(sample_folder_dir, name)) and name.endswith(".png")
    ]
    num_samples = len(existing)
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    if total_samples % world_size != 0:
        raise ValueError("Total samples must be divisible by world size.")
    samples_needed_this_device = total_samples // world_size
    if samples_needed_this_device % per_rank_batch != 0:
        raise ValueError("Per-rank sample count must be divisible by the per-device batch size.")
    iterations = samples_needed_this_device // per_rank_batch
    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
    total = (num_samples // world_size) * world_size

    label_sampler = build_label_sampler(
        args.label_sampling,
        num_classes,
        args.num_fid_samples,
        total_samples,
        samples_needed_this_device,
        per_rank_batch,
        device,
        rank,
        iterations,
        args.global_seed,
    )

    using_cfg = guidance_scale > 1.0

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    for step_idx in pbar:
        z = torch.randn(per_rank_batch, *latent_size, device=device, dtype=dtype)
        y = label_sampler(step_idx)

        sample_fn = model.forward
        model_kwargs: Optional[Dict[str, Any]] = None
        z_for_model = z
        y_for_model = y

        if using_cfg:
            if guidance_method == "autoguidance":
                if guid_model_forward is None:
                    raise RuntimeError("Guidance model forward is not initialized.")
                sample_fn = model.forward_with_autoguidance
                model_kwargs = {
                    "cfg_scale": guidance_scale,
                    "cfg_interval": (t_min, t_max),
                    "additional_model_forward": guid_model_forward,
                }
            elif guidance_method == "cfg":
                z_for_model = torch.cat([z, z], dim=0)
                y_null = torch.full((per_rank_batch,), null_label, device=device, dtype=torch.long)
                y_for_model = torch.cat([y, y_null], dim=0)
                sample_fn = model.forward_with_cfg
                model_kwargs = {
                    "cfg_scale": guidance_scale,
                    "cfg_interval": (t_min, t_max),
                }
            else:
                raise ValueError(f"Unsupported guidance method '{guidance_method}'.")

        with autocast(**autocast_kwargs):
            latents = manual_sample(
                sample_fn,
                z_for_model,
                y_for_model,
                schedule,
                device=device,
                model_kwargs=model_kwargs,
            )

        if using_cfg and guidance_method == "cfg":
            latents, _ = latents.chunk(2, dim=0)

        decoded = rae.decode(latents).clamp_(0, 1)
        samples = xm._maybe_convert_to_cpu(decoded.float().mul(255).permute(0, 2, 3, 1)).to(torch.uint8).numpy()

        for local_idx, sample in enumerate(samples):
            index = local_idx * world_size + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        total += global_batch_size
        xm.rendezvous(f"sample_step_{step_idx}")
        xm.mark_step()

    xm.rendezvous("sampling_done")
    if rank == 0:
        npz_path = create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        if args.fid_ref is not None:
            print(f"Computing FID from {npz_path} against {args.fid_ref} on {args.fid_device}...")
            fid_value = calculate_fid_from_path(
                npz_path,
                args.fid_ref,
                batch_size=args.fid_batch_size,
                device=args.fid_device,
                num_threads=args.fid_num_threads,
            )
            result_path = write_fid_result(
                f"{sample_folder_dir}.fid.json",
                fid=fid_value,
                sample_path=npz_path,
                reference_path=args.fid_ref,
                batch_size=args.fid_batch_size,
                device=args.fid_device,
                num_samples=args.num_fid_samples,
                num_threads=args.fid_num_threads,
            )
            print(f"FID: {fid_value:.6f}")
            print(f"Saved FID result to {result_path}")
        print("Done.")
    xm.rendezvous("npz_done")


def _mp_worker(rank: int, args: argparse.Namespace) -> None:
    del rank
    run_sampling(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--sample-dir", type=str, default="samples", help="Directory to store PNG samples.")
    parser.add_argument("--per-proc-batch-size", type=int, default=4)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="fp32",
        help="Computation precision for sampling.",
    )
    parser.add_argument(
        "--label-sampling",
        type=str,
        choices=["random", "equal"],
        default="random",
        help="Choose how to sample class labels when generating images.",
    )
    parser.add_argument(
        "--fid-ref",
        type=str,
        default=None,
        help="Optional path to a reference statistics .npz file with mu/sigma to compute FID after sampling.",
    )
    parser.add_argument(
        "--fid-device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for FID feature extraction when --fid-ref is set.",
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=64,
        help="Batch size for FID feature extraction when --fid-ref is set.",
    )
    parser.add_argument(
        "--fid-num-threads",
        type=int,
        default=None,
        help="Optional torch CPU thread count for FID feature extraction when --fid-ref is set.",
    )

    parsed_args = parser.parse_args()
    xmp.spawn(_mp_worker, args=(parsed_args,), start_method="fork")
