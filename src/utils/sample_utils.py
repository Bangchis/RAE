from typing import Callable
import torch
import torch_xla.core.xla_model as xm

"""
manual euler sampling with v prediction -- torchdiffeq doesn't support automatic xm.mark_step() and has problem in XLA devices
"""
def manual_sample(stage2_model, z_init, y, schedule, device: torch.device, model_kwargs = None) -> torch.Tensor:
    zt = z_init
    with torch.no_grad():
        iterator = range(len(schedule) - 1, 0, -1)
        for i in iterator:
            t_cur = torch.full((zt.shape[0],), schedule[i], device=device, dtype=zt.dtype)
            t_prev = torch.full((zt.shape[0],), schedule[i - 1], device=device, dtype=zt.dtype)
            v_pred = stage2_model(zt, t_cur, y, **(model_kwargs or {}))
            delta = (t_prev - t_cur).view(-1, 1, 1, 1)
            zt = zt + delta * v_pred
            xm.mark_step() # trigger compilation on XLA devices
    return zt

def make_timesteps(num_steps: int, t_min: float, t_max: float, shift: float) -> torch.Tensor:
    schedule = torch.linspace(t_min, t_max, steps=num_steps, dtype=torch.float32)
    schedule = shift * schedule / (1 + (shift - 1) * schedule)
    return schedule


def build_label_sampler(
    sampling_mode: str,
    num_classes: int,
    num_fid_samples: int,
    total_samples: int,
    samples_needed_this_device: int,
    batch_size: int,
    device: torch.device,
    rank: int,
    iterations: int,
    seed: int,
) -> Callable[[int], torch.Tensor]:
    if sampling_mode == "random":
        def random_sampler(_step_idx: int) -> torch.Tensor:
            return torch.randint(0, num_classes, (batch_size,), device=device)

        return random_sampler

    if sampling_mode == "equal":
        if num_fid_samples % num_classes != 0:
            raise ValueError(
                f"Equal label sampling requires num_fid_samples ({num_fid_samples}) "
                f"to be divisible by num_classes ({num_classes})."
            )

        labels_per_class = num_fid_samples // num_classes
        base_pool = torch.arange(num_classes, dtype=torch.long).repeat_interleave(labels_per_class)

        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(base_pool.numel(), generator=generator)
        base_pool = base_pool[permutation]

        if total_samples > num_fid_samples:
            tail = torch.randint(0, num_classes, (total_samples - num_fid_samples,), generator=generator)
            global_pool = torch.cat([base_pool, tail], dim=0)
        else:
            global_pool = base_pool

        start = rank * samples_needed_this_device
        end = start + samples_needed_this_device
        device_pool = global_pool[start:end]
        device_pool = device_pool.view(iterations, batch_size)

        def equal_sampler(step_idx: int) -> torch.Tensor:
            labels = device_pool[step_idx]
            return labels.to(device)

        return equal_sampler

    raise ValueError(f"Unknown label sampling mode: {sampling_mode}")
