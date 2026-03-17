import argparse
import hashlib
import math
import os
from typing import Any, Optional

import torch
import torch_xla.core.xla_model as xm
import wandb
from torchvision.utils import make_grid


def is_main_process():
    return xm.is_master_ordinal()

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(
    args,
    entity,
    exp_name,
    project_name,
    run_config: Optional[dict[str, Any]] = None,
):
    config_dict = {"cli": namespace_to_dict(args)}
    if run_config is not None:
        config_dict["config"] = run_config
    if "WANDB_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
    )


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None, key="samples/ema"):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({key: wandb.Image(sample)}, step=step)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(0,1))
    x = x.clamp(0, 1).mul(255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x
