from __future__ import annotations

from collections import namedtuple
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models


CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from disc.lpips_utils import get_ckpt_path


def normalize_tensor(x, eps: float = 1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim: bool = True):
    return x.mean([2, 3], keepdim=keepdim)


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out: int = 1, use_dropout: bool = False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)


class VGG16(nn.Module):
    def __init__(self, requires_grad: bool = False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(weights="DEFAULT").features
        self.slice1 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(4)])
        self.slice2 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(4, 9)])
        self.slice3 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(9, 16)])
        self.slice4 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(16, 23)])
        self.slice5 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(23, 30)])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


class LPIPS(nn.Module):
    def __init__(self, use_dropout: bool = True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = VGG16(requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name: str = "vgg_lpips"):
        ckpt = get_ckpt_path(name)
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)

    def forward(self, input, target):
        input_scaled = self.scaling_layer(input)
        target_scaled = self.scaling_layer(target)
        outs_input, outs_target = self.net(input_scaled), self.net(target_scaled)
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        res = []
        for idx, (feat_input, feat_target) in enumerate(zip(outs_input, outs_target)):
            diff = (normalize_tensor(feat_input) - normalize_tensor(feat_target)) ** 2
            res.append(spatial_average(lins[idx].model(diff), keepdim=True))
        value = res[0]
        for diff in res[1:]:
            value += diff
        return value


class ImgArrDataset(Dataset):
    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return torch.from_numpy(self.arr[idx]).permute(2, 0, 1)


def to_torch_tensor(np_array):
    tensor = torch.from_numpy(np_array).permute(0, 3, 1, 2)
    if tensor.max() > 1.0:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
    return tensor
