import torch
import numpy as np


def minmax_norm(tensor: torch.Tensor, vmax: float = 70.0, vmin: float = 0.0) -> torch.Tensor:
    tensor = torch.clip(tensor, vmin, vmax)
    tensor = ((tensor - vmin) / (vmax - vmin))
    return tensor


def inverse_minmax_norm(tensor: torch.Tensor, vmax: float = 70.0, vmin: float = 0.0) -> torch.Tensor:
    tensor = torch.clip(tensor, 0.0, 1.0)
    tensor = tensor * (vmax - vmin) + vmin
    return tensor


def convert_to_gray(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor * 255
    tensor = tensor.to(torch.uint8)
    return tensor


def ref_to_R(ref: torch.Tensor, a: float = 238, b: float = 1.57) -> torch.Tensor:
    Z = 10 ** (ref / 10)
    R = ((1 / a) * Z) ** (1 / b)
    R = torch.clip(R, 0.0, torch.inf)
    return R