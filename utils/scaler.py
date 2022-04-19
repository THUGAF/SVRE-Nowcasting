import torch


def minmax_norm(tensor: torch.Tensor, vmax: float, vmin: float) -> torch.Tensor:
    tensor[tensor > vmax] = vmax
    tensor[tensor < vmin] = vmin
    tensor = ((tensor - vmin) / (vmax - vmin))
    return tensor


def reverse_minmax_norm(tensor: torch.Tensor, vmax: float, vmin: float) -> torch.Tensor:
    tensor[tensor > 1.0] = 1.0
    tensor[tensor < 0.0] = 0.0
    tensor = tensor * (vmax - vmin) + vmin
    return tensor


def convert_to_gray(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor * 255
    tensor = tensor.to(torch.uint8)
    return tensor
