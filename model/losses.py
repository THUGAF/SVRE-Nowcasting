from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.scaler as scalar


def biased_mse_loss(pred: torch.Tensor, truth: torch.Tensor, vmax: float, vmin: float) -> torch.Tensor:
    points = scalar.minmax_norm(torch.tensor([10.0, 20.0, 30.0, 40.0]), vmax, vmin)
    weight = (truth < points[0]) * 1 \
        + (torch.logical_and(truth >= points[0], truth < points[1])) * 2 \
        + (torch.logical_and(truth >= points[1], truth < points[2])) * 5 \
        + (torch.logical_and(truth >= points[2], truth < points[3])) * 10 \
        + (truth >= points[3]) * 30
    return torch.mean(weight * (pred - truth) ** 2)


def biased_mae_loss(pred: torch.Tensor, truth: torch.Tensor, vmax: float, vmin: float) -> torch.Tensor:
    points = scalar.minmax_norm(torch.tensor([10.0, 20.0, 30.0, 40.0]), vmax, vmin)
    weight = (truth < points[0]) * 1 \
        + (torch.logical_and(truth >= points[0], truth < points[1])) * 2 \
        + (torch.logical_and(truth >= points[1], truth < points[2])) * 5 \
        + (torch.logical_and(truth >= points[2], truth < points[3])) * 10 \
        + (truth >= points[3]) * 30
    return torch.mean(weight * torch.abs(pred - truth))


def cv_loss(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_cv = torch.std(pred, dim=(1, 2, 3, 4)) / (torch.mean(pred, dim=(1, 2, 3, 4)) + eps)
    truth_cv = torch.std(truth, dim=(1, 2, 3, 4)) / (torch.mean(truth, dim=(1, 2, 3, 4)) + eps)
    return F.l1_loss(pred_cv, truth_cv)


def cal_d_loss(fake_score: torch.Tensor, real_score: torch.Tensor, loss_func: Callable = nn.MSELoss()) -> torch.Tensor:
    """Calculate loss function of the discriminators.

    Args:
        fake_score: Score of fake.
        real_score: Score of real.

    Returns:
        torch.Tensor: Loss of discriminator.
    """
    label = torch.ones_like(fake_score).type_as(fake_score)
    loss_pred = loss_func(fake_score, label * 0.0)
    loss_truth = loss_func(real_score, label * 1.0)
    d_loss = (loss_pred + loss_truth) / 2
    return d_loss


def cal_g_loss(fake_score: torch.Tensor, loss_func=nn.MSELoss()) -> torch.Tensor:
    """Calculate loss function of the generator.

    Args:
        fake_score: Score of fake.

    Returns:
        torch.Tensor: Loss of the generator.
    """

    label = torch.ones_like(fake_score).type_as(fake_score)
    g_loss = loss_func(fake_score, label * 1.0)
    return g_loss
