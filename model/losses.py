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


def ssd(tensor: torch.Tensor) -> torch.Tensor:
    left_pad = F.pad(tensor, (1, 0, 0, 0))
    right_pad = F.pad(tensor, (0, 1, 0, 0))
    up_pad = F.pad(tensor, (0, 0, 1, 0))
    bottom_pad = F.pad(tensor, (0, 0, 0, 1))

    diff_h = left_pad - right_pad
    diff_v = up_pad - bottom_pad
    ssd = torch.sum(diff_h[:, :, 1:-1] ** 2) + torch.sum(diff_v[:, :, 1:-1] ** 2)
    return ssd


def ssd_loss(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    pred, truth = pred.cpu(), truth.cpu()
    seq_len, batch_size = pred.size(0), pred.size(1)
    
    pred_ssd_list = []
    truth_ssd_list = []
    for s in range(seq_len):
        pred_ssd = ssd(pred[s]) / batch_size
        truth_ssd = ssd(truth[s]) / batch_size
        pred_ssd_list.append(pred_ssd)
        truth_ssd_list.append(truth_ssd)
    
    return F.l1_loss(torch.tensor(pred_ssd_list), torch.tensor(truth_ssd_list))


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
