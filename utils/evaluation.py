from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import utils.ssim as ssim
import utils.transform as transform


def _count(pred: torch.Tensor, truth: torch. Tensor, threshold: float) -> Tuple[int, int, int, int]:
    assert pred.size() == truth.size()          # (B, L, C, H, W)
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = pred.mean(dim=1), truth.mean(dim=1)
    # truth > th & pred > th
    hits = torch.sum(torch.logical_and(truth >= threshold, pred >= threshold)).item()
    # truth > th & pred < th
    misses = torch.sum(torch.logical_and(truth >= threshold, pred < threshold)).item()
    # truth < th & pred > th
    false_alarms = torch.sum(torch.logical_and(truth < threshold, pred >= threshold)).item()
    # truth < th & pred < th
    correct_rejections = torch.sum(torch.logical_and(truth < threshold, pred < threshold)).item()
    return hits, misses, false_alarms, correct_rejections


def evaluate_forecast(pred: torch.Tensor, truth: torch.Tensor, threshold: float, eps: float = 1e-4) \
    -> Tuple[float, float, float]:
    h, m, f, c = _count(pred, truth, threshold)
    pod = h / (h + m + eps)
    far = f / (h + f + eps)
    csi = h / (h + m + f + eps)
    return pod, far, csi


def evaluate_mbe(pred: torch.Tensor, truth: torch.Tensor) -> float:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = pred.mean(dim=1), truth.mean(dim=1)
    mbe = torch.mean(pred - truth).item()
    return mbe


def evaluate_mae(pred: torch.Tensor, truth: torch.Tensor) -> float:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = pred.mean(dim=1), truth.mean(dim=1)
    mae = torch.mean(torch.abs(pred - truth)).item()
    return mae


def evaluate_rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = pred.mean(dim=1), truth.mean(dim=1)
    rmse = torch.sqrt(torch.mean(torch.square(pred - truth))).item()
    return rmse


def evaluate_ssim(pred: torch.Tensor, truth: torch.Tensor) -> float:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = pred.mean(dim=1), truth.mean(dim=1)
    pred, truth = transform.convert_to_gray(pred), transform.convert_to_gray(truth)
    pred, truth = pred.float(), truth.float()
    ssim_ = ssim.ssim(pred, truth).item()
    return ssim_


def evaluate_jsd(pred: torch.Tensor, truth: torch.Tensor) -> float:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = pred.mean(dim=1), truth.mean(dim=1)
    pred_batch_flatten, truth_batch_flatten = pred.flatten(start_dim=1), truth.flatten(start_dim=1)
    pred_batch_flatten, truth_batch_flatten = pred_batch_flatten.softmax(dim=1), truth_batch_flatten.softmax(dim=1)
    log_mean_batch_flatten = ((pred_batch_flatten + truth_batch_flatten) / 2).log()
    jsd = 0.5 * F.kl_div(log_mean_batch_flatten, truth_batch_flatten, reduction='batchmean') + \
        0.5 * F.kl_div(log_mean_batch_flatten, pred_batch_flatten, reduction='batchmean')
    jsd = jsd.item()
    return jsd
