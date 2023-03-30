from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import utils.ssim as ssim
import utils.transform as transform


def _count(pred: torch.Tensor, truth: torch. Tensor, threshold: float) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert pred.size() == truth.size()          # (B, L, C, H, W)
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(1)
    hits = np.zeros(seq_len)
    misses = np.zeros(seq_len)
    false_alarms = np.zeros(seq_len)
    correct_rejections = np.zeros(seq_len)
    for s in range(seq_len):
        stat = 2 * (truth[:, s] > threshold).int() + (pred[:, s] > threshold).int()
        hits[s] = torch.sum(stat == 3).item()                   # truth > th & pred > th
        misses[s] = torch.sum(stat == 2).item()                 # truth > th & pred < th
        false_alarms[s] = torch.sum(stat == 1).item()           # truth < th & pred > th
        correct_rejections[s] = torch.sum(stat == 0).item()     # truth < th & pred < th
    return hits, misses, false_alarms, correct_rejections


def evaluate_forecast(pred: torch.Tensor, truth: torch.Tensor, threshold: float, eps: float = 1e-4) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, m, f, c = _count(pred, truth, threshold)
    pod = h / (h + m + eps)
    far = f / (h + f + eps)
    csi = h / (h + m + f + eps)
    return pod, far, csi


def evaluate_mbe(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(1)
    mbes = np.zeros(seq_len)
    for s in range(seq_len):
        mbes[s] = torch.mean(pred[:, s] - truth[:, s]).item()
    return mbes


def evaluate_mae(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(1)
    maes = np.zeros(seq_len)
    for s in range(seq_len):
        maes[s] = torch.mean(torch.abs(pred[:, s] - truth[:, s])).item()
    return maes


def evaluate_rmse(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(1)
    rmses = np.zeros(seq_len)
    for s in range(seq_len):
        rmses[s] = torch.sqrt(torch.mean(torch.square(pred[:, s] - truth[:, s]))).item()
    return rmses


def evaluate_ssim(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = transform.convert_to_gray(pred), transform.convert_to_gray(truth)
    pred, truth = pred.float(), truth.float()
    seq_len = pred.size(1)
    ssims = np.zeros(seq_len)
    for s in range(seq_len):
        ssims[s] = ssim.ssim(pred[:, s], truth[:, s])
    return ssims


def evaluate_jsd(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(1)
    jsds = np.zeros(seq_len)
    for s in range(seq_len):
        pred_batch_flatten, truth_batch_flatten = pred[:, s].flatten(start_dim=1), truth[:, s].flatten(start_dim=1)
        pred_batch_flatten, truth_batch_flatten = pred_batch_flatten.softmax(dim=1), truth_batch_flatten.softmax(dim=1)
        log_mean_batch_flatten = ((pred_batch_flatten + truth_batch_flatten) / 2).log()
        jsds[s] = 0.5 * F.kl_div(log_mean_batch_flatten, truth_batch_flatten, reduction='batchmean') + \
            0.5 * F.kl_div(log_mean_batch_flatten, pred_batch_flatten, reduction='batchmean')
    return jsds
