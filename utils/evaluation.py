from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import utils.ssim as ssim
import utils.scaler as scaler


def _count(pred: torch.Tensor, truth: torch. Tensor, threshold: float) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(1)
    
    hits = []
    misses = []
    false_alarms = []
    correct_rejections = []

    for s in range(seq_len):
        stat = 2 * (truth[:, s] > threshold).int() + (pred[:, s] > threshold).int()
        hit = torch.sum(stat == 3).item()
        miss = torch.sum(stat == 2).item()
        false_alarm = torch.sum(stat == 1).item()
        correct_rejection = torch.sum(stat == 0).item()
        hits.append(hit)
        misses.append(miss)
        false_alarms.append(false_alarm)
        correct_rejections.append(correct_rejection)

    return np.array(hits), np.array(misses), np.array(false_alarms), np.array(correct_rejections)


def evaluate_forecast(pred: torch.Tensor, truth: torch.Tensor, threshold: float, eps: float = 1e-4) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""To calculate POD, FAR, CSI for the prediction at each time step.
    
    Args:
        pred (torch.Tensor): The prediction sequence in tensor form with 5D shape `(S, B, C, H, W)`.
        truth (torch.Tensor): The ground truth sequence in tensor form with 5D shape `(S, B, C, H, W)`.
        threshold (float, optional): The threshold of POD, FAR, CSI. Range: (0, 1).
    
    Return:
        numpy.ndarray: POD at each time step.
        numpy.ndarray: FAR at each time step.
        numpy.ndarray: CSI at each time step.
    """

    h, m, f, c = _count(pred, truth, threshold)
    pod = h / (h + m + eps)
    far = f / (h + f + eps)
    csi = h / (h + m + f + eps)
    return pod, far, csi


def evaluate_me(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(1)

    me_list = []
    for s in range(seq_len):
        me = torch.mean(pred[:, s] - truth[:, s])
        me_list.append(me)

    return np.array(me_list)


def evaluate_mae(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(1)

    mae_list = []
    for s in range(seq_len):
        mae = F.l1_loss(pred[:, s], truth[:, s])
        mae_list.append(mae)

    return np.array(mae_list)


def evaluate_kld(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(1)

    kld_list = []
    for s in range(seq_len):
        pred_batch_flatten, truth_batch_flatten = pred[:, s].flatten(start_dim=1), truth[:, s].flatten(start_dim=1)
        pred_batch_flatten, truth_batch_flatten = pred[:, s].softmax(dim=-1), truth[:, s].softmax(dim=-1)
        kld = F.kl_div(pred_batch_flatten, truth_batch_flatten)
        kld_list.append(kld)
    
    return np.array(kld_list)


def evaluate_ssim(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = scaler.convert_to_gray(pred), scaler.convert_to_gray(truth)
    pred, truth = pred.float(), truth.float()
    seq_len = pred.size(1)

    ssim_list = []
    for s in range(seq_len):
        ssim_ = ssim.ssim(pred[:, s], truth[:, s])
        ssim_list.append(ssim_)

    return np.array(ssim_list)
