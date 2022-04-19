import os
import torch


def save_tensors(input_: torch.Tensor, pred: torch.Tensor, truth: torch.Tensor, root: str, stage: str):
    torch.save(input_.cpu(), os.path.join(root, 'images', stage, 'input', 'input.pt'))
    torch.save(pred.cpu(), os.path.join(root, 'images', stage, 'pred', 'pred.pt'))
    torch.save(truth.cpu(), os.path.join(root, 'images', stage, 'truth', 'truth.pt'))
