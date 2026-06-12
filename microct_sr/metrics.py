from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def mse_value(output: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.mse_loss(output.detach(), target.detach()).cpu().item())


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def psnr_value(output: torch.Tensor, target: torch.Tensor) -> float:
    return psnr_from_mse(mse_value(output, target))
