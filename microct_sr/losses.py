from __future__ import annotations

import torch.nn as nn

def build_loss(name: str = "mse") -> nn.Module:
    if name.lower() != "mse":
        raise ValueError("Only MSE loss is used in this workflow.")
    return nn.MSELoss()
