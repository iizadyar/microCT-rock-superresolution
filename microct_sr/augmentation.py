from __future__ import annotations

import random
import torch


def paired_augment(lr: torch.Tensor, hr: torch.Tensor, horizontal_flip=True, vertical_flip=True, rot90=True) -> tuple[torch.Tensor, torch.Tensor]:
    if horizontal_flip and random.random() < 0.5:
        lr = torch.flip(lr, dims=[2])
        hr = torch.flip(hr, dims=[2])
    if vertical_flip and random.random() < 0.5:
        lr = torch.flip(lr, dims=[1])
        hr = torch.flip(hr, dims=[1])
    if rot90:
        k = random.randint(0, 3)
        if k:
            lr = torch.rot90(lr, k, dims=[1, 2])
            hr = torch.rot90(hr, k, dims=[1, 2])
    return lr.contiguous(), hr.contiguous()
