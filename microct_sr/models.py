from __future__ import annotations

import math
import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class FSRCNN(nn.Module):
    def __init__(self, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        layers = [
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(),
        ]
        for _ in range(m):
            layers.extend([nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU()])
        layers.extend([
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(d, 1, kernel_size=9, padding=4),
        ])
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class ResidualBlock(nn.Module):
    def __init__(self, num_features=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EDSR(nn.Module):
    def __init__(self, scale_factor=4, num_features=64, num_blocks=16, res_scale=0.1):
        super(EDSR, self).__init__()
        self.head = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, 1 * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        self.res_scale = res_scale

    def forward(self, x):
        x = self.head(x)
        res = self.body(x) * self.res_scale
        x = x + res
        x = self.tail(x)
        return x


class SimplifiedEDSR(nn.Module):
    def __init__(self, scale_factor=4, num_features=64, num_blocks=8, res_scale=0.1):
        super(SimplifiedEDSR, self).__init__()
        self.head = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, 1 * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        self.res_scale = res_scale

    def forward(self, x):
        x = self.head(x)
        res = self.body(x) * self.res_scale
        x = x + res
        x = self.tail(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_features=64, reduction=16):
        super(ChannelAttention, self).__init__()
        hidden = max(num_features // reduction, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_features, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class RCAB(nn.Module):
    def __init__(self, num_features=64, reduction=16):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction),
        )

    def forward(self, x):
        return x + self.body(x) * 0.1


class RCAN(nn.Module):
    def __init__(self, scale_factor=4, num_features=64, num_blocks=8, reduction=16):
        super(RCAN, self).__init__()
        self.head = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[RCAB(num_features, reduction) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, 1 * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


def model_uses_preupsampling(model_name: str) -> bool:
    return model_name.lower() in {"srcnn", "fsrcnn"}


def build_model(model_name: str, scale_factor: int, options: dict | None = None) -> nn.Module:
    options = options or {}
    name = model_name.lower()
    if name == "srcnn":
        return SRCNN()
    if name == "fsrcnn":
        return FSRCNN(d=options.get("d", 56), s=options.get("s", 12), m=options.get("m", 4))
    if name == "edsr":
        return EDSR(
            scale_factor=scale_factor,
            num_features=options.get("num_features", 64),
            num_blocks=options.get("num_blocks", 16),
            res_scale=options.get("res_scale", 0.1),
        )
    if name == "simplified_edsr":
        return SimplifiedEDSR(
            scale_factor=scale_factor,
            num_features=options.get("num_features", 64),
            num_blocks=options.get("num_blocks", 8),
            res_scale=options.get("res_scale", 0.1),
        )
    if name == "rcan":
        return RCAN(
            scale_factor=scale_factor,
            num_features=options.get("num_features", 64),
            num_blocks=options.get("num_blocks", 8),
            reduction=options.get("reduction", 16),
        )
    raise ValueError(f"Unknown model: {model_name}")
