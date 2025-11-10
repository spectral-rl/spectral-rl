import math

import torch
import torch.nn as nn


class PositionalFeature(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.dim = dim

    def forward(self, x):
        freqs = torch.arange(0, self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.to(freqs.dtype) @ freqs.unsqueeze(0)
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return x


class FourierFeature(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("weight", torch.randn(1, dim // 2) * scale)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)
