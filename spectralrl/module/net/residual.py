from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float=0.0,
        activation: nn.Module=nn.ReLU,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.activation = activation
        if input_dim != hidden_dim:
            self.residual = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual = nn.Identity()
        if dropout and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dropout(x)
        x = self.ln(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x + self.residual(residual)

class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int=0,
        hidden_dims: Sequence[int]=[],
        activation: Callable=F.relu,
        dropout: float=None,
        device: str="cpu",
    ):
        super().__init__()

        assert len(hidden_dims) > 0, "hidden_dims must be a list of at least one integer"
        self.activation = activation
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        hidden_dims = [hidden_dims[0]] + list(hidden_dims)
        self.blocks = nn.Sequential(*[
            ResidualMLPBlock(
                hidden_dims[i],
                hidden_dims[i+1],
                dropout,
                activation,
            ) for i in range(len(hidden_dims) - 1)
        ])

    def forward(self, x):
        x = self.fc1(x)
        x = self.blocks(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class SimbaBlock(nn.Module):
    """Pre-LN residual block with an inverted bottleneck (SimBa, Lee et al., 2025):
    x + W2 relu(W1 LN(x)), W1: d -> expansion * d, W2: expansion * d -> d."""
    def __init__(self, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion)
        self.fc2 = nn.Linear(hidden_dim * expansion, hidden_dim)
        for fc in (self.fc1, self.fc2):  # he_normal, as in the reference implementation
            nn.init.kaiming_normal_(fc.weight, nonlinearity="relu")
            nn.init.zeros_(fc.bias)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.ln(x))))


class SimbaEncoder(nn.Module):
    """SimBa encoder: linear embedding -> ``num_blocks`` SimbaBlocks -> LayerNorm (post-LN)."""
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int, expansion: int = 4):
        super().__init__()
        self.output_dim = hidden_dim
        self.embed = nn.Linear(input_dim, hidden_dim)
        nn.init.orthogonal_(self.embed.weight, gain=1.0)
        nn.init.zeros_(self.embed.bias)
        self.blocks = nn.Sequential(*[SimbaBlock(hidden_dim, expansion) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.ln(self.blocks(self.embed(x)))
