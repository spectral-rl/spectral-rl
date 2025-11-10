from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

import torch
import torch.nn as nn

from spectralrl.module.net.basic import weight_init
from spectralrl.module.net.mlp import MLP, EnsembleMLP

ModuleType = Type[nn.Module]

class EnsembleQ(nn.Module):
    """
    A vanilla critic module, which can be used as Q(s, a) or V(s).

    Parameters
    ----------
    input_dim :  The dimensions of input.
    output_dim :  The dimension of critic's output.
    device :  The device which the model runs on. Default is cpu.
    ***(any args of MLP or EnsembleMLP)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int=1,
        device: Union[str, int, torch.device] = "cpu",
        *,
        ensemble_size: int=1,
        hidden_dims: Sequence[int] = [],
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        dropout: Optional[Union[float, Sequence[float]]] = None,
        share_hidden_layer: Union[Sequence[bool], bool] = False,
    ) -> None:
        super().__init__()
        self.critic_type = "Critic"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.ensemble_size = ensemble_size

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if ensemble_size == 1:
            self.output_layer = MLP(
                input_dim = input_dim,
                output_dim = output_dim,
                hidden_dims = hidden_dims,
                norm_layer = norm_layer,
                activation = activation,
                dropout = dropout,
                device = device
            )
        elif isinstance(ensemble_size, int) and ensemble_size > 1:
            self.output_layer = EnsembleMLP(
                input_dim = input_dim,
                output_dim = output_dim,
                hidden_dims = hidden_dims,
                norm_layer = norm_layer,
                activation = activation,
                dropout = dropout,
                device = device,
                ensemble_size = ensemble_size,
                share_hidden_layer = share_hidden_layer
            )
        else:
            raise ValueError(f"ensemble size should be int >= 1.")

    def forward(self, obs: torch.Tensor, action: Optional[torch.Tensor]=None, *args, **kwargs) -> torch.Tensor:
        """Compute the Q-value (when action is given) or V-value (when action is None).

        Parameters
        ----------
        obs :  The observation, should be torch.Tensor.
        action :  The action, should be torch.Tensor.

        Returns
        -------
        torch.Tensor :  Q(s, a) or V(s).
        """
        if action is not None:
            obs = torch.cat([obs, action], dim=-1)
        return self.output_layer(obs)


class RffLayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        rff_dim: int,
        learnable: bool = True
    ):
        super().__init__()
        assert rff_dim % 2 == 0, "rff_dim must be even"
        self.learnable = learnable
        if learnable:
            self.layer = nn.Linear(feature_dim, rff_dim//2)
        else:
            self.register_buffer("noise", torch.randn([feature_dim, rff_dim//2], requires_grad=False))

    def forward(self, x):
        if self.learnable:
            x = self.layer(x)
            return torch.concat([torch.sin(x), torch.cos(x)], dim=-1)
        else:
            x = x @ self.noise
            return torch.concat([torch.sin(x), torch.cos(x)], dim=-1)


class RffReward(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: list[int],
        linear: bool = False,
        rff_dim: int | None = None,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(feature_dim)
        if linear:
            self.rff_layer = nn.Identity()
            self.net = MLP(
                input_dim=feature_dim,
                output_dim=1,
                hidden_dims=hidden_dims,
                norm_layer=nn.LayerNorm,
                activation=nn.ELU
            )
        else:
            self.rff_layer = RffLayer(feature_dim, rff_dim, learnable=True)
            self.net = MLP(
                input_dim=rff_dim,
                output_dim=1,
                hidden_dims=hidden_dims,
                norm_layer=nn.LayerNorm,
                activation=nn.ELU
            )

    def forward(self, x):
        return self.net(self.rff_layer(self.ln(x)))


class RffDoubleQ(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int],
        linear: bool | None = None,
        rff_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.net1 = RffReward(feature_dim, hidden_dims, linear, rff_dim)
        self.net2 = RffReward(feature_dim, hidden_dims, linear, rff_dim)

    def forward(self, x):
        q1 = self.net1(x)
        q2 = self.net2(x)
        return torch.stack([q1, q2], dim=0)
