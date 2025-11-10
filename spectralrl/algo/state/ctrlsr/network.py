import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.module.critic import RffReward
from spectralrl.module.misc import PositionalFeature
from spectralrl.module.net.residual import ResidualMLP
from spectralrl.module.noise_schedule import get_noise_schedule


class FactorizedNCE(nn.Module):
    def __init__(
        self,
        batch_size,
        obs_dim,
        action_dim,
        feature_dim,
        phi_hidden_dims,
        mu_hidden_dims,
        reward_hidden_dims,
        rff_dim=None,
        num_noises=0,
        linear=False,
        ranking=True,
        device="cpu"
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.rff_dim = rff_dim
        self.linear = linear
        self.ranking = ranking
        self.device = device
        if num_noises > 0:
            self.use_noise_perturbation = True
            self.N = num_noises
            self.betas, self.alphas, self.alphabars = get_noise_schedule(
                "vp", num_noises,
            )
            self.alphabars_prev = F.pad(self.alphabars[:-1], (1, 0), value=1.0)
            self.betas, self.alphas, self.alphabars, self.alphabars_prev = \
                self.betas[..., None].to(self.device), \
                self.alphas[..., None].to(self.device), \
                self.alphabars[..., None].to(self.device), \
                self.alphabars_prev[..., None].to(self.device)
        else:
            self.use_noise_perturbation = False
            self.N = 1

        self.mlp_t = nn.Sequential(
            PositionalFeature(128),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 128)
        )
        self.mlp_phi = ResidualMLP(
            input_dim=obs_dim+action_dim,
            output_dim=feature_dim,
            hidden_dims=phi_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device
        )
        self.mlp_mu = ResidualMLP(
            input_dim=obs_dim + (128 if self.use_noise_perturbation else 0),
            output_dim=feature_dim,
            hidden_dims=mu_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device
        )

        self.reward = RffReward(
            feature_dim=feature_dim,
            hidden_dims=reward_hidden_dims,
            linear=linear,
            rff_dim=rff_dim
        )

        if self.ranking:
            self.labels = torch.arange(batch_size).unsqueeze(0).repeat(self.N, 1).to(self.device)
        else:
            self.labels = torch.eye(batch_size).unsqueeze(0).repeat(self.N, 1, 1).to(self.device)
            self.normalizer = torch.nn.Parameter(torch.zeros([self.N, ], dtype=torch.float32, device=self.device))

    def forward_phi(self, s, a):
        x = torch.concat([s, a], dim=-1)
        x = self.mlp_phi(x)
        return x

    def forward_mu(self, sp, t=None):
        if t is not None:
            t_ff = self.mlp_t(t)
            sp = torch.concat([sp, t_ff], dim=-1)
        sp = self.mlp_mu(sp)
        return torch.nn.functional.tanh(sp)

    def compute_logits(self, s, a, sp, z_phi=None):
        B = sp.shape[0]
        if z_phi is None:
            z_phi = self.forward_phi(s, a)
        if self.use_noise_perturbation:
            sp = sp.unsqueeze(0).repeat(self.N, 1, 1)
            t = torch.arange(0, self.N).to(self.device)
            t = t.repeat_interleave(B).reshape(self.N, B)
            alphabars = self.alphabars[t]
            eps = torch.randn_like(sp)
            xt = alphabars.sqrt() * sp + (1 - alphabars).sqrt() * eps
            t = t.unsqueeze(-1)
        else:
            xt = sp.unsqueeze(0)
            t = None
        z_mu = self.forward_mu(xt, t) # (N, RB, z_dim)
        z_phi = z_phi.unsqueeze(0).repeat(z_mu.shape[0], 1, 1) # (N, LB, z_dim)
        logits = torch.bmm(z_phi, z_mu.transpose(-1, -2))      # (N, LB, RB)
        return logits

    def compute_loss(self, s, a, sp, r):
        B = s.shape[0]
        z_phi = self.forward_phi(s, a)
        logits = self.compute_logits(s, a, sp, z_phi)

        if self.linear:
            eff_logits = (torch.nn.functional.softplus(logits, beta=3.0) + 1e-6).log()
            if self.ranking:
                model_loss = F.cross_entropy(eff_logits, self.labels, reduction="none").mean(-1)
            else:
                eff_logits = eff_logits * self.normalizer.exp()[:, None, None] / B
                model_loss = F.binary_cross_entropy_with_logits(eff_logits, self.labels, reduction="none").mean([-2, -1])
        else:
            eff_logits = logits
            if self.ranking:
                model_loss = F.cross_entropy(eff_logits, self.labels, reduction="none").mean(-1)
            else:
                eff_logits = eff_logits + self.normalizer[:, None, None] - math.log(B)
                model_loss = F.binary_cross_entropy_with_logits(eff_logits, self.labels, reduction="none").mean([-2, -1])

        reward_loss = F.mse_loss(self.reward(z_phi), r)

        # compute info
        pos_logits = logits[torch.arange(self.N).unsqueeze(1), torch.arange(B), torch.arange(B).unsqueeze(0).repeat(self.N, 1)]  # (N, B1)
        pos_logits_per_noise = pos_logits.mean(dim=1)  # (N,)
        neg_logits = (logits.sum(dim=-1) - pos_logits) / (logits.shape[-1]-1)
        neg_logits_per_noise = neg_logits.mean(dim=-1)  # (N,)
        metrics = {
            "loss/model_loss": model_loss.mean().item(),
            "loss/reward_loss": reward_loss.item(),
            "misc/phi_norm": z_phi.abs().mean().item(),
            "misc/phi_std": z_phi.std(0).mean().item(),
        }
        checkpoints = list(range(0, self.N, 5))
        metrics.update({
            f"detail/pos_logits_{i}": pos_logits_per_noise[i].item() for i in checkpoints
        })
        metrics.update({
            f"detail/neg_logits_{i}": neg_logits_per_noise[i].item() for i in checkpoints
        })
        metrics.update({
            f"detail/logit_gap_{i}": (pos_logits_per_noise[i] - neg_logits_per_noise[i]).item() for i in checkpoints
        })
        metrics.update({
            f"detail/model_loss_{i}": model_loss[i].item() for i in checkpoints
        })
        return metrics, model_loss.mean(), reward_loss

    def compute_feature(self, s, a):
        return self.forward_phi(s, a)
