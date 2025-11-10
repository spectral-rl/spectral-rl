from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.module.critic import RffReward
from spectralrl.module.misc import PositionalFeature
from spectralrl.module.net.residual import ResidualMLP
from spectralrl.module.noise_schedule import get_noise_schedule


class Encoder(nn.Module):
    def __init__(
        self,
        obs_dim,
        latent_dim: int=None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.convs = nn.Sequential(
            nn.Conv2d(obs_dim[0], 32, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ELU(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(torch.zeros(obs_dim, dtype=torch.float32)[None, ...])
            self.output_dim = self.convs(sample).flatten().shape[0]
        if self.latent_dim is not None:
            self.out = nn.Sequential(
                nn.Linear(self.output_dim, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
                nn.Tanh()
            )
            self.output_dim = latent_dim
        else:
            self.out = nn.Identity()

    def forward(self, obs):
        B = obs.shape[0]
        obs = obs / 255.0 - 0.5
        obs = self.convs(obs)
        obs = obs.reshape(B, -1)
        return self.out(obs)


class FactorizedNCE(nn.Module):
    def __init__(
        self,
        batch_size: int,
        obs_dim: int,
        action_dim: int,
        feature_dim: int,
        embed_dim: int,
        phi_hidden_dims: Sequence[int],
        mu_hidden_dims: Sequence[int],
        reward_hidden_dims: Sequence[int],
        rff_dim: Optional[int]=None,
        num_noises: int=0,
        device="cpu"
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.rff_dim = rff_dim
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

        self.network = nn.ModuleDict({
            "state_mlp": nn.Sequential(
                nn.Linear(obs_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Tanh(),
            ),
            "action_mlp": nn.Sequential(
                nn.Linear(action_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Tanh(),
            ),
            "t_mlp": nn.Sequential(
                PositionalFeature(128),
                nn.Linear(128, 256),
                nn.Mish(),
                nn.Linear(256, 128)
            ),
            "phi_mlp": ResidualMLP(
                input_dim=embed_dim*2,
                output_dim=feature_dim,
                hidden_dims=phi_hidden_dims,
                activation=nn.Mish(),
                dropout=None,
            ),
            "mu_mlp": ResidualMLP(
                input_dim=embed_dim + (128 if self.use_noise_perturbation else 0),
                output_dim=feature_dim,
                hidden_dims=mu_hidden_dims,
                activation=nn.Mish(),
                dropout=None,
                device=device
            ),
            "reward": RffReward(
                feature_dim=feature_dim,
                hidden_dims=reward_hidden_dims,
                rff_dim=rff_dim,
                linear=False
            )
        })

        self.labels = torch.arange(batch_size).unsqueeze(0).repeat(self.N, 1).to(self.device)

    def forward_phi(self, s, a):
        s = self.network["state_mlp"](s)
        a = self.network["action_mlp"](a)
        all = torch.concat([s, a], dim=-1)
        return self.network["phi_mlp"](all)

    def forward_mu(self, sp):
        sp = self.network["state_mlp"](sp).detach()
        if self.use_noise_perturbation:
            sp = sp.unsqueeze(0).repeat(self.N, 1, 1)
            t = torch.arange(0, self.N).to(self.device)
            t = t.repeat_interleave(sp.shape[1]).reshape(self.N, sp.shape[1])
            alphabars = self.alphabars[t]
            eps = torch.randn_like(sp)
            sp = alphabars.sqrt() * sp + (1 - alphabars).sqrt() * eps
            t = t.unsqueeze(-1)
        else:
            sp = sp.unsqueeze(0)
            t = None
        if t is not None:
            t_ff = self.network["t_mlp"](t)
            sp = torch.concat([sp, t_ff], dim=-1)
        sp = self.network["mu_mlp"](sp)
        return torch.nn.functional.tanh(sp)

    def compute_logits(self, z_phi, z_mu):
        z_phi = z_phi.unsqueeze(0).repeat(z_mu.shape[0], 1, 1)
        logits = torch.bmm(z_phi, z_mu.transpose(-1, -2))
        return logits

    def compute_losses(self, s, a, sp, r):
        B = s.shape[0]

        # s and sp are encoded
        z_phi = self.forward_phi(s, a)
        z_mu = self.forward_mu(sp)
        logits = self.compute_logits(z_phi, z_mu)
        model_loss = F.cross_entropy(logits, self.labels, reduction="none").mean(-1)
        reward_loss = F.mse_loss(self.network["reward"](z_phi), r)

        pos_logits = logits[torch.arange(self.N).unsqueeze(1), torch.arange(B), torch.arange(B).unsqueeze(0).repeat(self.N, 1)]  # (N, B1)
        pos_logits_per_noise = pos_logits.mean(dim=1)  # (N,)
        neg_logits = (logits.sum(dim=-1) - pos_logits) / (logits.shape[-1]-1)
        neg_logits_per_noise = neg_logits.mean(dim=-1)  # (N,)
        metrics = {
            "loss/model_loss": model_loss.mean().item(),
            "loss/reward_loss": reward_loss.item(),
            "misc/phi_norm": z_phi.abs().mean().item(),
            "misc/phi_std": z_phi.std(0).mean().item(),
            "misc/next_latent_norm": z_mu.abs().mean().item(),
            "misc/next_latent_mean": z_mu.mean().item(),
            "misc/next_latent_std": z_mu.std(0).mean().item(),
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
        return model_loss.mean(), reward_loss, metrics

    def compute_feature(self, s, a):
        return self.forward_phi(s, a)
