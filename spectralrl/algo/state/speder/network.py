import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.module.critic import RffReward
from spectralrl.module.misc import PositionalFeature
from spectralrl.module.net.mlp import MLP
from spectralrl.module.net.residual import ResidualMLP
from spectralrl.module.noise_schedule import get_noise_schedule


class FactorizedTransition(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        feature_dim,
        phi_hidden_dims,
        mu_hidden_dims,
        reward_hidden_dims,
        num_noises=0,
        lam=1.0,
        device="cpu"
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.lam = lam
        self.device = device

        if num_noises > 0:
            self.use_noise_perturbation = True
            self.num_noises = num_noises
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
            linear=True,
            rff_dim=None,
        )

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

    def compute_loss(self, s, a, sp, r):
        B = sp.shape[0]
        N = 1 if self.num_noises <= 0 else self.num_noises
        z_phi = self.forward_phi(s, a)

        # reward loss
        reward_loss = F.mse_loss(self.compute_reward(z_phi), r)

        # model loss
        if self.use_noise_perturbation:
            sp = sp.unsqueeze(0).repeat(self.num_noises, 1, 1)
            t = torch.arange(0, self.num_noises).to(self.device)
            t = t.repeat_interleave(B).reshape(self.num_noises, B)
            alphabars = self.alphabars[t]
            eps = torch.randn_like(sp)
            tilde_sp = alphabars.sqrt() * sp + (1 - alphabars).sqrt() * eps
            t = t.unsqueeze(-1)
        else:
            tilde_sp = sp.unsqueeze(0)
            t = None

        z_tilde_sp = self.forward_mu(tilde_sp, t)
        z_phi = z_phi.unsqueeze(0).repeat(z_tilde_sp.shape[0], 1, 1)
        inner = torch.bmm(z_phi, z_tilde_sp.transpose(-1, -2)) # (N, B, B)

        loss_pt1 = torch.diagonal(inner, dim1=-2, dim2=-1)
        loss_pt2 = inner.pow(2)
        # loss_pt2 = loss_pt2.mean(dim=[-2, -1])
        loss_pt2 = (loss_pt2.sum(dim=[-2, -1]) - torch.diagonal(loss_pt2, dim1=-2, dim2=-1).sum(-1)) / (B*(B-1)) # exclude the diagonal elements since they are not sampled from marginal dist.

        model_loss = self.lam * loss_pt2.mean() - 2*loss_pt1.mean()

        # metrics for logging
        with torch.no_grad():
            pos_probs = loss_pt1.mean(dim=-1)
            neg_probs = (inner.sum(dim=[-2, -1]) - loss_pt1.sum(dim=-1)) / (B*(B-1))

        metrics = {
            "loss/model_loss": model_loss.item(),
            "loss/reward_loss": reward_loss.item(),
            "misc/phi_norm": z_phi[0].abs().mean().item(),
            "misc/phi_std": z_phi[0].std(0).mean().item(),
        }
        checkpoints = list(range(0, N, 5))
        metrics.update({
            f"detail/pos_prob_{i}": pos_probs[i].item() for i in checkpoints
        })
        metrics.update({
            f"detail/neg_prob_{i}": neg_probs[i].item() for i in checkpoints
        })
        metrics.update({
            f"detail/prob_gap_{i}": (pos_probs[i] - neg_probs[i]).item() for i in checkpoints
        })
        metrics.update({
            f"detail/neg_loss_{i}": loss_pt2[i].item() for i in checkpoints
        })
        return metrics, model_loss, reward_loss

    def compute_feature(self, s, a):
        return self.forward_phi(s, a)

    def compute_reward(self, z_phi):
        return self.reward(z_phi)
