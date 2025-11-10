import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.module.critic import RffReward
from spectralrl.module.misc import PositionalFeature
from spectralrl.module.net.residual import ResidualMLP
from spectralrl.module.noise_schedule import get_noise_schedule


class DDPM(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        embed_dim,
        feature_dim,
        phi_hidden_dims,
        mu_hidden_dims,
        reward_hidden_dims,
        rff_dim,
        sample_steps,
        x_min=-10.0,
        x_max=10.0,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.sample_steps = sample_steps
        self.x_min, self.x_max = x_min, x_max
        self.betas, self.alphas, self.alphabars = get_noise_schedule(
            "vp", sample_steps,
        )
        self.alphabars_prev = F.pad(self.alphabars[:-1], (1, 0), value=1.0)
        self.betas, self.alphas, self.alphabars, self.alphabars_prev = \
            self.betas[..., None].to(self.device), \
            self.alphas[..., None].to(self.device), \
            self.alphabars[..., None].to(self.device), \
            self.alphabars_prev[..., None].to(self.device)

        self.mlp_s = nn.Sequential(
            nn.Linear(obs_dim, embed_dim*2),
            nn.Mish(),
            nn.Linear(embed_dim*2, embed_dim),
        )
        self.mlp_a = nn.Sequential(
            nn.Linear(action_dim, embed_dim*2),
            nn.Mish(),
            nn.Linear(embed_dim*2, embed_dim),
        )
        self.mlp_t = nn.Sequential(
            PositionalFeature(embed_dim),
            nn.Linear(embed_dim, embed_dim*2),
            nn.Mish(),
            nn.Linear(embed_dim*2, embed_dim),
        )
        self.mlp_phi = ResidualMLP(
            input_dim=embed_dim*2,
            output_dim=feature_dim,
            hidden_dims=phi_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device,
        )
        self.mlp_mu = ResidualMLP(
            input_dim=obs_dim+embed_dim,
            output_dim=obs_dim*feature_dim,
            hidden_dims=mu_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device,
        )
        self.reward = RffReward(
            feature_dim=feature_dim,
            hidden_dims=reward_hidden_dims,
            linear=False,
            rff_dim=rff_dim,
        )

    def add_noise(self, x0):
        noise_idx = torch.randint(0, self.sample_steps, (x0.shape[0], )).to(self.device)
        alphabars = self.alphabars[noise_idx]
        eps = torch.randn_like(x0)
        xt = alphabars.sqrt() * x0 + (1 - alphabars).sqrt() * eps
        return xt, noise_idx, eps

    def forward_phi(self, s, a):
        s_ff = self.mlp_s(s)
        a_ff = self.mlp_a(a)
        x = torch.concat([s_ff, a_ff], dim=-1)
        x = self.mlp_phi(x)
        return x

    def forward_mu(self, sp, t):
        t_ff = self.mlp_t(t)
        all = torch.concat([sp, t_ff], dim=-1)
        all = self.mlp_mu(all)
        return all.reshape(-1, self.feature_dim, self.obs_dim)

    def forward_eps(self, s=None, a=None, sp=None, t=None, z_phi=None, z_mu=None):
        if z_phi is None:
            z_phi = self.forward_phi(s=s, a=a)
        if z_mu is None:
            z_mu = self.forward_mu(sp=sp, t=t)
        eps = torch.bmm(z_phi.unsqueeze(1), z_mu).squeeze(1)
        return eps

    def compute_loss(self, s, a, sp, r):
        x0 = sp
        xt, t, eps = self.add_noise(x0)

        z_phi = self.forward_phi(s=s, a=a)
        z_mu = self.forward_mu(sp=xt.detach(), t=t.unsqueeze(-1))
        eps_pred = self.forward_eps(z_phi=z_phi, z_mu=z_mu)
        diffusion_loss = (eps_pred - eps).pow(2).sum(-1).mean()

        reward_loss = F.mse_loss(self.reward(z_phi), r)

        metrics = ({
            "info/eps_l1_norm": eps.abs().mean(),
            "info/eps_pred_l1_norm": eps_pred.abs().mean(),
            "info/x0_mean": x0.mean(),
            "info/x0_std": x0.std(0).mean(),
            "info/x0_l1_norm": x0.abs().mean(),
            "loss/diffusion_loss": diffusion_loss.item(),
            "loss/reward_loss": reward_loss.item(),
        })
        return metrics, diffusion_loss, reward_loss

    def compute_feature(self, s, a):
        return self.forward_phi(s, a)
