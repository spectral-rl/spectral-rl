import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.module.critic import RffReward
from spectralrl.module.misc import PositionalFeature
from spectralrl.module.net.residual import ResidualMLP
from spectralrl.module.noise_schedule import get_noise_schedule


class Diffusion(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg,
        device
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.feature_dim = cfg.feature_dim
        self.sample_steps = cfg.sample_steps
        self.x_min, self.x_max = cfg.x_min, cfg.x_max

        self.betas, self.alphas, self.alphabars = get_noise_schedule(
            "vp", cfg.sample_steps,
        )
        self.alphabars_prev = F.pad(self.alphabars[:-1], (1, 0), value=1.0)

        self.betas, self.alphas, self.alphabars, self.alphabars_prev = \
            self.betas[..., None].to(self.device), \
            self.alphas[..., None].to(self.device), \
            self.alphabars[..., None].to(self.device), \
            self.alphabars_prev[..., None].to(self.device)

        self.network = nn.ModuleDict({
            "action_mlp": nn.Sequential(
                nn.Linear(action_dim, cfg.embed_dim),
                nn.LayerNorm(cfg.embed_dim),
                nn.Tanh(),
            ),
            "t_mlp": nn.Sequential(
                PositionalFeature(128),
                nn.Linear(128, cfg.embed_dim),
                nn.LayerNorm(cfg.embed_dim),
                nn.ELU(),
            ),
            "phi_mlp": ResidualMLP(
                input_dim=obs_dim+cfg.embed_dim,
                output_dim=cfg.feature_dim,
                hidden_dims=cfg.phi_hidden_dims,
                activation=nn.Mish(),
                dropout=None,
            ),
            "mu_mlp": ResidualMLP(
                input_dim=obs_dim+cfg.embed_dim,
                output_dim=cfg.feature_dim*obs_dim,
                hidden_dims=cfg.mu_hidden_dims,
                activation=nn.Mish(),
                dropout=None,
            ),
            "reward": RffReward(
                feature_dim=cfg.feature_dim,
                hidden_dims=cfg.reward_hidden_dims,
                linear=False,
                rff_dim=cfg.rff_dim,
            ),
        })

    def add_noise(self, x0):
        noise_idx = torch.randint(0, self.sample_steps, (x0.shape[0], )).to(self.device)
        alphabars = self.alphabars[noise_idx]
        eps = torch.randn_like(x0)
        xt = alphabars.sqrt() * x0 + (1 - alphabars).sqrt() * eps
        return xt, noise_idx, eps

    def forward_phi(self, s, a):
        a = self.network["action_mlp"](a)
        x = torch.concat([s, a], dim=-1)
        x = self.network["phi_mlp"](x)
        return x

    def forward_mu(self, sp, t):
        t = self.network["t_mlp"](t)
        x = torch.concat([sp, t], dim=-1)
        x = self.network["mu_mlp"](x)
        x = torch.nn.functional.tanh(x)
        return x.reshape(-1, self.feature_dim, self.obs_dim)

    def forward_diffusion(self, s=None, a=None, sp=None, t=None, zsa=None, zsp=None):
        if zsa is None:
            zsa = self.forward_phi(s=s, a=a)
        if zsp is None:
            zsp = self.forward_mu(sp=sp, t=t)
        return torch.bmm(zsa.unsqueeze(1), zsp).squeeze(1)

    def compute_losses(self, obs, action, next_obs, reward):
        x0 = next_obs
        xt, t, eps = self.add_noise(x0)

        zsa = self.forward_phi(s=obs, a=action)
        zsp = self.forward_mu(sp=xt.detach(), t=t.unsqueeze(-1))
        noise_pred = self.forward_diffusion(zsa=zsa, zsp=zsp)
        diffusion_loss = (noise_pred - eps).pow(2).sum(-1).mean()
        if reward is not None:
            reward_loss = F.mse_loss(self.network["reward"](zsa), reward)
        else:
            reward_loss = torch.tensor(0.0)
        metrics = {
            "info/next_obs_mean": x0.mean().item(),
            "info/next_obs_std": x0.std(0).mean().item(),
            "info/next_obs_l1_norm": x0.abs().mean().item(),
            "loss/diffusion_loss": diffusion_loss.item(),
            "loss/reward_loss": reward_loss.item(),
            "info/zsa_std": zsa.std(0).mean().item(),
            "info/zsa_l1_norm": zsa.abs().mean().item(),
            "info/zsa_mean": zsa.mean().item(),
            "info/zsp_l1_norm": zsp.abs().mean().item(),
            "info/zsp_std": zsp.std(0).mean().item(),
            "info/zsp_mean": zsp.mean().item(),
        }
        return diffusion_loss, reward_loss, metrics

    def compute_feature(self, s, a):
        return self.forward_phi(s=s, a=a)

    def sample(self, shape, s, a, preserve_history=False):
        info = {}

        xt = torch.randn(shape, device=self.device)
        zsa = self.forward_phi(s=s, a=a)
        if preserve_history:
            info["sample_history"] = [xt.clone()]

        def reverse(xt, t):
            z = torch.randn_like(xt)
            timestep = torch.full([xt.shape[0], ], t, dtype=torch.int64, device=self.device)
            eps_pred = self.forward_diffusion(sp=xt, t=timestep.unsqueeze(-1), zsa=zsa)
            sigma_t = 0
            if t > 0:
                sigma_t_square = self.betas[timestep] * (1 - self.alphabars_prev[timestep]) / (1 - self.alphabars[timestep])
                sigma_t_square = sigma_t_square.clip(1e-20)
                sigma_t = sigma_t_square.sqrt()
            return 1. / self.alphas[timestep].sqrt() * (
                xt - self.betas[timestep] / (1 - self.alphabars[timestep]).sqrt() * eps_pred
            ) + sigma_t * z

        for t in reversed(range(0, self.sample_steps)):
            next_xt = reverse(xt, t)
            next_xt = torch.clip(next_xt, self.x_min, self.x_max)
            if preserve_history:
                info["sample_history"].append(next_xt.clone())
            xt = next_xt

        return xt, info
