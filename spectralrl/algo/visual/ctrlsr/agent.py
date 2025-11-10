import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.algo.visual.base import BaseVisualAlgorithm
from spectralrl.algo.visual.ctrlsr.network import Encoder, FactorizedNCE
from spectralrl.algo.visual.drqv2.network import RandomShiftsAug
from spectralrl.module.actor import SquashedDeterministicActor
from spectralrl.module.critic import RffDoubleQ
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target


class CtrlSR_TD3(BaseVisualAlgorithm):
    def __init__(
        self,
        obs_dim,
        action_dim,
        cfg,
        device
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ctrl_coef = cfg.ctrl_coef
        self.reward_coef = cfg.reward_coef
        self.critic_coef = cfg.critic_coef
        self.reg_coef = cfg.reg_coef
        self.tau = cfg.tau
        self.encoder_tau = cfg.encoder_tau
        self.extra_repr_step = cfg.extra_repr_step
        self.update_every = cfg.update_every
        self.back_critic_grad = cfg.back_critic_grad
        self.target_policy_noise = cfg.target_policy_noise
        self.noise_clip = cfg.noise_clip
        self.exploration_noise = cfg.exploration_noise

        # networks
        self.encoder = Encoder(
            obs_dim=obs_dim,
        ).to(self.device)
        self.ctrl = FactorizedNCE(
            batch_size=cfg.batch_size,
            obs_dim=self.encoder.output_dim,
            action_dim=action_dim,
            feature_dim=cfg.feature_dim,
            embed_dim=cfg.embed_dim,
            phi_hidden_dims=cfg.phi_hidden_dims,
            mu_hidden_dims=cfg.mu_hidden_dims,
            reward_hidden_dims=cfg.reward_hidden_dims,
            rff_dim=cfg.rff_dim,
            num_noises=cfg.num_noises,
            device=self.device
        ).to(self.device)

        # td3 part
        self.actor = SquashedDeterministicActor(
            input_dim=self.encoder.output_dim,
            output_dim=self.action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            activation=nn.ELU,
            norm_layer=nn.LayerNorm,
        ).to(self.device)
        self.critic = RffDoubleQ(
            feature_dim=cfg.feature_dim,
            hidden_dims=cfg.critic_hidden_dims,
            linear=False,
            rff_dim=cfg.rff_dim,
        ).to(self.device)

        self.encoder_target = make_target(self.encoder)
        self.actor_target = make_target(self.actor)
        self.critic_target = make_target(self.critic)
        self.aug = RandomShiftsAug(pad=4)

        # optimizers
        self.optim = {
            "encoder": torch.optim.AdamW(self.encoder.parameters(), lr=cfg.encoder_lr, weight_decay=cfg.wd),
            "ctrl": torch.optim.AdamW(self.ctrl.network.parameters(), lr=cfg.ctrl_lr, weight_decay=cfg.wd),
            "actor": torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.wd),
            "critic": torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.wd),
        }
        self._step = 1
        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.ctrl.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @torch.no_grad()
    def select_action(self, obs, step, deterministic=False):
        obs = torch.as_tensor(obs, device=self.device)[None, ...]
        obs = self.encoder(obs)
        obs = obs.reshape(1, -1)
        action, *_ = self.actor.sample(obs, deterministic=deterministic)
        if not deterministic:
            action = action + self.exploration_noise * torch.randn_like(action)
            action = action.clip(-1.0, 1.0)
        return action.squeeze(0).detach().cpu().numpy()

    def train_step(self, buffer, batch_size, step):
        self._step += 1
        if self._step % self.update_every != 0:
            return {}
        tot_metrics = {}
        for _ in range(self.extra_repr_step):
            batch = buffer.sample(batch_size)
            img, action, reward, discount, next_img, next_img_step = [
                convert_to_tensor(t, self.device) for t in batch
            ]
            img = self.aug(img.float()).detach()
            next_img = self.aug(next_img.float()).detach()

            ctrl_metrics, ctrl_loss, reward_loss, latent, next_latent = self.ctrl_step(img, action, next_img, reward)
            critic_metrics, critic_loss = self.critic_step(img, action, next_img, reward, discount, step, latent, next_latent)

            loss = (
                ctrl_loss * self.ctrl_coef + \
                reward_loss * self.reward_coef + \
                critic_loss * self.critic_coef
            )

            for key in ["encoder", "ctrl", "critic"]:
                self.optim[key].zero_grad(set_to_none=True)
            loss.backward()
            for key in ["encoder", "ctrl", "critic"]:
                self.optim[key].step()

        actor_metrics, actor_loss, reg_loss = self.actor_step(latent.detach(), step)
        self.optim["actor"].zero_grad()
        (
            actor_loss * 1.0 + \
            reg_loss * self.reg_coef
        ).backward()
        self.optim["actor"].step()

        self.update_target()

        tot_metrics.update(ctrl_metrics)
        tot_metrics.update(critic_metrics)
        tot_metrics.update(actor_metrics)
        return tot_metrics

    def ctrl_step(self, img, action, next_img, reward=None):
        latent = self.encoder(img)
        next_latent = self.encoder_target(next_img).detach()
        ctrl_loss, reward_loss, metrics = self.ctrl.compute_losses(
            s=latent,
            a=action,
            sp=next_latent.detach(),
            r=reward
        )
        return metrics, ctrl_loss, reward_loss, latent, next_latent

    def critic_step(self, img, action, next_img, reward, discount, step, latent, next_latent):
        feature = self.ctrl.compute_feature(s=latent, a=action)
        if not self.back_critic_grad:
            feature = feature.detach()

        with torch.no_grad():
            next_latent = self.encoder(next_img)
            noise = (torch.randn_like(action) * self.target_policy_noise).clip(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target.sample(next_latent)[0] + noise).clip(-1.0, 1.0)
            next_feature = self.ctrl.compute_feature(s=next_latent, a=next_action)
            q_target = self.critic_target(next_feature)
            q_target = reward + discount * q_target.min(0)[0]
        q_pred = self.critic(feature)
        critic_loss = F.mse_loss(q_pred, q_target.unsqueeze(0).repeat(2, 1, 1))

        return {
            "loss/critic_loss": critic_loss.item(),
            "info/q_pred": q_pred.mean().item(),
            "info/q_target": q_target.mean().item(),
            "info/reward": reward.mean().item()
        }, critic_loss

    def actor_step(self, latent, step):
        new_action, _, info = self.actor.sample(latent)
        new_feature = self.ctrl.compute_feature(s=latent, a=new_action)
        actor_loss = -self.critic(new_feature).mean()
        reg_loss = info["pre_value"].pow(2).mean()

        return {
            "loss/actor_loss": actor_loss.item(),
            "loss/reg_loss": reg_loss.item(),
        }, actor_loss, reg_loss

    def update_target(self):
        sync_target(self.critic, self.critic_target, self.tau)
        sync_target(self.actor, self.actor_target, self.tau)
        sync_target(self.encoder, self.encoder_target, self.encoder_tau)
