import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from spectralrl.algo.visual.base import BaseVisualAlgorithm
from spectralrl.algo.visual.diffsr.ae import Decoder, Encoder
from spectralrl.algo.visual.diffsr.ddpm import Diffusion
from spectralrl.algo.visual.drqv2.network import RandomShiftsAug
from spectralrl.module.actor import SquashedDeterministicActor
from spectralrl.module.critic import RffDoubleQ
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target


class DiffSR_TD3(BaseVisualAlgorithm):
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
        self.var_coef = cfg.var_coef
        self.recon_coef = cfg.recon_coef
        self.diffusion_coef = cfg.diffusion_coef
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
        self.encoder = Encoder(cfg.latent_dim, cfg.ae_num_layers, cfg.ae_num_filters).to(self.device)
        self.decoder = Decoder(cfg.latent_dim, cfg.ae_num_layers, cfg.ae_num_filters).to(self.device)
        self.diffusion = Diffusion(
            obs_dim=cfg.latent_dim*3,
            action_dim=self.action_dim,
            cfg=cfg,
            device=device
        ).to(self.device)
        self.actor = SquashedDeterministicActor(
            input_dim=cfg.latent_dim*3,
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
            "ae": torch.optim.Adam([*self.encoder.parameters(), *self.decoder.parameters()], lr=cfg.ae_lr, weight_decay=cfg.wd),
            "diffusion": torch.optim.AdamW(self.diffusion.parameters(), lr=cfg.diffusion_lr, weight_decay=cfg.wd),
            "actor": torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.wd),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.wd),
        }
        self._step = 1
        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.decoder.train(training)
        self.diffusion.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @torch.no_grad()
    def evaluate(self, buffer):
        self.train(False)
        try:
            batch = buffer.sample(3)
        except Exception as e:
            return {}, None
        img, action, reward, discount, next_img, next_img_step = [
            convert_to_tensor(t, self.device) for t in batch
        ]
        img = img.float()
        next_img = next_img.float()

        encoder, decoder, diffusion = self.encoder, self.decoder, self.diffusion

        B, S, W, H = img.shape
        img = img.view(B*3, S//3, W, H)
        next_img = next_img.view(B*3, S//3, W, H)

        latent = encoder(img).reshape(B, -1)
        next_latent = encoder(next_img).reshape(B, -1)

        raw_next_img = next_img / 255. - 0.5
        recon_next_img = decoder(next_latent.reshape(B*3, -1))

        with torch.no_grad():
            _, info = diffusion.sample(next_latent.shape, latent, action, preserve_history=True)
            history = info["sample_history"]

        stepsize = diffusion.sample_steps // 5
        history = list(reversed(history))
        checkpoints = list(range(0, diffusion.sample_steps, stepsize)) + [diffusion.sample_steps]
        checkpoints = list(reversed(checkpoints))
        checkpoint_next_latents = [history[c] for c in checkpoints]
        checkpoint_gen_next_imgs = [decoder(xt.reshape(B*3, -1)) for xt in checkpoint_next_latents]

        checkpoint_next_latents = torch.stack(checkpoint_next_latents, dim=0)
        img_to_show = torch.stack([raw_next_img, recon_next_img]+checkpoint_gen_next_imgs, dim=0)
        latent_l1_diff = (checkpoint_next_latents - next_latent.repeat(6, 1, 1)).abs().mean([1,2])
        N = img_to_show.shape[0]
        img_to_show = img_to_show.reshape(N*9, 3, img_to_show.shape[-2], img_to_show.shape[-1])
        img_to_show = img_to_show + 0.5
        grid = torchvision.utils.make_grid(img_to_show, nrow=9)

        metrics = {
            f"l1diff_step{checkpoints[i]}": latent_l1_diff[i].item() for i in range(len(checkpoints))
        }
        self.train(True)
        torch.cuda.empty_cache()
        return metrics, grid

    @torch.no_grad()
    def select_action(self, obs, step, deterministic=False):
        obs = torch.as_tensor(obs, device=self.device)[None, ...]
        obs = obs.view(3, 3, obs.shape[-1], obs.shape[-1])
        obs = self.encoder(obs).reshape(1, -1)
        action, *_ = self.actor.sample(obs, deterministic=deterministic)
        if not deterministic:
            action = action + self.exploration_noise * torch.randn_like(action)
            action = action.clip(-1.0, 1.0)
        return action.squeeze().cpu().numpy()

    def pretrain_step(self, buffer, batch_size, step):
        batch = buffer.sample(batch_size)
        img, action, reward, discount, next_img, next_img_step = [
            convert_to_tensor(t, self.device) for t in batch
        ]
        img = self.aug(img.float()).detach()
        next_img_step = next_img_step[:, -3:].float().detach()
        tot_metrics = {}
        ae_metrics, recon_loss, var_loss = self.ae_step(img, next_img_step)
        tot_metrics.update(ae_metrics)
        self.optim["ae"].zero_grad(set_to_none=True)
        (recon_loss*self.recon_coef + var_loss*self.var_coef).backward()
        self.optim["ae"].step()
        sync_target(self.encoder, self.encoder_target, 1.0)
        return tot_metrics

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
            img = img.float()
            next_img = next_img.float()
            next_img_step = next_img_step[:, -3:].float().detach()

            ae_metrics, recon_loss, var_loss = self.ae_step(img, next_img_step)
            diffusion_metrics, diffusion_loss, reward_loss, latent, next_latent = self.diffusion_step(img, action, next_img, reward)
            critic_metrics, critic_loss = self.critic_step(latent, action, next_latent, reward, discount, step)

            loss = (
                recon_loss * self.recon_coef + \
                var_loss * self.var_coef + \
                diffusion_loss * self.diffusion_coef + \
                reward_loss * self.reward_coef + \
                critic_loss * self.critic_coef
            )

            self.optim["ae"].zero_grad(set_to_none=True)
            self.optim["diffusion"].zero_grad(set_to_none=True)
            self.optim["critic"].zero_grad(set_to_none=True)
            loss.backward()
            self.optim["ae"].step()
            self.optim["diffusion"].step()
            self.optim["critic"].step()

        actor_metrics, actor_loss, reg_loss = self.actor_step(latent.detach(), step)
        self.optim["actor"].zero_grad(set_to_none=True)
        (
            actor_loss * 1.0 + \
            reg_loss * self.reg_coef
        ).backward()
        self.optim["actor"].step()

        self.update_target()
        tot_metrics.update(ae_metrics)
        tot_metrics.update(diffusion_metrics)
        tot_metrics.update(actor_metrics)
        tot_metrics.update(critic_metrics)
        return tot_metrics

    def ae_step(self, img, next_img_step):
        img_target = next_img_step / 255. - 0.5
        aug_img_input = self.aug(next_img_step)
        latent = self.encoder(aug_img_input)
        img_pred = self.decoder(latent)

        recon_loss = F.mse_loss(img_pred, img_target, reduction="sum") / img_pred.shape[0]
        var_loss = (1.0 - latent.std(0)).clip(0.0).sum()

        return {
            "loss/recon_loss": recon_loss.item(),
            "loss/var_loss": var_loss.item(),
            "info/latent_mean": latent.mean().item(),
            "info/latent_std": latent.std(0).mean().item(),
            "info/latent_l1_norm": latent.abs().mean().item(),
        }, recon_loss, var_loss

    def diffusion_step(self, img, action, next_img, reward):
        B, S, W, H = img.shape
        with torch.no_grad():
            img = img.view(B*3, S//3, W, H)
            next_img = next_img.view(B*3, S//3, W, H)
            latent = self.encoder(img)
            next_latent = self.encoder_target(next_img)
            latent = latent.reshape(B, -1)
            next_latent = next_latent.reshape(B, -1)
        diffusion_loss, reward_loss, metrics = self.diffusion.compute_losses(latent, action, next_latent, reward)

        next_latent = self.encoder(next_img).reshape(B, -1).detach()
        return metrics, diffusion_loss, reward_loss, latent, next_latent

    def critic_step(self, latent, action, next_latent, reward, discount, step):
        feature = self.diffusion.compute_feature(s=latent, a=action).detach()
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.target_policy_noise).clip(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target.sample(next_latent)[0] + noise).clip(-1.0, 1.0)
            next_feature = self.diffusion.compute_feature(s=next_latent, a=next_action)

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
        new_feature = self.diffusion.compute_feature(s=latent, a=new_action)
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
