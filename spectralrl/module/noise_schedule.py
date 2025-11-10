import numpy as np
import torch


def linear_beta_schedule(beta_min: float = 1e-4, beta_max: float = 0.02, T: int = 1000):
    return np.linspace(beta_min, beta_max, T)

def cosine_beta_schedule(s: float = 0.008, T: int = 1000):
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2.) ** 2
    alpha_bar = f / f[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clip(None, 0.999)

def vp_beta_schedule(T: int = 1000):
    t = np.arange(1, T+1)
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas

def get_noise_schedule(
    noise_schedule: str="linear",
    num_noises: int=1000,
    beta_min: float=1e-4,
    beta_max: float=0.02,
    s: float=0.008
):
    if noise_schedule == "linear":
        betas = linear_beta_schedule(beta_min, beta_max, num_noises)
    elif noise_schedule == "cosine":
        betas = cosine_beta_schedule(s, num_noises)
    elif noise_schedule == "vp":
        betas = vp_beta_schedule(num_noises)
    else:
        raise NotImplementedError
    alphas = 1 - betas
    alphabars = np.cumprod(alphas, axis=0)
    return torch.as_tensor(betas, dtype=torch.float32), \
           torch.as_tensor(alphas, dtype=torch.float32), \
           torch.as_tensor(alphabars, dtype=torch.float32)
