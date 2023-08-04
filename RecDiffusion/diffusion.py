import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .utils import tensor_to_dict


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps: int):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape):
    if isinstance(t, int):
        return a[t]
    elif isinstance(t, list):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    else:
        raise TypeError("t must be int or a list of int")


class diffusion_sample(object):
    def __init__(
        self,
        time_step: int,
        scheduler: str = "linear_beta_schedule",
    ) -> None:
        self.time_step = time_step
        betas = eval(scheduler + f"({time_step})")
        self.betas = betas
        # define alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model: torch.nn.Module, data: dict, t: int) -> torch.tensor:
        device = next(model.parameters()).device
        x = data["pos"]
        t_tensor = torch.Tensor([t], device=device)
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(data, t_tensor) / sqrt_one_minus_alphas_cumprod_t
        )

        if t == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, model, data: dict):
        device = next(model.parameters()).device
        input_pose = data["pos"]

        # start from pure noise (for each example in the batch)
        poses = torch.zeros(self.time_step + 1, *input_pose.shape).to(device)
        poses[0] = input_pose

        for i in tqdm(
            reversed(range(0, self.time_step)),
            desc="sampling loop time step",
            total=self.time_step,
        ):
            data["pos"] = poses[i]
            poses[i + 1] = self.p_sample(model, data, i)
        return poses

    @torch.no_grad()
    def sample_random(self, model, n_atoms: int):
        device = next(model.parameters()).device
        input_pose = torch.randn((n_atoms, 3), device=device)
        data = tensor_to_dict(input_pose)
        return self.p_sample_loop(model, data)

    def sample_pose(self, model, data):
        assert data["pos"].shape[-1] == 3
        return self.p_sample_loop(model, data)
