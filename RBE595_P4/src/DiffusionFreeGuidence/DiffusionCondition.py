import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Function for cosine scheduling
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((t / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', cosine_beta_schedule(T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # For diffusion, we need to store the alphas, alpha bars, and betas
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    # define forward pass. It consists of generating Xt from X0
    def forward(self, x_0, labels):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = T
        self.w = w

        # we have put cosine schedule. can revert back to linear also if needed
        self.register_buffer('betas', cosine_beta_schedule(T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    
    # function to get previous mean
    def estimate_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    # For this diffusion model, I am getting much better results with mean and variance
    def get_noise_mean_variance(self, x_t, t, labels):
        variance = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        variance = extract(variance, t, x_t.shape)
        eps_ = self.model(x_t, t, labels)
        nonEps_ = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps_ = (1. + self.w) * eps_ - self.w * nonEps_
        xt_prev_mean = self.estimate_prev_mean_from_eps(x_t, t, eps=eps_)
        return xt_prev_mean, variance

    def forward(self, x_T, labels):
        x_t = x_T
        samplingSteps = self.T # we reduced this and checked with lesser steps also
        for time_step in reversed(range(samplingSteps)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.get_noise_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)  