# QAGD_schedulers.py

import torch
import torch.nn as nn
from math import pi as PI

class QAGD_Scheduler(nn.Module):

    def __init__(self):
        super(QAGD_Scheduler, self).__init__()

    # Noise Schedules: Linear (Currently Unused)
    def linear_beta_schedule(timesteps: int, beta_start: float = 0.0, beta_end: float = 1.0):
        betas = torch.linspace(beta_start, beta_end, timesteps)
        return betas

    # Noise Schedules: Cosine
    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * PI / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
        return betas

    # Noise Schedules: Sigmoid (Currently Unused)
    @staticmethod
    def sigmoid_beta_schedule(timesteps: int, beta_start: float = 0.0, beta_end: float = 1.0):
        t = torch.arange(timesteps)
        betas = beta_end + (beta_start - beta_end) / (1 + torch.exp(-t))
        return betas

    # Noise Schedules: Exponential (Currently Unused)
    @staticmethod
    def exponential_beta_schedule(timesteps: int, beta_start: float = 0.0, beta_end: float = 1.0):
        t = torch.arange(timesteps)
        betas = beta_end + (beta_start - beta_end)^t
        return betas

    # Create Beta Noise Scheduler
    def create_beta_noise_scheduler(self, timesteps: int, beta_start: float = 0.0, beta_end: float = 1.0):
        betas = self.cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        return betas, alphas, alphas_cumprod
