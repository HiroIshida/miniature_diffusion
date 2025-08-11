import math

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor


class DiffusionSinusoidalPosEmb(nn.Module):
    # fetched from LeRobot (Apache 2.0 License)
    # https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/modeling_diffusion.py

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be divisible by 2"
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SimpleMLP(nn.Module):
    def __init__(self, T, hidden_size=128, pos_emb_dim=64):
        super(SimpleMLP, self).__init__()
        input_size = 2 * T + 2 + pos_emb_dim  # (2 × T + 2 + 1)
        output_size = 2 * T  # (2 × T)
        self.pos_emb = DiffusionSinusoidalPosEmb(pos_emb_dim)
        self.pos_emb_dim = pos_emb_dim
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.input_size = input_size

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type="epsilon",
        )

    def compute_loss(self, features_batch, context_batch):
        b_size, f_size = features_batch.shape
        _, c_size = context_batch.shape
        assert f_size + c_size + self.pos_emb_dim == self.input_size
        assert torch.all((features_batch >= -1) & (features_batch <= 1)), (
            "features_batch must be scaled to the range [-1, 1]"
        )
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(b_size,),
            device=features_batch.device,
            dtype=torch.long,
        )
        eps = torch.randn(features_batch.shape, device=features_batch.device)
        noised = self.noise_scheduler.add_noise(features_batch, eps, timesteps)
        timesteps_norm = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
        embed_timesteps = self.pos_emb(timesteps_norm)
        concated = torch.cat((noised, context_batch, embed_timesteps), dim=1)
        pred_noise = self.model(concated)
        loss = nn.functional.mse_loss(pred_noise, eps, reduction="mean")
        return loss

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    T = 50
    model = SimpleMLP(T)
    batch = torch.rand(2, 2 * T) * 2 - 1  # Batch size of 2
    contexts = torch.rand(2, 2)
    loss = model.compute_loss(batch, contexts)  # Example context vector
    print(loss)
