# Conditional Flow Matching (CFM)

__author__ = "Massyl A."



import torch
import torch.nn as nn
from Unet import UNet


class CFM(nn.Module):

    """
    Conditional Flow Matching for mel-spectrogram generation.
    Wraps the U-Net decoder and implements : CFM for training and ODE-based sampling for Inference.
    """

    def __init__(self, in_channels, out_channel, cfm_params, decoder_params):
        super().__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel

        # Store CFM parameters
        self.sigma_min = cfm_params.sigma_min

        # Initialize decoder (U-Net)
        self.decoder = UNet(
            in_channels=in_channels,
            out_channels=out_channel,
            downsampling_upsampling_channels=decoder_params.downsampling_upsampling_channels,
            dropout=decoder_params.dropout,
            attention_head_dim=decoder_params.attention_head_dim,
            n_transformer_per_block=decoder_params.n_transformer_per_block,
            num_mid_blocks=decoder_params.num_mid_blocks,
            num_attention_heads=decoder_params.num_attention_heads
        )


    # 1. Inference mode : Infere X1 using Euler's ODE solver
    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0):
        """
        Generate mel-spectrogram using ODE solver (Euler method).

        Args:
            mu: Encoder output / condition (batch, n_feats, time)
            mask: Binary mask (batch, 1, time)
            n_timesteps: Number of ODE integration steps
            temperature: Sampling temperature (controls randomness)

        Returns:
            Generated mel-spectrogram (batch, n_feats, time)
        """
        batch, channels, length = mu.shape
        device = mu.device
        dtype = mu.dtype

        # Start from random noise
        x = torch.randn(batch, channels, length, device=device, dtype=dtype) * temperature
        x = x * mask

        # Euler ODE solver
        dt = 1.0 / n_timesteps

        for step in range(n_timesteps):
            # create tensor t (1d vector)
            t = torch.full((batch,), step / n_timesteps, device=device, dtype=dtype)

            # Predict velocity field
            velocity = self.decoder(x, mask, mu, t)

            # Euler step: x_{t+dt} = x_t + velocity * dt
            x = x + velocity * dt
            x = x * mask

        return x


    # 2. Training mode : compute flow matching loss at a random time step
    def compute_loss(self, x1, mask, mu, cond=None):
        batch, _, _ = mu.shape
        
        # 1. Sample t and x0
        t = torch.rand(batch, device=x1.device, dtype=x1.dtype)
        x0 = torch.randn_like(x1)
        
        # 2. Add correct broadcasting for math
        t_expanded = t.view(batch, 1, 1)
        
        # 3. Interpolate between x0 and x1: x_t = t * x1 + (1 - t) * x0     (Apply sigma_min for stability)
        x_t = (1 - (1 - self.sigma_min) * t_expanded) * x0 + t_expanded * x1
        
        # 4. Calculate True Velocity (Target)
        true_velocity = x1 - (1 - self.sigma_min) * x0
        
        # 5. Predict (Pass 1D 't' to decoder, not t_expanded)
        pred_velocity = self.decoder(x_t, mask, mu, t)
        
        # 6. Loss
        loss = torch.sum((pred_velocity - true_velocity) ** 2 * mask)
        loss = loss / (torch.sum(mask) * x1.shape[1])
        
        return loss, {}