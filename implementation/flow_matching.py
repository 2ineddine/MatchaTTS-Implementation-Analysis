# Conditional Flow Matching (CFM)



import torch
import torch.nn as nn
from .decoder import Decoder


class CFM(nn.Module):
    """
    Conditional Flow Matching for mel-spectrogram generation.
    Wraps the U-Net decoder and implements ODE-based sampling.
    """

    def __init__(self, in_channels, out_channel, cfm_params, decoder_params):
        super().__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel

        # Store CFM parameters
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min

        # Initialize decoder (U-Net)
        self.decoder = Decoder(
            in_channels=in_channels,
            out_channels=out_channel,
            channels=decoder_params.channels,
            dropout=decoder_params.dropout,
            attention_head_dim=decoder_params.attention_head_dim,
            n_blocks=decoder_params.n_blocks,
            num_mid_blocks=decoder_params.num_mid_blocks,
            num_heads=decoder_params.num_heads
        )

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
            t = torch.full((batch,), step / n_timesteps, device=device, dtype=dtype)

            # Predict velocity field
            velocity = self.decoder(x, mask, mu, t)

            # Euler step: x_{t+dt} = x_t + velocity * dt
            x = x + velocity * dt
            x = x * mask

        return x

    def compute_loss(self, x1, mask, mu, cond=None):
        """
        Compute flow matching loss.

        Args:
            x1: Target mel-spectrogram (batch, n_feats, time)
            mask: Binary mask (batch, 1, time)
            mu: Encoder output (batch, n_feats, time)
            cond: Optional additional conditioning

        Returns:
            loss: Scalar loss value
            stats: Dictionary with loss statistics (empty for now)
        """
        batch = x1.shape[0]
        device = x1.device
        dtype = x1.dtype

        # Sample random timestep t ~ U(0, 1)
        t = torch.rand(batch, device=device, dtype=dtype)

        # Sample noise (x0) - standard Gaussian noise
        x0 = torch.randn_like(x1)
        x0 = x0 * mask

        # Interpolate between x0 and x1: x_t = t * x1 + (1 - t) * x0
        t_expanded = t.view(batch, 1, 1)  # Shape: (batch, 1, 1)
        x_t = t_expanded * x1 + (1 - t_expanded) * x0

        # True velocity: dx/dt = x1 - x0
        true_velocity = x1 - x0

        # Predict velocity using decoder
        pred_velocity = self.decoder(x_t, mask, mu, t)

        # MSE loss between predicted and true velocity
        loss = torch.sum((pred_velocity - true_velocity) ** 2 * mask)
        loss = loss / (torch.sum(mask) * x1.shape[1])  # Normalize by num valid elements

        return loss, {}
