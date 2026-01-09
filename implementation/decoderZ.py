from abc import ABC
import torch
import torch.nn.functional as F
from srcZ.Unet import UNet


# decoder_wrapper.py
import torch
import torch.nn as nn
from config import DecoderParams
from srcZ.Unet import UNet
class Decoder(nn.Module):
    """
    Decoder CMF
    """
    def __init__(self, in_channels, out_channel, decoder_params: DecoderParams):
        super().__init__()
       
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channel,
            channels=decoder_params.downsampling_upsampling_channels,
            dropout=decoder_params.dropout,
            n_blocks=decoder_params.n_transformer_per_block,
            num_mid_blocks=decoder_params.num_mid_blocks,
            attention_head_dim=decoder_params.attention_head_dim,
            num_heads=decoder_params.num_attention_heads,
        )

        self.sigma_min = 1e-4
        # Training settings
        self.sigma_min = 1e-4
    
    def compute_loss(self, x1, mask, mu, cond=None):
        """Training"""
        b, _, t = mu.shape
        t_rand = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)
        
        # Noisy input
        y = (1 - (1 - self.sigma_min) * t_rand) * z + t_rand * x1
        # Target
        u = x1 - (1 - self.sigma_min) * z
        
        # UNet prediction
        pred = self.unet(y, mu, mask, t_rand.squeeze())
        
        # Loss
        loss = torch.nn.functional.mse_loss(
            pred, u, reduction="sum"
        ) / (torch.sum(mask) * u.shape[1])
        
        return loss, y

    def forward(self, mu, mask, n_timesteps, temperature=1.0):
        """Inference """
        # Euler solver
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        
        # Run Euler steps
        x, t = z, t_span[0]
        dt = t_span[1] - t_span[0]
        
        for step in range(1, len(t_span)):
            dphi_dt = self.unet(x, mu, mask, t)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        
        return x



# quick test
if __name__ == "__main__":
   

    # Initialize trainer
    trainer = Decoder(
        in_channels=80,
        out_channel=80
        
    )

    # Dummy data
    x1 = torch.randn(4, 80, 160)  # Target features
    mu = torch.randn(4, 80, 160)   # Condition features
    mask = torch.ones(4, 1, 160)   # Valid frames mask

    # Compute loss
    loss, y = trainer.compute_loss(x1, mask, mu)
    print(f"Training loss: {loss.item()}")

    # Initialize inference model
    inference_model = Decoder(
        in_channels=80,
        out_channel=80,
      
    )

    # Generate samples
    generated = inference_model(mu, mask, n_timesteps=10)
    print(f"Generated shape: {generated.shape}") # we expect (4, 80, 160)