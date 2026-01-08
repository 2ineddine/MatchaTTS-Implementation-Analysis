from abc import ABC
import torch
import torch.nn.functional as F
from Unet import UNet



class CFM(torch.nn.Module, ABC):
    def __init__(self, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.sigma_min = 1e-4
        self.estimator = None


class Decoder_Trainer(CFM):
    """Handles training: compute_loss for conditional flow matching"""
    def __init__(self, in_channels, out_channel):
        super().__init__(n_feats=in_channels)
        in_channels = in_channels 
        self.estimator = UNet(in_channels=in_channels, out_channels=out_channel)

    def compute_loss(self, x1, mask, mu):
        """Compute conditional flow matching loss"""
        b, _, t = mu.shape
        t_rand = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)

        # y is the noisy input to the estimator
        y = (1 - (1 - self.sigma_min) * t_rand) * z + t_rand * x1
        # u is the target
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(
            self.estimator(y, mu, mask, t_rand.squeeze()), u, reduction="sum"
        ) / (torch.sum(mask) * u.shape[1])

        return loss, y


class Decoder_inference(CFM):
    """Handles inference: Euler ODE solver for generation"""
    def __init__(self, in_channels, out_channel):
        super().__init__(n_feats=in_channels,)
        in_channels = in_channels 
        self.estimator = UNet(in_channels=in_channels, out_channels=out_channel)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0):
        """Run Euler solver to generate samples"""
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span, mu, mask)

    def solve_euler(self, x, t_span, mu, mask):
        """Fixed-step Euler ODE solver"""
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x,mu,mask,  t)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]


if __name__ == "__main__":
   

    # Initialize trainer
    trainer = Decoder_Trainer(
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
    inference_model = Decoder_inference(
        in_channels=80,
        out_channel=80,
      
    )

    # Generate samples
    generated = inference_model(mu, mask, n_timesteps=10)
    print(f"Generated shape: {generated.shape}") # we expect (4, 80, 160)