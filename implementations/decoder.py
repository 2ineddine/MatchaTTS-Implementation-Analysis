"""
Matcha-TTS Decoder with Conditional Flow Matching
Generates mel spectrograms using flow matching
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

# ============================================================================
# DECODER CONFIGURATION - Edit these parameters
# ============================================================================
IN_CHANNELS = 80           # Input mel channels (from encoder)
OUT_CHANNELS = 80          # Output mel channels
CHANNELS = [256, 256]      # Hidden dimensions for each down/up block
DROPOUT = 0.05             # Dropout probability
ATTENTION_HEAD_DIM = 64    # Dimension per attention head
N_BLOCKS = 1               # Number of transformer blocks per stage
NUM_MID_BLOCKS = 2         # Number of middle blocks
NUM_HEADS = 2              # Number of attention heads

# Flow Matching
SIGMA_MIN = 1e-4           # Minimum noise level
N_TIMESTEPS = 10           # Number of ODE steps for inference

# ============================================================================


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """Projects timestep embeddings"""
    def __init__(self, in_channels, time_embed_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Block1D(nn.Module):
    """Basic 1D convolutional block with GroupNorm"""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(nn.Module):
    """Residual block with timestep conditioning"""
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Downsample1D(nn.Module):
    """Downsamples by 2x"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsamples by 2x using transposed convolution"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class FeedForward(nn.Module):
    """Simple FFN for transformer"""
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim, num_heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]
            attn = attn.masked_fill(~mask.bool(), -1e9)

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Transformer block with attention + FFN"""
    def __init__(self, dim, num_heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, num_heads, head_dim, dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.attn(self.norm1(x), mask) + x
        x = self.ff(self.norm2(x)) + x
        return x


class Decoder(nn.Module):
    """
    UNet-style decoder for flow matching

    Architecture:
    1. Time embedding
    2. Down blocks: ResNet + Transformer + Downsample
    3. Mid blocks: ResNet + Transformer
    4. Up blocks: ResNet + Transformer + Upsample (with skip connections)
    5. Final projection
    """
    def __init__(
        self,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels=CHANNELS,
        dropout=DROPOUT,
        attention_head_dim=ATTENTION_HEAD_DIM,
        n_blocks=N_BLOCKS,
        num_mid_blocks=NUM_MID_BLOCKS,
        num_heads=NUM_HEADS,
        n_spks=1,
        spk_emb_dim=64,
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_spks = n_spks

        # Adjust input channels for speaker conditioning
        if n_spks > 1:
            in_channels = in_channels + spk_emb_dim

        # Time embeddings
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(in_channels, time_embed_dim)

        # Down blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = in_channels
        for i, ch in enumerate(channels):
            input_channel = output_channel
            output_channel = ch
            is_last = i == len(channels) - 1

            resnet = ResnetBlock1D(input_channel, output_channel, time_embed_dim)
            transformer_blocks = nn.ModuleList([
                TransformerBlock(output_channel, num_heads, attention_head_dim, dropout)
                for _ in range(n_blocks)
            ])
            downsample = Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)

            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        # Mid blocks
        self.mid_blocks = nn.ModuleList([])
        for _ in range(num_mid_blocks):
            resnet = ResnetBlock1D(output_channel, output_channel, time_embed_dim)
            transformer_blocks = nn.ModuleList([
                TransformerBlock(output_channel, num_heads, attention_head_dim, dropout)
                for _ in range(n_blocks)
            ])
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        # Up blocks
        self.up_blocks = nn.ModuleList([])
        channels_reversed = list(channels[::-1]) + [channels[0]]
        for i in range(len(channels_reversed) - 1):
            input_channel = channels_reversed[i]
            output_channel = channels_reversed[i + 1]
            is_last = i == len(channels_reversed) - 2

            resnet = ResnetBlock1D(2 * input_channel, output_channel, time_embed_dim)  # 2x for skip connections
            transformer_blocks = nn.ModuleList([
                TransformerBlock(output_channel, num_heads, attention_head_dim, dropout)
                for _ in range(n_blocks)
            ])
            upsample = Upsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        # Final layers
        self.final_block = Block1D(channels_reversed[-1], channels_reversed[-1])
        self.final_proj = nn.Conv1d(channels_reversed[-1], self.out_channels, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None):
        """
        Args:
            x: Noisy input [batch, in_channels, time]
            mask: Mask [batch, 1, time]
            mu: Encoder output (conditioning) [batch, in_channels, time]
            t: Timestep [batch]
            spks: Speaker embeddings [batch, spk_emb_dim] (optional)

        Returns:
            Predicted velocity [batch, out_channels, time]
        """
        # Embed timestep
        t_emb = self.time_embeddings(t)
        t_emb = self.time_mlp(t_emb)

        # Concatenate input with encoder output (mu)
        x = pack([x, mu], "b * t")[0]

        # Add speaker conditioning
        if spks is not None and self.n_spks > 1:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]

        # Down blocks with skip connections
        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t_emb)

            # Apply transformer blocks
            x = rearrange(x, "b c t -> b t c")
            mask_down_flat = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(x, mask_down_flat)
            x = rearrange(x, "b t c -> b c t")

            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        # Mid blocks
        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t_emb)

            x = rearrange(x, "b c t -> b t c")
            mask_mid_flat = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(x, mask_mid_flat)
            x = rearrange(x, "b t c -> b c t")

        # Up blocks with skip connections
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()

            # Match sizes if there's a mismatch (due to odd-sized downsampling)
            if x.size(2) != skip.size(2):
                min_size = min(x.size(2), skip.size(2))
                x = x[:, :, :min_size]
                skip = skip[:, :, :min_size]
                mask_up = mask_up[:, :, :min_size]

            x = resnet(pack([x, skip], "b * t")[0], mask_up, t_emb)

            x = rearrange(x, "b c t -> b t c")
            mask_up_flat = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(x, mask_up_flat)
            x = rearrange(x, "b t c -> b c t")

            x = upsample(x * mask_up)

        # Final projection
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)

        return output * mask


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching (CFM) for mel spectrogram generation

    Uses Optimal Transport path: interpolates between noise and target
    """
    def __init__(
        self,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels=CHANNELS,
        dropout=DROPOUT,
        attention_head_dim=ATTENTION_HEAD_DIM,
        n_blocks=N_BLOCKS,
        num_mid_blocks=NUM_MID_BLOCKS,
        num_heads=NUM_HEADS,
        sigma_min=SIGMA_MIN,
        n_spks=1,
        spk_emb_dim=64,
    ):
        super().__init__()
        self.sigma_min = sigma_min

        # The decoder estimates the velocity field
        # Input is 2x mel channels (noisy mel + encoder output)
        decoder_in_channels = in_channels * 2
        self.decoder = Decoder(
            in_channels=decoder_in_channels,
            out_channels=out_channels,
            channels=channels,
            dropout=dropout,
            attention_head_dim=attention_head_dim,
            n_blocks=n_blocks,
            num_mid_blocks=num_mid_blocks,
            num_heads=num_heads,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

    def forward(self, mu, mask, n_timesteps=N_TIMESTEPS, temperature=1.0, spks=None):
        """
        Generate mel spectrogram using flow matching

        Args:
            mu: Encoder output [batch, channels, time]
            mask: Mask [batch, 1, time]
            n_timesteps: Number of ODE solver steps
            temperature: Noise temperature
            spks: Speaker embeddings [batch, spk_emb_dim]

        Returns:
            Generated mel spectrogram [batch, channels, time]
        """
        # Sample initial noise
        z = torch.randn_like(mu) * temperature

        # Create timestep span [0, 1]
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)

        # Solve ODE using Euler method
        return self.solve_euler(z, t_span, mu, mask, spks)

    def solve_euler(self, x, t_span, mu, mask, spks):
        """Euler ODE solver"""
        t = t_span[0]
        dt = t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            # Estimate velocity
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            dphi_dt = self.decoder(x, mask, mu, t_batch, spks)

            # Update using Euler step
            x = x + dt * dphi_dt
            t = t + dt

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x

    def compute_loss(self, x1, mask, mu, spks=None):
        """
        Compute conditional flow matching loss

        Args:
            x1: Target mel spectrogram [batch, channels, time]
            mask: Mask [batch, 1, time]
            mu: Encoder output [batch, channels, time]
            spks: Speaker embeddings [batch, spk_emb_dim]

        Returns:
            loss: CFM loss
        """
        batch_size = x1.shape[0]

        # Sample random timestep for each batch element
        t = torch.rand(batch_size, device=x1.device, dtype=x1.dtype)

        # Sample noise
        z = torch.randn_like(x1)

        # Optimal transport conditional path
        t_expanded = t[:, None, None]
        y_t = (1 - (1 - self.sigma_min) * t_expanded) * z + t_expanded * x1

        # Target velocity
        u = x1 - (1 - self.sigma_min) * z

        # Predict velocity
        predicted_u = self.decoder(y_t, mask, mu, t, spks)

        # MSE loss
        loss = F.mse_loss(predicted_u * mask, u * mask, reduction="sum") / (torch.sum(mask) * x1.shape[1])

        return loss


if __name__ == "__main__":
    # Test the decoder
    batch_size = 2
    mel_len = 100

    cfm = ConditionalFlowMatching()

    # Create dummy input
    mu = torch.randn(batch_size, IN_CHANNELS, mel_len)
    mask = torch.ones(batch_size, 1, mel_len)
    x1 = torch.randn(batch_size, OUT_CHANNELS, mel_len)

    # Test inference
    output = cfm(mu, mask, n_timesteps=10)
    print(f"Encoder output (mu) shape: {mu.shape}")
    print(f"Generated mel shape: {output.shape}")

    # Test training loss
    loss = cfm.compute_loss(x1, mask, mu)
    print(f"CFM Loss: {loss.item():.4f}")
