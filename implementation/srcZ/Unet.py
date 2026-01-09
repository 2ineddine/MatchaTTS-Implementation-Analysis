import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .AttentionBlock import Attention
from .activation_function import SnakeBeta

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
    """MLP for processing timestep embeddings"""
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
    
    # basic bloc,  take an input 
    def __init__(self, dim, dim_out, groups=8,dropout = 0.0):
        super().__init__()
        
        
        # 1s convolutional layer 
        self.conv = nn.Conv1d(dim, dim_out, 3, padding=1)
        
        
        # normalization for example  (B,C,f), it divids C into groups and normalize each group (sum (group1))/number of sample  
        self.norm = nn.GroupNorm(groups, dim_out)
        
        # activation funciton  :  Mish:    f(x) = x * tanh(ln(1 + e^x))
        self.act = nn.Mish()
        
        
    
    def forward(self, x, mask):
        x = self.conv(x * mask)
        x = self.norm(x)
        x = self.act(x)
        return x * mask


class ResnetBlock1D(nn.Module):
    """ResNet block """
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)
    
    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h = h + self.time_mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class Downsample1D(nn.Module):
    """Downsampling by factor of 2"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsampling by factor of 2 using transpose convolution"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Transformer(nn.Module):
    """Transformer block """
    def __init__(self, dim, num_heads, dim_head, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            SnakeBeta(in_features=dim, out_features=dim*4),  # activation in expanded space
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention with pre-norm and residual
        x = x + self.attn(self.norm1(x), attention_mask=mask)
        # MLP with pre-norm and residual
        x = x + self.mlp(self.norm2(x))
        return x


class UNet(nn.Module):
    """
    1D UNet with transformer blocks for diffusion models
    Takes x, mu, and t as inputs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=2,
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        
        
        
        
        # Time embedding
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(in_channels, time_embed_dim)
        
        
        
        
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        
        
        
        
        
        
        # ========== DOWN BLOCKS ==========
        # Input will be concatenated [x, mu], so double the input channels
        output_channel = in_channels * 2   # input channels size is [B, C, T] , mu has the same size as x so ouptut_channel = [B, 2*C, T]
        for i in range(len(channels)):   # are the channel progession for example (2C, 256, 512)
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            
            # ResNet block
            resnet = ResnetBlock1D(input_channel, output_channel, time_embed_dim)
            
            # Transformer blocks
            transformers = nn.ModuleList([
                Transformer(output_channel, num_heads, attention_head_dim, dropout)
                for _ in range(n_blocks)  # for our case is only one (is the number of transformer block in each level)
            ])
            
            # Downsample or keep resolution
            downsample = (
                Downsample1D(output_channel) if not is_last 
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            
            self.down_blocks.append(nn.ModuleList([resnet, transformers, downsample]))
        
        
        
        
        # ========== MID BLOCKS ==========
        for _ in range(num_mid_blocks):
            resnet = ResnetBlock1D(channels[-1], channels[-1], time_embed_dim)
            transformers = nn.ModuleList([
                Transformer(channels[-1], num_heads, attention_head_dim, dropout)
                for _ in range(n_blocks)
            ])
            self.mid_blocks.append(nn.ModuleList([resnet, transformers]))
        
        
        
        
        # ========== UP BLOCKS ==========
        channels_reversed = channels[::-1] + (channels[0],)
        for i in range(len(channels_reversed) - 1):
            input_channel = channels_reversed[i]
            output_channel = channels_reversed[i + 1]
            is_last = i == len(channels_reversed) - 2
            
            # ResNet block (takes concatenated skip connection)
            resnet = ResnetBlock1D(2 * input_channel, output_channel, time_embed_dim)
            
            # Transformer blocks
            transformers = nn.ModuleList([
                Transformer(output_channel, num_heads, attention_head_dim, dropout)
                for _ in range(n_blocks)
            ])
            
            # Upsample or keep resolution
            upsample = (
                Upsample1D(output_channel) if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            
            self.up_blocks.append(nn.ModuleList([resnet, transformers, upsample]))
        
        
        
        
        # ========== FINAL PROJECTION ==========
        self.final_block = Block1D(channels_reversed[-1], channels_reversed[-1])
        self.final_proj = nn.Conv1d(channels_reversed[-1], out_channels, 1)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize network weights"""
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

    def forward(self, x, mu, mask, t):
        """
        Forward pass
        
        Args:
            x: (B, C, T) - noisy input
            mu: (B, C, T) - mean/condition
            t: (B,) - timesteps
            
        Returns:
            output: (B, out_channels, T)
        """
        # # Create mask from input (assumes non-zero values are valid)
        # mask = (x.abs().sum(dim=1, keepdim=True) > 0).float()
        
        # Process timestep
        t_emb = self.time_embeddings(t)
        t_emb = self.time_mlp(t_emb)
        
        # Concatenate input with condition
        x = torch.cat([x, mu], dim=1)   # (B, 2*C, T) 
        
        # ========== ENCODER ==========
        skip_connections = []
        masks = [mask]
        
        for resnet, transformers, downsample in self.down_blocks:
            current_mask = masks[-1]
            
            # ResNet block
            x = resnet(x, current_mask, t_emb)
            
            # Transformer blocks (convert to B, T, C format)
            x_t = x.permute(0, 2, 1)  # B, T, C
            mask_t = current_mask.squeeze(1)  # B, T
            
            for transformer in transformers:
                x_t = transformer(x_t, mask_t)
            
            x = x_t.permute(0, 2, 1)  # B, C, T
            
            # Save for skip connection
            skip_connections.append(x)
            
            # Downsample
            x = downsample(x * current_mask)
            
            # Update mask for next level
            if current_mask.shape[-1] > x.shape[-1]:
                masks.append(current_mask[:, :, ::2])
            else:
                masks.append(current_mask)
        
        # Remove last mask (not used in middle)
        masks = masks[:-1]
        
        # ========== MIDDLE ==========
        mid_mask = masks[-1]
        
        for resnet, transformers in self.mid_blocks:
            # ResNet block
            x = resnet(x, mid_mask, t_emb)
            
            # Transformer blocks
            x_t = x.permute(0, 2, 1)
            mask_t = mid_mask.squeeze(1)
            
            for transformer in transformers:
                x_t = transformer(x_t, mask_t)
            
            x = x_t.permute(0, 2, 1)
        
        # ========== DECODER ==========
        for resnet, transformers, upsample in self.up_blocks:
            current_mask = masks.pop()
            skip = skip_connections.pop()

            # If the upsampled x is 1 pixel larger (due to odd/even mismatch), crop it.
            if x.shape[-1] != skip.shape[-1]:
                x = x[:, :, :skip.shape[-1]]
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            
            # ResNet block
            x = resnet(x, current_mask, t_emb)
            
            # Transformer blocks
            x_t = x.permute(0, 2, 1)
            mask_t = current_mask.squeeze(1)
            
            for transformer in transformers:
                x_t = transformer(x_t, mask_t)
            
            x = x_t.permute(0, 2, 1)
            
            # Upsample
            x = upsample(x * current_mask)
        
        # ========== FINAL PROJECTION ==========
        x = self.final_block(x, mask)
        output = self.final_proj(x * mask)
        
        return output * mask
    
    
# Example usage
model = UNet(
    in_channels=80,
    out_channels=80,
    channels=(256, 256, 512),
    dropout=0.1,
    attention_head_dim=64,
    n_blocks=2,
    num_mid_blocks=2,
    num_heads=8,
)
x = torch.randn(4, 80, 160)  # Batch of 4, 80 channels, 160 timesteps
mu = torch.randn(4, 80, 160)
t = torch.randint(0, 1000, (4,))
mask = torch.ones(4, 1, 160)  # All valid frames
output = model(x, mu, mask, t)
print(output.shape)  # Should be (4, 80, 160)