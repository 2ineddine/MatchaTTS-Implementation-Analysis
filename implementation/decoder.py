# The 1D U-Net & Flow Prediction

__author__ = "Massyl A."
__author__ = "Zinnedine"



import torch
import torch.nn as nn
import math

from matcha_transformer import MatchaTransformer



# 1. The Building Blocks

class SinusoidalPosEmb(torch.nn.Module):
    """
    Embed the flow matching time step into a vector
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0 # SinusoidalPosEmb requires dim to be even

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
    """
    TimestepEmbedding for Matcha-TTS.
    Create a rich, high-level representation of the timestep vector
    Structure: Linear -> SiLU -> Linear
    Size : (Batch, in_channels) -> (Batch, time_embed_dim_out)
    """
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()

        # 1. First Linear Layer
        # Projects from sinusoidal dimension (e.g., 128) to the hidden dimension (e.g., 512)
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        # 2. Activation
        # Hardcoded to SiLU (Swish), which is standard for diffusion models
        self.act = nn.SiLU()

        # 3. Second Linear Layer
        # Projects from hidden dimension to hidden dimension (refining the vector)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        """
        Args:
            sample: Sinusoidal time embeddings of shape (Batch, in_channels)
        """
        # 1. Project
        sample = self.linear_1(sample)
        
        # 2. Activate
        sample = self.act(sample)
        
        # 3. Project
        sample = self.linear_2(sample)
        
        return sample



class Block1D(torch.nn.Module):

    """
    Basic 1D Convolutional Unit
    Convolution → Normalization → Activation

    Steps : 
    1 - masking
    2 - Convolution D with 3 sized kernel
    3 - GroupNorm to stabilize training
    4 - Mish activation : x⋅tanh(ln(1+ex))
    5 - masking
    """

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        x = x * mask
        output = self.block(x)
        output = output * mask
        return output 
    


class ResnetBlock1D(torch.nn.Module):

    """
    1 - Block1D : Feature processing
    2 - Time injection : convert to correct dim through simple MLP
    3 - Block1D : Feature processing
    4 - Residual Connection : Adds Conv1d(x*mask)
    """

    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()

        self.block1 = Block1D(dim, dim_out, groups=groups)

        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))
        
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output
    


class Downsample1D(nn.Module):

    """
    Downsample block using strided ConvD : (B,C,T) -> (B,C,2*T)
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)
    


class Upsample1D(nn.Module):
    """1D upsampling layer using Transposed Convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv_transpose=True, name="conv"):
        super().__init__()

        self.channels = channels
        self.out_channels = channels
        self.name = name

        self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)

    def forward(self, inputs):

        # Sanity check to ensure dimensions match
        assert inputs.shape[1] == self.channels

        # Applies the upsampling (B, C, T) -> (B, C, 2*T)
        outputs = self.conv(inputs)
    
        return outputs
    

    
# 1. Main Decoder Block

class Decoder(nn.Module):

    """
    Main Unet that inferes the velocity field based on : noisy spectrogram x(t), mean spectrogram mu, timestep t, and the mask 
    """

    def __init__(self, 
                 in_channels,
                 out_channels,
                 downsampling_upsampling_channels=(256, 256),   # The channel dimension for each downsampling/upsampling block
                 num_mid_blocks=2,                              # Number of blocks in the middle part of the Unet (the bottleneck)
                 dropout=0.05,                                  # dropout value of the Unet blocks (for resnet and transformer)
                 n_transformer_per_block=1,                     # number of transformers in a Unet's block
                 attention_head_dim=64,                         # Size of the key/query/value vectors inside the Transformer's attention
                 num_attention_heads=2                          # Number of attention heads in the multi-head attention transformers
                 ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # --------------------------------------------------------
        # 1. Time Embedding Modules
        # --------------------------------------------------------
        # Sinusoidal -> Linear -> SiLU -> Linear
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        
        # The hidden dimension for time is typically 4x the base channel size (transformer block parameter)
        time_embed_dim = downsampling_upsampling_channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim
        )

        # --------------------------------------------------------
        # 2. Down Blocks (Encoder)
        # --------------------------------------------------------
        self.down_blocks = nn.ModuleList([])
        
        # Start with the input spectrogram dimension (e.g., 80)
        output_channel = in_channels
        
        for i in range(len(downsampling_upsampling_channels)):
            input_channel = output_channel
            output_channel = downsampling_upsampling_channels[i]
            is_last = i == len(downsampling_upsampling_channels) - 1

            # A. ResNet Block (Mixes features + Time embedding)
            resnet = ResnetBlock1D(
                dim=input_channel, 
                dim_out=output_channel, 
                time_emb_dim=time_embed_dim
            )

            # B. Transformer Blocks (Self-Attention)
            transformer_blocks = nn.ModuleList([
                MatchaTransformer(
                    dim=output_channel,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout
                )
                for _ in range(n_transformer_per_block)
            ])

            # C. Downsampling (Strided Conv) or Identity Conv if last block
            downsample = (
                Downsample1D(output_channel) 
                if not is_last 
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        # --------------------------------------------------------
        # 3. Mid Blocks (Bottleneck)
        # --------------------------------------------------------
        self.mid_blocks = nn.ModuleList([])
        
        # The channels remain constant in the bottleneck (e.g., 256 -> 256)
        mid_channel = downsampling_upsampling_channels[-1]

        for i in range(num_mid_blocks):
            resnet = ResnetBlock1D(
                dim=mid_channel, 
                dim_out=mid_channel, 
                time_emb_dim=time_embed_dim
            )
            
            transformer_blocks = nn.ModuleList([
                MatchaTransformer(
                    dim=mid_channel,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout
                )
                for _ in range(n_transformer_per_block)
            ])

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        # --------------------------------------------------------
        # 4. Up Blocks (Decoder)
        # --------------------------------------------------------
        self.up_blocks = nn.ModuleList([])
        
        # Reverse the channel list for the path up, adding the first channel at the end
        # Example: (256, 256) -> (256, 256, 256)
        up_channels = downsampling_upsampling_channels[::-1] + (downsampling_upsampling_channels[0],)
        
        for i in range(len(up_channels) - 1):
            input_channel = up_channels[i]
            output_channel = up_channels[i + 1]
            is_last = i == len(up_channels) - 2

            # A. ResNet Block
            # NOTE: dim is 2 * input_channel because we concatenate the skip connection
            resnet = ResnetBlock1D(
                dim=2 * input_channel, 
                dim_out=output_channel, 
                time_emb_dim=time_embed_dim
            )

            # B. Transformer Blocks
            transformer_blocks = nn.ModuleList([
                MatchaTransformer(
                    dim=output_channel,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout
                )
                for _ in range(n_transformer_per_block)
            ])

            # C. Upsampling (Transposed Conv) or Identity Conv if last block
            upsample = (
                Upsample1D(output_channel) 
                if not is_last 
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        # --------------------------------------------------------
        # 5. Final Projection
        # --------------------------------------------------------
        self.final_block = Block1D(up_channels[-1], up_channels[-1])
        self.final_proj = nn.Conv1d(up_channels[-1], out_channels, 1)

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

    def forward(self, x, mask, mu, t):

        """
            Args:
                x (Tensor): Noisy Mel-spectrogram (Batch, Channels, Time)
                mask (Tensor): Binary mask for valid sequence length (Batch, 1, Time)
                mu (Tensor): Predicted average spectrogram from Encoder (Batch, Channels, Time)
                t (Tensor): Time steps (Batch,)
        """
    
        # --------------------------------------------------------
        # 1. Prepare Inputs
        # --------------------------------------------------------
        # Compute the global time vector (B, Time_Embed_Dim)
        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        # Concatenate the noisy input 'x' with the condition 'mu'
        # Shape becomes: (Batch, in_channels, Time). 
        # Note: self.in_channels must equal 2 * n_mels if defined this way.
        x = torch.cat([x, mu], dim=1)
        
        # --------------------------------------------------------
        # 2. Down-Sampling Path (Encoder)
        # --------------------------------------------------------
        hiddens = []       # Store features for skip connections
        masks = [mask]     # Store masks for each resolution level

        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            
            # A. ResNet: Local features + Time Injection
            x = resnet(x, mask_down, t)
            
            # B. Transformer: Global Self-Attention
            # Permute for Transformer: (B, C, T) -> (B, T, C)
            x = x.permute(0, 2, 1)
            for transformer_block in transformer_blocks:
                x = transformer_block(x, attention_mask=mask_down)
            # Permute back: (B, T, C) -> (B, C, T)
            x = x.permute(0, 2, 1)

            # Save 'x' before downsampling for the skip connection later
            hiddens.append(x)
            
            # C. Downsample (Strided Conv)
            x = downsample(x * mask_down)
            
            # D. Update Mask for next level (slice every 2nd element)
            masks.append(mask_down[:, :, ::2])

        # --------------------------------------------------------
        # 3. Mid-Blocks (Bottleneck)
        # --------------------------------------------------------
        masks = masks[:-1] # Drop the last mask (it matches the downsampled output)
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            # A. ResNet
            x = resnet(x, mask_mid, t)
            
            # B. Transformer
            x = x.permute(0, 2, 1)
            for transformer_block in transformer_blocks:
                x = transformer_block(x, attention_mask=mask_mid)
            x = x.permute(0, 2, 1)

        # --------------------------------------------------------
        # 4. Up-Sampling Path (Decoder)
        # --------------------------------------------------------
        for resnet, transformer_blocks, upsample in self.up_blocks:
            # Retrieve the mask and skip connection for this resolution
            mask_up = masks.pop()
            skip = hiddens.pop()

            # If the upsampled x is 1 pixel larger (due to odd/even mismatch), crop it.
            if x.shape[-1] != skip.shape[-1]:
                x = x[:, :, :skip.shape[-1]]
            
            # Concatenate Skip Connection with current 'x'
            # Shape: (B, 2*C, T) -> ResNet handles the channel reduction
            x = torch.cat([x, skip], dim=1)
            
            # A. ResNet
            x = resnet(x, mask_up, t)
            
            # B. Transformer
            x = x.permute(0, 2, 1)
            for transformer_block in transformer_blocks:
                x = transformer_block(x, attention_mask=mask_up)
            x = x.permute(0, 2, 1)
            
            # C. Upsample (Transposed Conv)
            x = upsample(x * mask_up)

        # --------------------------------------------------------
        # 5. Final Projection
        # --------------------------------------------------------
        # Apply final processing and project to output channels (velocity)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)

        return output * mask # Ensure output is masked with original mask