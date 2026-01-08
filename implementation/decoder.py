# The 1D U-Net & Flow Prediction

__author__ = "Massyl A."



import torch
import torch.nn as nn
import torch.nn.functional as F
import math



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

class SnakeBeta(nn.Module):
    def __init__(self, channels, is_seq=False):
        super().__init__()
        # is_seq=True: for transformer (B, T, C) - shape (1, 1, C)
        # is_seq=False: for conv (B, C, T) - shape (1, C, 1)
        if is_seq:
            self.alpha = nn.Parameter(torch.zeros(1, 1, channels))
        else:
            self.alpha = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        alpha = self.alpha.exp()
        x = x + (1 / (alpha + 1e-9)) * (torch.sin(alpha * x) ** 2)
        return x

class ResNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, kernel_size=3, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        
        # Original uses Mish in ResNets for stability
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(8, out_channels),
            nn.Mish(), 
            nn.Dropout(dropout)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
            nn.Dropout(dropout)
        )
        
        self.time_proj = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_dim, out_channels)
        )
        
        self.skip_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, mask, t_emb):
        # 1. Apply Mask to input
        x = x * mask
        
        # 2. First Block
        h = self.block1(x)
        
        # 3. Add Time Embedding (Broadcast over time)
        # t_emb: [B, C] -> [B, C, 1]
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        
        # 4. Second Block
        h = self.block2(h)
        
        # 5. Residual + Masking
        output = h + self.skip_conv(x)
        return output * mask

class TransformerBasic(nn.Module):
    def __init__(self, channels, n_heads=4, head_dim=64, dropout=0.0):
        super().__init__()
        # Use simple attention mechanism
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            SnakeBeta(channels * 4, is_seq=True), # Snake here - for sequence data (B, T, C)
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        # x: [B, C, T] -> [B, T, C] for Transformer
        B, C, T = x.shape
        x_in = x.permute(0, 2, 1)
        
        # Create Attention Mask (T, T) or Key Padding Mask (B, T)
        # For simplicity, we just rely on the input embeddings being masked 
        # But for correctness, we should generate a boolean mask from 'mask' tensor
        # mask shape: [B, 1, T] -> key_padding_mask: [B, T] (True where padded)
        # Note: torch.nn.MHA expects True for ignored positions
        bool_mask = (mask.squeeze(1) == 0) 
        
        # Attention
        h = self.norm1(x_in)
        # MHA requires key_padding_mask to ignore silence
        h, _ = self.attn(h, h, h, key_padding_mask=bool_mask) 
        x_in = x_in + h
        
        # Feed Forward
        h = self.norm2(x_in)
        h = self.ff(h)
        x_in = x_in + h
        
        return x_in.permute(0, 2, 1) * mask

class UnetBlock(nn.Module):
    """
    Combines 1 ResNet + N Transformers.
    """
    def __init__(self, in_ch, out_ch, time_dim, n_transformer_blocks, num_heads, head_dim, dropout):
        super().__init__()
        self.resnet = ResNet1D(in_ch, out_ch, time_dim, dropout=dropout)
        self.transformers = nn.ModuleList([
            TransformerBasic(out_ch, n_heads=num_heads, head_dim=head_dim, dropout=dropout)
            for _ in range(n_transformer_blocks)
        ])
        
    def forward(self, x, t_emb, mask): 
        x = self.resnet(x, mask, t_emb) 
        for transformer in self.transformers:
            x = transformer(x, mask)
        return x



# 2. Dynamic Decoder Implementation

class Decoder(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 channels=(256, 256),
                 dropout=0.05,
                 attention_head_dim=64,
                 n_blocks=1,           # Number of Transformers per level
                 num_mid_blocks=2,     # Number of blocks in the bottleneck
                 num_heads=4
                 ):
        super().__init__()
        
        # 0. Time Embedding and Input Projection
        # --------------------------------------
        time_dim = channels[0] 
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Mish(),
            nn.Linear(time_dim, time_dim)
        )

        # In: Noisy(80) + Condition(80) -> First Hidden Channel
        self.input_proj = nn.Conv1d(in_channels * 2, channels[0], 1)

        # 1. Downsampling Path
        # --------------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        current_ch = channels[0] 
        prev_ch = channels[0] 
        
        for i, ch in enumerate(channels):
            # Block
            block = UnetBlock(
                in_ch=prev_ch, 
                out_ch=ch, 
                time_dim=time_dim,
                n_transformer_blocks=n_blocks,
                num_heads=num_heads,
                head_dim=attention_head_dim,
                dropout=dropout
            )
            self.down_blocks.append(block)
            
            # Downsampler (Conv stride 2)
            down = nn.Conv1d(ch, ch, kernel_size=3, stride=2, padding=1)
            #self.down_blocks.append(down)
            self.downsamples.append(down)
            
            prev_ch = ch

        # 2. Middle Path (Bottleneck)
        # ---------------------------
        self.mid_blocks = nn.ModuleList()
        mid_ch = channels[-1] 
        
        for _ in range(num_mid_blocks):
            block = UnetBlock(
                in_ch=mid_ch,
                out_ch=mid_ch,
                time_dim=time_dim,
                n_transformer_blocks=n_blocks,
                num_heads=num_heads,
                head_dim=attention_head_dim,
                dropout=dropout
            )
            self.mid_blocks.append(block)

        # 3. Upsampling Path
        # ------------------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        reversed_channels = list(reversed(channels))
        current_ch = reversed_channels[0]
        
        for i in range(len(reversed_channels)):
            target_ch = reversed_channels[i]
            
            # Upsampler
            up = nn.ConvTranspose1d(current_ch, current_ch, kernel_size=4, stride=2, padding=1)
            self.upsamples.append(up)
            
            # Block
            block = UnetBlock(
                in_ch=current_ch + target_ch, 
                out_ch=target_ch, 
                time_dim=time_dim,
                n_transformer_blocks=n_blocks,
                num_heads=num_heads,
                head_dim=attention_head_dim,
                dropout=dropout
            )
            self.up_blocks.append(block)
            
            current_ch = target_ch

        # 4. Final Output
        # ---------------
        self.final_norm = nn.GroupNorm(8, channels[0])
        self.final_act = SnakeBeta(channels[0])
        self.final_proj = nn.Conv1d(channels[0], out_channels, 1)

    def forward(self, x, mask, mu, t):
        """
        Args:
            x: Noisy input [B, C, T]
            mask: Binary mask [B, 1, T]
            mu: Conditioner [B, C, T]
            t: Timesteps [B]
        """
        # Time Embedding
        t_emb = self.time_mlp(t)
        
        # Initial Projection
        x_in = torch.cat([x, mu], dim=1) 
        x = self.input_proj(x_in)
        x = x * mask # Apply mask immediately
        
        # Stacks to store skip connections and their corresponding masks
        skips = []
        mask_stack = [mask]
        
        # --- DOWN ---
        # --- DOWN ---
        for block, downsample in zip(self.down_blocks, self.downsamples):
            # 1. Process Block (The UnetBlock)
            current_mask = mask_stack[-1]
            x = block(x, t_emb, current_mask) # Now 'block' is definitely a UnetBlock
            x = x * current_mask
            
            # 2. Save Skip connection (Before downsampling)
            skips.append(x) 
            
            # 3. Downsample Feature (The Conv1d)
            # Conv1d only takes 'x', so we don't pass t_emb or mask here
            x = downsample(x)
            
            # 4. Downsample Mask
            # Use slicing for stride 2 logic
            new_mask = current_mask[:, :, ::2]
            
            # Safety check for padding/shape alignment
            if new_mask.shape[2] != x.shape[2]:
                new_mask = new_mask[:, :, :x.shape[2]]
            
            mask_stack.append(new_mask)
            x = x * new_mask
            
        # --- MID ---
        mid_mask = mask_stack[-1]
        for block in self.mid_blocks:
            x = block(x, t_emb, mid_mask)
            x = x * mid_mask
            
        # --- UP ---
        for block, upsample in zip(self.up_blocks, self.upsamples):
            # 1. Retrieve Skip and its Mask
            skip = skips.pop()
            
            # We pop the current low-res mask (we are done with it)
            # The next mask in the stack (now at -1) is the one that matches 'skip'
            _ = mask_stack.pop() 
            target_mask = mask_stack[-1]
            
            # 2. Upsample
            x = upsample(x)
            
            # 3. Align shapes
            # ConvTranspose1d can result in size +1 or -1 vs the skip connection
            # We strictly force x to match skip's length
            if x.shape[2] != skip.shape[2]:
                x = x[:, :, :skip.shape[2]]
                
            # 4. Concat
            x = torch.cat([x, skip], dim=1)
            
            # 5. Process
            x = block(x, t_emb, target_mask)
            x = x * target_mask
            
        # --- FINAL ---
        x = self.final_norm(x)
        x = self.final_act(x)
        velocity = self.final_proj(x)
        
        return velocity * mask # Ensure output is masked with original mask