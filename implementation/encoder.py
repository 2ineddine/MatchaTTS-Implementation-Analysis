# Text Encoder & Duration Predictor

import math

import torch
import torch.nn as nn  
from einops import rearrange

import matcha.utils as utils  
from matcha.utils.model import sequence_mask



class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Position Embeddings (RoPE)
    
    Encodes position by rotating feature pairs in 2D planes.
    2D rotation formula: x' = x*cos(θ) + rotate(x)*sin(θ).
    Each pair rotates at a different frequency based on position.
    
    Args:
        d: Number of dimensions to apply RoPE to (must be even) ( it is the dimesions we apply rope to and not the embedding dimension)
        base: Base for computing rotation frequencies (default: 10000)
    """
    
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.d = int(d)
        self.base = base
        
        # Cache for cos/sin values (computed lazily)
        self.cos_cached = None
        self.sin_cached = None
    
    def _build_cache(self, x: torch.Tensor):
        """
        Pre-compute cos/sin values for all positions up to seq_len.
        
        Formula: θ_i = position / (base^(2i/d))
        """
        # Skip if cache already covers this sequence length
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        
        seq_len = x.shape[0]
        
        # Compute rotation frequencies: θ_i = 1 / (base^(2i/d))
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d))
        theta = theta.to(x.device)
        
        # Position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        
        # Compute angles: position * θ for each (position, dimension) pair
        # Shape: (seq_len, d/2)
        angles = torch.einsum('n,d->nd', positions, theta)
        
        # Duplicate angles for pair-wise rotation
        # Shape: (seq_len, d)
        angles = torch.cat([angles, angles], dim=1)
        
        # Cache cos/sin values with shape for broadcasting
        # Shape: (seq_len, 1, 1, d)
        self.cos_cached = angles.cos()[:, None, None, :]
        self.sin_cached = angles.sin()[:, None, None, :]
    
    def _rotate_half(self, x: torch.Tensor):
        """
        Rearrange features for 2D rotation.
        Transforms: [(x0, x1), (x2, x3), ...] -> [(-x1, x0), (-x3, x2), ...]
        This creates the 90° rotated version needed for rotation formula.
        """
        d_2 = self.d // 2
        # Split into two halves and rearrange
        x1, x2 = x[..., :d_2], x[..., d_2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor (batch, heads, seq_len, dim)
        Returns:
            Rotated tensor with same shape as input
        """
        # Rearrange for position-first indexing
        # (batch, heads, seq_len, dim) -> (seq_len, batch, heads, dim)
        x = rearrange(x, 'b h t d -> t b h d')
        
        self._build_cache(x)
        
        # Split into RoPE-applied and pass-through dimensions
        x_rope = x[..., :self.d]      
        x_pass = x[..., self.d:]      
        
        # Apply 2D rotation formula
        rotated_half = self._rotate_half(x_rope) # First d dimensions : apply rotation
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (rotated_half * self.sin_cached[:x.shape[0]])
        
        # Concatenate rotated and pass-through parts
        output = torch.cat([x_rope, x_pass], dim=-1)
        
        # Restore original shape
        return rearrange(output, 't b h d -> b h t d')



class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + self.eps)
        return x * self.gamma.view(1, -1, *[1]*(x.ndim-2)) + self.beta.view(1, -1, *[1]*(x.ndim-2))