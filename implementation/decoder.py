# The 1D U-Net & Flow Prediction

__author__ = "Massyl A."



import torch
import torch.nn as nn
import torch.nn.functional as F
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
    

    