# Implementation of the matchatts transformer block of the Unet (no positional embedding, snakebeta activation)

__author__ = "Massyl A."



from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from diffusers.models.lora import LoRACompatibleLinear
from diffusers.models.attention_processor import Attention



class Linear_plus_SnakeBeta(nn.Module):
    """
    Linear Layer + SnakeBeta Activation
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.alpha_beta_size = out_features if isinstance(out_features, list) else [out_features]
        self.proj = LoRACompatibleLinear(in_features, out_features)

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(self.alpha_beta_size) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.alpha_beta_size) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(self.alpha_beta_size) * alpha)
            self.beta = nn.Parameter(torch.ones(self.alpha_beta_size) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta

        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

        return x
    


class FeedForward(nn.Module):
    r"""
    A feed-forward layer of the transformer (Linear + Activation + Linear)

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim_in`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim_in * mult)
        dim_out = dim_out if dim_out is not None else dim_in

        # Activation function
        act_fn = Linear_plus_SnakeBeta(dim_in, inner_dim)

        # Assemble layers
        self.net = nn.ModuleList([])
        # project in + activation
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(LoRACompatibleLinear(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
    


class MatchaTransformer(nn.Module):
    """
    A Straight Forward Non-Modular implementation of the Transformer block used in Shivam's MatchaTTS Paper.
    Structure: Pre-Norm -> Self-Attention -> Residual -> Pre-Norm -> FeedForward -> Residual
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        ff_mult: int = 4  # Multiplier for the FeedForward inner dimension
    ):
        super().__init__()

        # 1. Self-Attention Layer
        # We use standard LayerNorm because time conditioning happens in the ResNet block
        self.norm1 = nn.LayerNorm(dim)
        
        # If using the diffusers version, ensure 'cross_attention_dim' is None.
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,             # Usually False for modern Transformers
            cross_attention_dim=None # Pure Self-Attention
        )

        # 2. Feed-Forward Layer
        # LayerNorm again
        self.norm2 = nn.LayerNorm(dim)
        
        # This uses the custom FF (Linear + SnakeBeta + Linear) defined above
        self.ff = FeedForward(
            dim_in=dim,
            dim_out=dim,
            mult=ff_mult,
            dropout=dropout,
            final_dropout=False 
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: Input tensor of shape (Batch, Time, Dim)
            attention_mask: Optional mask for attention (Batch, Time)
        """
        
        # --- Block 1: Self-Attention ---
        # 1. Pre-Normalization
        norm_hidden_states = self.norm1(hidden_states)
        
        # 2. Attention
        attn_output = self.attn1(
            norm_hidden_states,
            attention_mask=attention_mask
        )
        
        # 3. Residual Connection
        hidden_states = hidden_states + attn_output

        # --- Block 2: Feed-Forward ---
        # 1. Pre-Normalization
        norm_hidden_states = self.norm2(hidden_states)
        
        # 2. Feed-Forward (with SnakeBeta)
        ff_output = self.ff(norm_hidden_states)
        
        # 3. Residual Connection
        hidden_states = hidden_states + ff_output

        return hidden_states