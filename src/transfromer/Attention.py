import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Attention(nn.Module):
    """
    Minimalistic multi-head attention implementation.
    """
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        out_bias: bool = True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.upcast_attention = upcast_attention
        
        # Query, Key, Value projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        
        # Output projection
        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout)
        ])
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> torch.FloatTensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # If no encoder_hidden_states, this is self-attention
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        
        # Project to Q, K, V
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)
        
        # Reshape to multi-head format: (batch, heads, seq_len, dim_head)
        q = q.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        
        # Compute attention scores
        if self.upcast_attention:
            q = q.float()
            k = k.float()
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        if self.upcast_attention:
            attn_weights = attn_weights.to(v.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back: (batch, seq_len, heads * dim_head)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.heads * self.dim_head)
        
        # Output projection
        for layer in self.to_out:
            attn_output = layer(attn_output)
        
        return attn_output