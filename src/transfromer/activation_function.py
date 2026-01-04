import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



class GELU(nn.Module):
    """GELU activation function with linear projection."""
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate=self.approximate)


class ApproximateGELU(nn.Module):
    """Approximate GELU activation (tanh approximation)."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate="tanh")


class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


# ============================================================================
# ADAPTIVE NORMALIZATION LAYERS
# ============================================================================

class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization."""
    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        if num_embeddings is not None:
            self.linear = nn.Linear(num_embeddings, embedding_dim * 2)
        else:
            self.linear = None
    
    def forward(self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(x)
        if timestep is not None and self.linear is not None:
            emb = self.linear(timestep)
            scale, shift = emb.chunk(2, dim=-1)
            x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Normalization Zero."""
    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        if num_embeddings is not None:
            self.linear = nn.Linear(num_embeddings, embedding_dim * 6)
        else:
            self.linear = None
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> tuple:
        x = self.norm(x)
        if timestep is not None and self.linear is not None:
            emb = timestep
            if class_labels is not None:
                emb = emb + class_labels
            emb = self.linear(emb)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
        return x, None, None, None, None