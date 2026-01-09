import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformer.lora import LoRACompatibleLinear


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


# ADAPTIVE NORMALIZATION LAYERS
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
    
    

class SnakeBeta(nn.Module):


    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
     
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        self.proj = LoRACompatibleLinear(in_features, out_features)

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
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