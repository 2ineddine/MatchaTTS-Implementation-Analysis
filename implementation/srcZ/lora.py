import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRACompatibleLinear(nn.Linear):
    """
    LoRA-compatible Linear layer - fully compatible with nn.Linear.
    
    Inherits from nn.Linear so it behaves exactly like a standard linear layer,
    but can be extended with LoRA weights without breaking existing code.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        # LoRA parameters (initialized as None, can be added later)
        self.lora_A = None
        self.lora_B = None
        self.lora_scale = 1.0
        self.lora_rank = 0
    
    def forward(self, x: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass with optional LoRA.
        
        Args:
            x: Input tensor
            scale: Optional scale factor for LoRA (defaults to self.lora_scale)
        
        Returns:
            Output tensor
        """
        # Standard linear transformation
        output = F.linear(x, self.weight, self.bias)
        
        # Add LoRA contribution if LoRA weights exist
        if self.lora_A is not None and self.lora_B is not None:
            lora_scale = scale if scale is not None else self.lora_scale
            lora_output = (x @ self.lora_A.T @ self.lora_B.T) * (lora_scale / max(self.lora_rank, 1))
            output = output + lora_output
        
        return output
    
    def add_lora(self, rank: int = 4, scale: float = 1.0):
        """
        Add LoRA parameters to this linear layer.
        
        Args:
            rank: Rank of LoRA decomposition
            scale: Scale factor for LoRA contribution
        """
        self.lora_rank = rank
        self.lora_scale = scale
        
        # Initialize LoRA matrices
        # A: (rank, in_features) - initialized with small random values
        # B: (out_features, rank) - initialized with zeros
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features, device=self.weight.device, dtype=self.weight.dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=self.weight.device, dtype=self.weight.dtype))
    
    def remove_lora(self):
        """Remove LoRA parameters and return to standard linear layer."""
        self.lora_A = None
        self.lora_B = None
        self.lora_rank = 0
    
    def merge_lora(self):
        """Merge LoRA weights into the main weight matrix."""
        if self.lora_A is not None and self.lora_B is not None:
            with torch.no_grad():
                # Compute LoRA contribution: B @ A
                lora_weight = (self.lora_B @ self.lora_A) * (self.lora_scale / max(self.lora_rank, 1))
                # Add to main weight
                self.weight.data += lora_weight
            # Remove LoRA parameters
            self.remove_lora()