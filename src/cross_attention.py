import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    Minimalist Cross-Attention for BasicTransformerBlock
    
    In your case:
    - Queries: From (μ + x_t) - the combined noisy + clean spectrogram
    - Keys & Values: From μ - the clean reference spectrogram only
    - Each noisy pixel asks the clean reference: "What should I look like?"
    """
    def __init__(self, dim, cross_attention_dim, num_heads=8, dim_head=64, dropout=0.0):
        """
        Args:
            dim: Feature dimension of queries (from noisy spectrogram)
            cross_attention_dim: Feature dimension of keys/values (from clean μ)
            num_heads: Number of attention heads
            dim_head: Dimension per head
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * num_heads
        
        # Query projection - from noisy spectrogram (μ + x_t)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        
        # Key and Value projections - from clean reference μ
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, context, mask=None):
        """
        Args:
            x: [batch, seq_len, dim] 
               Queries from (μ + x_t) - noisy spectrogram
               Example: [4, 65536, 512]
            
            context: [batch, context_len, cross_attention_dim]
                    Keys & Values from μ - clean reference spectrogram
                    Example: [4, 65536, 512]
                    (In your case context_len == seq_len, same shape!)
            
            mask: [batch, context_len] - optional mask for context
        
        Returns:
            [batch, seq_len, dim] - attended features
        """
        batch_size, seq_len, _ = x.shape
        _, context_len, _ = context.shape
        
        print(f"\n=== CROSS-ATTENTION INPUT ===")
        print(f"x (queries from noisy): {x.shape}")      # [4, 65536, 512]
        print(f"context (K,V from μ): {context.shape}")  # [4, 65536, 512]
        
        # ========================================================================
        # STEP 1: Project to Q, K, V
        # KEY DIFFERENCE FROM SELF-ATTENTION:
        # - Q comes from x (noisy spectrogram)
        # - K, V come from context (clean reference μ)
        # ========================================================================
        q = self.to_q(x)           # Queries from noisy: [batch, seq_len, inner_dim]
        k = self.to_k(context)     # Keys from clean μ: [batch, context_len, inner_dim]
        v = self.to_v(context)     # Values from clean μ: [batch, context_len, inner_dim]
        
        print(f"\n=== AFTER Q,K,V PROJECTION ===")
        print(f"q.shape: {q.shape}")  # [4, 65536, 512]
        print(f"k.shape: {k.shape}")  # [4, 65536, 512]
        print(f"v.shape: {v.shape}")  # [4, 65536, 512]
        print(f"Q comes from noisy (μ + x_t)")
        print(f"K,V come from clean reference μ")
        
        # ========================================================================
        # STEP 2: Reshape for multi-head attention
        # ========================================================================
        q = q.view(batch_size, seq_len, self.num_heads, self.dim_head)
        k = k.view(batch_size, context_len, self.num_heads, self.dim_head)
        v = v.view(batch_size, context_len, self.num_heads, self.dim_head)
        
        # Transpose to [batch, num_heads, seq, dim_head]
        q = q.transpose(1, 2)  # [batch, heads, seq_len, dim_head]
        k = k.transpose(1, 2)  # [batch, heads, context_len, dim_head]
        v = v.transpose(1, 2)  # [batch, heads, context_len, dim_head]
        
        print(f"\n=== AFTER RESHAPE ===")
        print(f"q.shape: {q.shape}")  # [4, 8, 65536, 64]
        print(f"k.shape: {k.shape}")  # [4, 8, 65536, 64]
        print(f"v.shape: {v.shape}")  # [4, 8, 65536, 64]
        
        # ========================================================================
        # STEP 3: Compute cross-attention scores
        # scores[i,j] = similarity between noisy pixel i and clean pixel j
        # ========================================================================
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        print(f"\n=== ATTENTION SCORES ===")
        print(f"scores.shape: {scores.shape}")  # [4, 8, 65536, 65536]
        print(f"scores[b, h, i, j] = how much noisy pixel i should attend to clean pixel j")
        print(f"Each noisy pixel asks ALL clean pixels: 'How similar are you to what I should be?'")
        
        # ========================================================================
        # STEP 4: Apply mask if provided
        # ========================================================================
        if mask is not None:
            print(f"\n=== MASKING ===")
            print(f"mask.shape: {mask.shape}")  # [4, 65536]
            mask = mask.unsqueeze(1).unsqueeze(2)  # [4, 1, 1, 65536]
            scores = scores.masked_fill(mask == 0, float('-inf'))
            print(f"Masked positions in clean μ won't be attended to")
        
        # ========================================================================
        # STEP 5: Softmax to get attention weights
        # ========================================================================
        attn_weights = F.softmax(scores, dim=-1)
        
        print(f"\n=== ATTENTION WEIGHTS ===")
        print(f"attn_weights.shape: {attn_weights.shape}")  # [4, 8, 65536, 65536]
        print(f"Each noisy pixel has a distribution over ALL clean pixels")
        print(f"High weight = 'This clean pixel tells me what I should be'")
        
        # ========================================================================
        # STEP 6: Apply attention weights to values (from clean μ)
        # ========================================================================
        out = torch.matmul(attn_weights, v)
        
        print(f"\n=== WEIGHTED SUM ===")
        print(f"out.shape: {out.shape}")  # [4, 8, 65536, 64]
        print(f"Each noisy pixel now has information from relevant clean pixels")
        
        # ========================================================================
        # STEP 7: Reshape back
        # ========================================================================
        out = out.transpose(1, 2)  # [batch, seq_len, heads, dim_head]
        out = out.contiguous().view(batch_size, seq_len, -1)  # [batch, seq_len, inner_dim]
        
        print(f"\n=== MERGE HEADS ===")
        print(f"out.shape: {out.shape}")  # [4, 65536, 512]
        
        # Final projection
        out = self.to_out(out)
        
        print(f"\n=== FINAL OUTPUT ===")
        print(f"out.shape: {out.shape}")  # [4, 65536, 512]
        print(f"Each noisy pixel has been guided by the clean reference μ")
        
        return out


# ============================================================================
# EXAMPLE USAGE WITH YOUR SPECTROGRAM CASE
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("CROSS-ATTENTION: Noisy Spectrogram conditioned on Clean Reference μ")
    print("="*80)
    
    # Spectrogram dimensions
    batch_size = 4
    height = 128      # frequency bins
    width = 512       # time frames
    seq_len = height * width  # 65536
    dim = 512         # feature dimension
    
    # Simulate features from (μ + x_t) - noisy combined
    noisy_features = torch.randn(batch_size, seq_len, dim)
    print(f"\nNoisy features (μ + x_t): {noisy_features.shape}")
    
    # Simulate features from μ only - clean reference
    clean_mu_features = torch.randn(batch_size, seq_len, dim)
    print(f"Clean μ features: {clean_mu_features.shape}")
    
    # Initialize cross-attention
    cross_attn = CrossAttention(
        dim=dim,                      # Query dimension (from noisy)
        cross_attention_dim=dim,      # Key/Value dimension (from clean μ)
        num_heads=8,
        dim_head=64,
        dropout=0.1
    )
    
    # Forward pass
    attended_features = cross_attn(
        x=noisy_features,           # Queries: from (μ + x_t)
        context=clean_mu_features   # Keys & Values: from μ
    )
    
    print(f"\n" + "="*80)
    print("WHAT HAPPENED:")
    print("  1. Each pixel in noisy (μ + x_t) generated a query")
    print("  2. Each pixel in clean μ provided keys and values")
    print("  3. Noisy pixels attended to relevant clean pixels")
    print("  4. Result: Noisy pixels know what they should look like!")
    print("="*80)
    
    print(f"\n" + "="*80)
    print("CONCRETE EXAMPLE:")
    print("  Noisy pixel at 440Hz, t=0.5s (Query):")
    print("    - 'I have noise + weak signal'")
    print("  Clean μ pixel at 440Hz, t=0.5s (Key/Value):")
    print("    - 'You should have strong signal at 0.8 amplitude'")
    print("  Cross-attention:")
    print("    - High attention weight (0.9) to this clean pixel")
    print("    - Result: Noisy pixel learns 'I should be strong, remove noise!'")
    print("="*80)