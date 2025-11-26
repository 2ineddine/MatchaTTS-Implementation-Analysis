import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """
    Minimalist Self-Attention for BasicTransformerBlock
    
    For a 128x512 spectrogram:
    - 128 = frequency bins (height)
    - 512 = time frames (width)
    - We flatten to sequence: 128*512 = 65536 positions
    - Each position has 'dim' features
    """
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5  # 1/sqrt(dim_head) for scaled dot-product
        
        inner_dim = dim_head * num_heads  # Total dimension across all heads
        
        # Three linear projections: Q, K, V
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        """
        For 128x512 spectrogram case:
        
        Args:
            x: [batch, seq_len, dim] where seq_len = 128*512 = 65536
               - batch: how many spectrograms (e.g., 4)
               - seq_len: number of positions (65536 for 128x512)
               - dim: feature dimension per position (e.g., 512)
            
            mask: [batch, seq_len] - optional, tells which positions to ignore
                  - 1 = attend to this position
                  - 0 = ignore this position
        
        Returns:
            [batch, seq_len, dim] - attended features
        """
        batch_size, seq_len, _ = x.shape
        # Example: batch_size=4, seq_len=65536, dim=512
        
        print(f"\n=== INPUT ===")
        print(f"x.shape: {x.shape}")  # [4, 65536, 512]
        print(f"Interpretation: 4 spectrograms, each with 65536 positions (128*512), 512 features per position")
        
        # ========================================================================
        # STEP 1: Project to Q, K, V
        # ========================================================================
        q = self.to_q(x)  # [batch, seq_len, inner_dim]
        k = self.to_k(x)  # [batch, seq_len, inner_dim]
        v = self.to_v(x)  # [batch, seq_len, inner_dim]
        
        print(f"\n=== AFTER Q,K,V PROJECTION ===")
        print(f"q.shape: {q.shape}")  # [4, 65536, 512] (if inner_dim=512)
        print(f"Each position now has {self.num_heads * self.dim_head} features")
        
        # ========================================================================
        # STEP 2: Reshape for multi-head attention
        # ========================================================================
        # We want to split the features into multiple "heads"
        # Think of it as: instead of 1 attention with 512 features,
        # we have 8 attentions with 64 features each
        
        q = q.view(batch_size, seq_len, self.num_heads, self.dim_head)
        k = k.view(batch_size, seq_len, self.num_heads, self.dim_head)
        v = v.view(batch_size, seq_len, self.num_heads, self.dim_head)
        
        print(f"\n=== AFTER RESHAPING TO HEADS ===")
        print(f"q.shape: {q.shape}")  # [4, 65536, 8, 64]
        print(f"Interpretation: 4 spectrograms, 65536 positions, 8 heads, 64 features per head")
        
        # Now transpose to put heads before sequence
        # Why? So each head can independently process all positions
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, dim_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        print(f"\n=== AFTER TRANSPOSE ===")
        print(f"q.shape: {q.shape}")  # [4, 8, 65536, 64]
        print(f"Interpretation: 4 spectrograms, 8 heads, each head processes 65536 positions with 64 features")
        print(f"What transpose did: swapped axes 1 and 2")
        print(f"  Before: [batch=4, seq=65536, heads=8, dim=64]")
        print(f"  After:  [batch=4, heads=8, seq=65536, dim=64]")
        
        # ========================================================================
        # STEP 3: Compute attention scores
        # ========================================================================
        # Q @ K^T creates a matrix where entry [i,j] = similarity between position i and j
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        print(f"\n=== ATTENTION SCORES ===")
        print(f"scores.shape: {scores.shape}")  # [4, 8, 65536, 65536]
        print(f"Interpretation: For each spectrogram, for each head, we have a 65536x65536 matrix")
        print(f"  scores[b, h, i, j] = how much position i should attend to position j")
        print(f"  This is HUGE: 65536*65536 = 4,294,967,296 values per head!")
        print(f"  In practice, you'd use patches or windowed attention to make this tractable")
        
        # ========================================================================
        # STEP 4: Apply mask if provided
        # ========================================================================
        if mask is not None:
            print(f"\n=== MASKING ===")
            print(f"mask.shape before unsqueeze: {mask.shape}")  # [4, 65536]
            print(f"mask tells us which positions are valid (1) or should be ignored (0)")
            
            # Add dimensions for heads and query positions
            mask = mask.unsqueeze(1).unsqueeze(2)
            
            print(f"mask.shape after unsqueeze(1).unsqueeze(2): {mask.shape}")  # [4, 1, 1, 65536]
            print(f"Why add dimensions?")
            print(f"  - Dimension 1 (heads): We want the same mask for all 8 heads")
            print(f"  - Dimension 2 (query positions): We want the same mask for all query positions")
            print(f"  Result: [batch=4, heads=1, queries=1, keys=65536]")
            print(f"  Broadcasting will expand this to match scores shape [4, 8, 65536, 65536]")
            
            # Replace masked positions with -inf so they become 0 after softmax
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
            print(f"What masked_fill does:")
            print(f"  - Where mask==0 (invalid positions), set score to -inf")
            print(f"  - After softmax, exp(-inf) = 0")
            print(f"  - Result: invalid positions don't contribute to attention")
        
        # ========================================================================
        # STEP 5: Softmax to get attention weights
        # ========================================================================
        attn_weights = F.softmax(scores, dim=-1)
        
        print(f"\n=== ATTENTION WEIGHTS ===")
        print(f"attn_weights.shape: {attn_weights.shape}")  # [4, 8, 65536, 65536]
        print(f"After softmax on dim=-1 (last dimension):")
        print(f"  - Each row sums to 1")
        print(f"  - attn_weights[b, h, i, :] = distribution over all positions that position i attends to")
        
        # ========================================================================
        # STEP 6: Apply attention weights to values
        # ========================================================================
        out = torch.matmul(attn_weights, v)
        
        print(f"\n=== WEIGHTED SUM OF VALUES ===")
        print(f"out.shape: {out.shape}")  # [4, 8, 65536, 64]
        print(f"Each position now has information from all positions it attended to")
        
        # ========================================================================
        # STEP 7: Reshape back and project to output
        # ========================================================================
        out = out.transpose(1, 2)
        
        print(f"\n=== TRANSPOSE BACK ===")
        print(f"out.shape after transpose: {out.shape}")  # [4, 65536, 8, 64]
        print(f"We moved heads back to position 2")
        
        # Now we need to merge all heads back together
        # contiguous() ensures memory is laid out correctly for view()
        out = out.contiguous().view(batch_size, seq_len, -1)
        
        print(f"\n=== RESHAPE TO MERGE HEADS ===")
        print(f"out.shape after contiguous().view(): {out.shape}")  # [4, 65536, 512]
        print(f"What happened:")
        print(f"  - contiguous(): Reorganizes memory so data is stored sequentially")
        print(f"  - view(batch_size, seq_len, -1): Reshape to [4, 65536, 512]")
        print(f"    - -1 means 'infer this dimension' = 8*64 = 512")
        print(f"  - Result: All 8 heads (each 64 dim) merged into single 512 dim")
        
        print(f"\n=== WHAT DOES CONTIGUOUS MEAN? ===")
        print(f"After transpose, data in memory is not sequential:")
        print(f"  Original: [batch, seq, heads, dim] stored as [0,0,0,0], [0,0,0,1], ...")
        print(f"  After transpose: [batch, heads, seq, dim] but memory still in old order!")
        print(f"  contiguous() copies data to new sequential layout")
        print(f"  This is required before view() can work")
        
        # Final linear projection
        out = self.to_out(out)
        
        print(f"\n=== FINAL OUTPUT ===")
        print(f"out.shape: {out.shape}")  # [4, 65536, 512]
        print(f"Back to original shape!")
        
        return out


# ============================================================================
# DETAILED EXAMPLE WITH 128x512 SPECTROGRAM
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("SELF-ATTENTION WITH 128x512 SPECTROGRAM")
    print("="*80)
    
    # Your spectrogram dimensions
    height = 128  # frequency bins
    width = 512   # time frames
    seq_len = height * width  # 65536 total positions
    
    # Feature dimension per position
    dim = 512  # each position has 512 features
    
    # Batch size
    batch_size = 4  # processing 4 spectrograms at once
    
    print(f"\nSpectrogram dimensions:")
    print(f"  Height (frequency bins): {height}")
    print(f"  Width (time frames): {width}")
    print(f"  Total positions: {seq_len}")
    print(f"  Feature dimension: {dim}")
    print(f"  Batch size: {batch_size}")
    
    # Create noisy input
    # In reality, your spectrogram would be processed by earlier layers
    # to create these feature vectors
    noisy_features = torch.randn(batch_size, seq_len, dim)
    
    # Optional: create a mask (e.g., to ignore padded regions)
    # mask = torch.ones(batch_size, seq_len)  # All positions valid
    # mask[:, 60000:] = 0  # Last 5536 positions invalid
    
    # Initialize self-attention
    self_attn = SelfAttention(dim=dim, num_heads=8, dim_head=64, dropout=0.1)
    
    # Forward pass
    attended_features = self_attn(noisy_features, mask=None)
    
    print(f"\n" + "="*80)
    print(f"SUMMARY:")
    print(f"  Input: {noisy_features.shape} -> Output: {attended_features.shape}")
    print(f"  Each of the 65536 positions looked at ALL other 65536 positions")
    print(f"  and computed a weighted combination based on similarity")
    print("="*80)