"""
Matcha-TTS Encoder
Converts phoneme tokens to mel features and predicts durations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ============================================================================
# ENCODER CONFIGURATION - Edit these parameters
# ============================================================================
N_VOCAB = 178              # Vocabulary size (number of phonemes)
N_MEL_CHANNELS = 80        # Number of mel spectrogram channels
ENCODER_CHANNELS = 192     # Hidden dimension size
FILTER_CHANNELS = 768      # FFN hidden dimension
N_HEADS = 2                # Number of attention heads
N_LAYERS = 6               # Number of transformer layers
KERNEL_SIZE = 3            # Convolution kernel size
DROPOUT = 0.1              # Dropout probability
USE_PRENET = True          # Use prenet before encoder

# Duration Predictor
DURATION_FILTER_CHANNELS = 256
DURATION_KERNEL_SIZE = 3

# ============================================================================


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvReluNorm(nn.Module):
    """Prenet: Stack of Conv1D + ReLU + LayerNorm layers"""
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))

        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))

        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, d, base=10000):
        super().__init__()
        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        seq_len = x.shape[0]
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
        seq_idx = torch.arange(seq_len, device=x.device).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x):
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x):
        x = rearrange(x, "b h t d -> t b h d")
        self._build_cache(x)

        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

        return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE"""
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.k_channels = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)

        self.query_rotary_pe = RotaryPositionalEmbedding(self.k_channels * 0.5)
        self.key_rotary_pe = RotaryPositionalEmbedding(self.k_channels * 0.5)

        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, _ = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))

        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        query = self.query_rotary_pe(query)
        key = self.key_rotary_pe(key)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)

        return output, p_attn


class FFN(nn.Module):
    """Feed-Forward Network"""
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class TransformerEncoder(nn.Module):
    """Stack of Multi-Head Attention + FFN layers"""
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(len(self.attn_layers)):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class DurationPredictor(nn.Module):
    """Predicts log-duration for each phoneme"""
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    """
    Main Encoder: Phoneme Tokens -> Mel Features + Duration Prediction

    Architecture:
    1. Embedding layer for phoneme tokens
    2. Optional Prenet (Conv-ReLU-Norm layers)
    3. Transformer Encoder (Multi-Head Attention + FFN)
    4. Projection to mel features (mu)
    5. Duration Predictor (predicts phoneme durations)
    """
    def __init__(
        self,
        n_vocab=N_VOCAB,
        n_mel_channels=N_MEL_CHANNELS,
        encoder_channels=ENCODER_CHANNELS,
        filter_channels=FILTER_CHANNELS,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        kernel_size=KERNEL_SIZE,
        p_dropout=DROPOUT,
        use_prenet=USE_PRENET,
        duration_filter_channels=DURATION_FILTER_CHANNELS,
        duration_kernel_size=DURATION_KERNEL_SIZE,
        n_spks=1,
        spk_emb_dim=64,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_mel_channels = n_mel_channels
        self.encoder_channels = encoder_channels
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim

        # Phoneme embedding
        self.emb = nn.Embedding(n_vocab, encoder_channels)
        nn.init.normal_(self.emb.weight, 0.0, encoder_channels ** -0.5)

        # Optional prenet
        if use_prenet:
            self.prenet = ConvReluNorm(
                encoder_channels,
                encoder_channels,
                encoder_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
        else:
            self.prenet = lambda x, x_mask: x

        # Transformer encoder
        self.encoder = TransformerEncoder(
            encoder_channels + (spk_emb_dim if n_spks > 1 else 0),
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )

        # Project to mel features
        self.proj_m = nn.Conv1d(encoder_channels + (spk_emb_dim if n_spks > 1 else 0), n_mel_channels, 1)

        # Duration predictor
        self.proj_w = DurationPredictor(
            encoder_channels + (spk_emb_dim if n_spks > 1 else 0),
            duration_filter_channels,
            duration_kernel_size,
            p_dropout,
        )

    def forward(self, x, x_lengths, spks=None):
        """
        Args:
            x: Phoneme token IDs [batch, max_text_len]
            x_lengths: Text lengths [batch]
            spks: Speaker embeddings [batch, spk_emb_dim] (optional, for multi-speaker)

        Returns:
            mu: Encoder output (mel features) [batch, n_mel_channels, max_text_len]
            logw: Log-duration predictions [batch, 1, max_text_len]
            x_mask: Input mask [batch, 1, max_text_len]
        """
        # Embed phonemes
        x = self.emb(x) * math.sqrt(self.encoder_channels)
        x = torch.transpose(x, 1, -1)  # [batch, encoder_channels, max_text_len]

        # Create mask
        x_mask = self._sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        # Apply prenet
        x = self.prenet(x, x_mask)

        # Add speaker embedding if multi-speaker
        if self.n_spks > 1 and spks is not None:
            x = torch.cat([x, spks.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)

        # Transformer encoding
        x = self.encoder(x, x_mask)

        # Project to mel features
        mu = self.proj_m(x) * x_mask

        # Predict durations (detach to prevent gradient flow)
        x_dp = torch.detach(x)
        logw = self.proj_w(x_dp, x_mask)

        return mu, logw, x_mask

    @staticmethod
    def _sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)


if __name__ == "__main__":
    # Test the encoder
    batch_size = 2
    max_text_len = 50

    encoder = TextEncoder()

    # Create dummy input
    x = torch.randint(0, N_VOCAB, (batch_size, max_text_len))
    x_lengths = torch.tensor([50, 45])

    mu, logw, x_mask = encoder(x, x_lengths)

    print(f"Input shape: {x.shape}")
    print(f"Encoder output (mu) shape: {mu.shape}")
    print(f"Duration predictions (logw) shape: {logw.shape}")
    print(f"Mask shape: {x_mask.shape}")
