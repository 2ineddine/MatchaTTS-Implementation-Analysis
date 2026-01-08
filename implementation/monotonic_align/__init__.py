__author__ = "Mouad"
"""
Monotonic Alignment Search (MAS) for Matcha-TTS.
This module acts as a bridge between PyTorch (GPU) and the optimized Cython implementation (CPU).
"""

import numpy as np
import torch
from .core import maximum_path_c


def maximum_path(log_probs, mask):
    """
    Finds the optimal monotonic alignment path between text phonemes and audio frames.

    This function bridges PyTorch and Cython. It prepares the data by converting
    tensors to numpy arrays and calculating real sequence lengths from the 2D mask.

    Arguments:
        log_probs: Tensor of precomputed log-probabilities (similarity matrix).
                   Shape: [batch, text_len, mel_len]
        mask:      Binary mask indicating valid regions (1) and padding (0).
                   Shape: [batch, text_len, mel_len]

    Returns:
        alignment_path: Binary tensor representing the optimal path (1=aligned, 0=not).
                        Shape: [batch, text_len, mel_len]
    """

    # 1. Apply mask to log_probs to ensure padding regions have zero influence
    # Shape: [Batch, Text_Length, Mel_Length]
    log_probs = log_probs * mask

    # 2. Save device and dtype to restore the result to GPU later
    device = log_probs.device
    dtype = log_probs.dtype

    # 3. Move data to CPU and convert to Numpy (Cython engine requires float32 on CPU)
    # Shape: [Batch, Text_Length, Mel_Length]
    log_probs_np = log_probs.data.cpu().numpy().astype(np.float32)
    mask_np = mask.data.cpu().numpy()

    # 4. Initialize the result container (Binary Alignment Matrix)
    # The Cython function will fill this matrix in-place with 0s and 1s.
    # Shape: [Batch, Text_Length, Mel_Length]
    alignment_path_np = np.zeros_like(log_probs_np).astype(np.int32)

    # 5. Extract real sequence lengths from the 2D mask
    # The Cython engine needs to know exactly where the valid data stops to avoid processing padding.

    # Summing vertically (axis=1) counts valid phonemes -> Text Lengths
    # Shape: [Batch]
    text_lengths = mask_np.sum(axis=1)[:, 0].astype(np.int32)

    # Summing horizontally (axis=2) counts valid frames -> Audio (Mel) Lengths
    # Shape: [Batch]
    mel_lengths = mask_np.sum(axis=2)[:, 0].astype(np.int32)

    # 6. CALL CYTHON FUNCTION (The Viterbi Algorithm)
    # This runs the heavy dynamic programming calculation on the CPU.
    # It modifies 'alignment_path_np' directly.
    maximum_path_c(alignment_path_np, log_probs_np, text_lengths, mel_lengths)

    # 7. Convert result back to PyTorch and move to original Device (GPU)
    # Shape: [Batch, Text_Length, Mel_Length]
    return torch.from_numpy(alignment_path_np).to(device=device, dtype=dtype)