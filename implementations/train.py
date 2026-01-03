"""
Matcha-TTS Training Script
Complete training pipeline with data loading, model training, and checkpointing
"""

import os
import sys
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from tqdm import tqdm

# Change to project root if running from implementations folder
if Path.cwd().name == 'implementations':
    os.chdir('..')
    sys.path.insert(0, str(Path.cwd() / 'implementations'))

from encoder import TextEncoder
from decoder import ConditionalFlowMatching
import text_utils

# ============================================================================
# TRAINING CONFIGURATION - Edit these parameters
# ============================================================================

# Data paths
TRAIN_FILELIST = "../LJSpeech-1.1/train.txt"
VALID_FILELIST = "../LJSpeech-1.1/val.txt"
DATA_DIR = "../LJSpeech-1.1"

# Model configuration
N_VOCAB = text_utils.N_VOCAB     # Vocabulary size (from text_utils)
N_MEL_CHANNELS = 80        # Mel spectrogram channels
ENCODER_CHANNELS = 192     # Encoder hidden size
FILTER_CHANNELS = 768      # FFN hidden size
N_HEADS = 2                # Attention heads
N_LAYERS = 6               # Encoder layers
DECODER_CHANNELS = [256, 256]  # Decoder hidden channels
N_SPEAKERS = 1             # Set > 1 for multi-speaker
SPEAKER_EMB_DIM = 64       # Speaker embedding dimension

# Audio processing
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
F_MIN = 0
F_MAX = 8000

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
GRAD_CLIP = 1.0
WARMUP_STEPS = 4000
SAVE_INTERVAL = 5000       # Save checkpoint every N steps

# Loss weights
DURATION_LOSS_WEIGHT = 1.0
PRIOR_LOSS_WEIGHT = 1.0
CFM_LOSS_WEIGHT = 1.0

# Misc
NUM_WORKERS = 4
CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ============================================================================


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_filelist(filelist_path, split_char="|"):
    """Parse filelist: audio_path|text|speaker_id"""
    with open(filelist_path, encoding="utf-8") as f:
        lines = [line.strip().split(split_char) for line in f]
    return lines


# Note: text_to_sequence and intersperse are now imported from text_utils


def compute_mel_spectrogram(audio, n_fft, hop_length, win_length, f_min, f_max, n_mels):
    """Compute mel spectrogram from audio"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=1.0,  # Energy mel spectrogram
    )

    mel = mel_transform(audio)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel


class TextMelDataset(Dataset):
    """Dataset for loading audio and text pairs"""
    def __init__(self, filelist_path, data_dir, n_mels=80):
        self.data_dir = Path(data_dir)
        self.n_mels = n_mels
        self.data = parse_filelist(filelist_path)

        # Compute mel statistics for normalization
        self.mel_mean = None
        self.mel_std = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        parts = self.data[idx]
        
        # Handle both formats: "path|text|spk_id" or "path|text"
        if len(parts) >= 3:
            audio_path, text, spk_id = parts[0], parts[1], parts[2]
            spk_id = int(spk_id) if N_SPEAKERS > 1 else None
        else:
            audio_path, text = parts[0], parts[1]
            spk_id = None
    
        # Load audio
        audio, sr = torchaudio.load(self.data_dir / audio_path)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio[0]  # Take first channel
    
        # Compute mel spectrogram
        mel = compute_mel_spectrogram(
            audio, N_FFT, HOP_LENGTH, WIN_LENGTH, F_MIN, F_MAX, self.n_mels
        )
    
        # Convert text to phoneme sequence with blanks
        phonemes = text_utils.process_text(text)
    
        return {
            'phonemes': torch.LongTensor(phonemes),
            'mel': mel.squeeze(0),
            'spk_id': spk_id,
        }


def collate_fn(batch):
    """Collate batch with padding"""
    # Get lengths
    phoneme_lengths = torch.LongTensor([item['phonemes'].size(0) for item in batch])
    mel_lengths = torch.LongTensor([item['mel'].size(1) for item in batch])

    # Pad sequences
    phonemes_padded = pad_sequence(
        [item['phonemes'] for item in batch],
        batch_first=True,
        padding_value=0
    )

    mels_padded = pad_sequence(
        [item['mel'].transpose(0, 1) for item in batch],
        batch_first=True,
        padding_value=0
    ).transpose(1, 2)

    # Speaker IDs
    if N_SPEAKERS > 1:
        spk_ids = torch.LongTensor([item['spk_id'] for item in batch])
    else:
        spk_ids = None

    return {
        'phonemes': phonemes_padded,
        'phoneme_lengths': phoneme_lengths,
        'mels': mels_padded,
        'mel_lengths': mel_lengths,
        'spk_ids': spk_ids,
    }


def sequence_mask(length, max_length=None):
    """Create sequence mask"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(durations, mask):
    """
    Generate alignment path from durations
    Uses Monotonic Alignment Search in actual implementation
    """
    batch_size, text_len, mel_len = mask.shape
    path = torch.zeros(batch_size, text_len, mel_len, dtype=mask.dtype, device=mask.device)

    for b in range(batch_size):
        mel_pos = 0
        for t in range(text_len):
            duration = int(durations[b, t].item())
            for _ in range(duration):
                if mel_pos < mel_len:
                    path[b, t, mel_pos] = 1
                    mel_pos += 1

    return path


def duration_loss(logw_predicted, logw_target, x_lengths):
    """Compute duration prediction loss"""
    loss = torch.sum((logw_predicted - logw_target) ** 2) / torch.sum(x_lengths)
    return loss


class MatchaTTS(nn.Module):
    """
    Complete Matcha-TTS Model

    Combines:
    - Text Encoder (phonemes -> mel features + durations)
    - Conditional Flow Matching Decoder (generates mel spectrograms)
    """
    def __init__(
        self,
        n_vocab=N_VOCAB,
        n_mel_channels=N_MEL_CHANNELS,
        n_spks=N_SPEAKERS,
        spk_emb_dim=SPEAKER_EMB_DIM,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_spks = n_spks

        # Speaker embedding (if multi-speaker)
        if n_spks > 1:
            self.spk_emb = nn.Embedding(n_spks, spk_emb_dim)

        # Text Encoder
        self.encoder = TextEncoder(
            n_vocab=n_vocab,
            n_mel_channels=n_mel_channels,
            encoder_channels=ENCODER_CHANNELS,
            filter_channels=FILTER_CHANNELS,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        # Decoder (CFM)
        self.decoder = ConditionalFlowMatching(
            in_channels=n_mel_channels,
            out_channels=n_mel_channels,
            channels=DECODER_CHANNELS,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        # Data statistics for normalization
        self.register_buffer('mel_mean', torch.zeros(n_mel_channels))
        self.register_buffer('mel_std', torch.ones(n_mel_channels))

    def forward(self, x, x_lengths, y, y_lengths, spks=None):
        """
        Training forward pass

        Returns:
            dur_loss: Duration prediction loss
            prior_loss: Prior loss (encoder MSE with target)
            cfm_loss: Flow matching loss
        """
        # Get speaker embeddings
        if self.n_spks > 1 and spks is not None:
            spks = self.spk_emb(spks)

        # Encode text
        mu, logw, x_mask = self.encoder(x, x_lengths, spks)

        # Create masks
        y_mask = sequence_mask(y_lengths, y.size(2)).unsqueeze(1).to(x_mask.dtype)

        # For simplicity, use uniform durations (in practice, use MAS)
        # Monotonic Alignment Search finds optimal alignment
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w)

        # Simple alignment (replace with MAS in production)
        attn = torch.zeros(x.size(0), x.size(1), y.size(2), device=x.device)
        logw_target = torch.log(w + 1e-8) * x_mask

        # Duration loss
        dur_loss = duration_loss(logw, logw_target, x_lengths)

        # Align encoder output with mel
        mu_expanded = torch.matmul(mu, attn)

        # Prior loss (MSE between encoder output and target mel)
        prior_loss = F.mse_loss(mu_expanded * y_mask, y * y_mask)

        # Flow matching loss
        cfm_loss = self.decoder.compute_loss(y, y_mask, mu_expanded, spks)

        return dur_loss, prior_loss, cfm_loss

    @torch.no_grad()
    def synthesize(self, x, x_lengths, spks=None, length_scale=1.0, temperature=1.0, n_timesteps=10):
        """
        Inference: Generate mel spectrogram from text

        Args:
            x: Phoneme sequence [batch, seq_len]
            x_lengths: Sequence lengths [batch]
            spks: Speaker IDs [batch]
            length_scale: Duration scaling (>1 = slower, <1 = faster)
            temperature: Sampling temperature
            n_timesteps: Number of ODE steps

        Returns:
            mel: Generated mel spectrogram
        """
        # Get speaker embeddings
        if self.n_spks > 1 and spks is not None:
            spks = self.spk_emb(spks)

        # Encode text
        mu, logw, x_mask = self.encoder(x, x_lengths, spks)

        # Predict durations
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w * length_scale)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()

        # Expand encoder output to mel length
        y_max_length = int(y_lengths.max())
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)

        # Simple expansion (replace with proper alignment in production)
        mu_expanded = mu.repeat_interleave(int(length_scale), dim=2)[:, :, :y_max_length]

        # Generate mel using flow matching
        mel = self.decoder(mu_expanded, y_mask, n_timesteps, temperature, spks)

        # Denormalize
        mel = mel * self.mel_std[:, None] + self.mel_mean[:, None]

        return mel, y_lengths


def get_lr_schedule(step, warmup_steps=WARMUP_STEPS, d_model=ENCODER_CHANNELS):
    """Learning rate schedule with warmup"""
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dur_loss = 0
    total_prior_loss = 0
    total_cfm_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        phonemes = batch['phonemes'].to(device)
        phoneme_lengths = batch['phoneme_lengths'].to(device)
        mels = batch['mels'].to(device)
        mel_lengths = batch['mel_lengths'].to(device)
        spk_ids = batch['spk_ids'].to(device) if batch['spk_ids'] is not None else None

        # Forward pass
        dur_loss, prior_loss, cfm_loss = model(
            phonemes, phoneme_lengths, mels, mel_lengths, spk_ids
        )

        # Total loss
        loss = (
            DURATION_LOSS_WEIGHT * dur_loss +
            PRIOR_LOSS_WEIGHT * prior_loss +
            CFM_LOSS_WEIGHT * cfm_loss
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # Update learning rate
        global_step = epoch * len(dataloader) + batch_idx
        lr = get_lr_schedule(global_step) * LEARNING_RATE
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_dur_loss += dur_loss.item()
        total_prior_loss += prior_loss.item()
        total_cfm_loss += cfm_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dur': f'{dur_loss.item():.4f}',
            'prior': f'{prior_loss.item():.4f}',
            'cfm': f'{cfm_loss.item():.4f}',
            'lr': f'{lr:.6f}'
        })

    return {
        'loss': total_loss / len(dataloader),
        'dur_loss': total_dur_loss / len(dataloader),
        'prior_loss': total_prior_loss / len(dataloader),
        'cfm_loss': total_cfm_loss / len(dataloader),
    }


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            phonemes = batch['phonemes'].to(device)
            phoneme_lengths = batch['phoneme_lengths'].to(device)
            mels = batch['mels'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            spk_ids = batch['spk_ids'].to(device) if batch['spk_ids'] is not None else None

            dur_loss, prior_loss, cfm_loss = model(
                phonemes, phoneme_lengths, mels, mel_lengths, spk_ids
            )

            loss = (
                DURATION_LOSS_WEIGHT * dur_loss +
                PRIOR_LOSS_WEIGHT * prior_loss +
                CFM_LOSS_WEIGHT * cfm_loss
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    """Main training loop"""
    print(f"Using device: {DEVICE}")
    set_seed(SEED)

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_dataset = TextMelDataset(TRAIN_FILELIST, DATA_DIR, N_MEL_CHANNELS)
    valid_dataset = TextMelDataset(VALID_FILELIST, DATA_DIR, N_MEL_CHANNELS)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Create model
    print("Creating model...")
    model = MatchaTTS().to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE, epoch)

        # Validate
        val_loss = validate(model, valid_loader, DEVICE)

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")

        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)


if __name__ == "__main__":
    main()
