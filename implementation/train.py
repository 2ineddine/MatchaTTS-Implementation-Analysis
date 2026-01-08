# Training loop

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os

#  modules
from config import (
    #paths
    SEED, TRAIN_SPLIT_PATH, VALID_SPLIT_PATH, LJSPEECH_WAV_PATH,
    #model_training params
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, GRAD_CLIP, SAVE_DIR, LOG_INTERVAL,
    #model params
    encoder_params, duration_predictor_params, decoder_params, cfm_params, data_stats, audio_config
    )


from data import LJSpeechDataset, matcha_collate_fn
from matchatts import MatchaTTS
from utils import symbols

# Set seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



os.makedirs(SAVE_DIR, exist_ok=True)

# ====================
# 1. CREATE DATASETS
# ====================
print("Loading datasets...")
train_dataset = LJSpeechDataset(TRAIN_SPLIT_PATH, LJSPEECH_WAV_PATH)
valid_dataset = LJSpeechDataset(VALID_SPLIT_PATH, LJSPEECH_WAV_PATH)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=matcha_collate_fn,
    num_workers=4,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=matcha_collate_fn,
    num_workers=2,
    pin_memory=True
)

print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")

# ====================
# 2. CREATE MODEL
# ====================
print("Initializing model...")
model = MatchaTTS(
    n_vocab=len(symbols),
    n_feats=encoder_params.n_feats,
    encoder_params=encoder_params,
    duration_predictor_params=duration_predictor_params,
    decoder_params=decoder_params,
    cfm_params=cfm_params,
    out_size=None,  # Use full sequences
    mel_mean=data_stats.mel_mean,
    mel_std=data_stats.mel_std,
    prior_loss=True
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ====================
# 3. OPTIMIZER
# ====================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.01)

# Learning rate scheduler (optional)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# ====================
# 4. TRAINING LOOP
# ====================
def train_epoch(epoch):
    model.train()
    total_loss = 0
    total_dur_loss = 0
    total_prior_loss = 0
    total_diff_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        x = batch["x"].to(device)
        x_lengths = batch["x_lengths"].to(device)
        y = batch["y"].to(device)
        y_lengths = batch["y_lengths"].to(device)
        
        # Forward pass
        dur_loss, prior_loss, diff_loss, attn = model(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,

            cond=None
        )
        
        # Total loss
        loss = dur_loss + prior_loss + diff_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_dur_loss += dur_loss.item()
        total_prior_loss += prior_loss.item() if isinstance(prior_loss, torch.Tensor) else prior_loss
        total_diff_loss += diff_loss.item()
        
        # Log
        if batch_idx % LOG_INTERVAL == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Dur: {dur_loss.item():.4f} | "
                  f"Prior: {prior_loss.item() if isinstance(prior_loss, torch.Tensor) else prior_loss:.4f} | "
                  f"Diff: {diff_loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    avg_dur = total_dur_loss / len(train_loader)
    avg_prior = total_prior_loss / len(train_loader)
    avg_diff = total_diff_loss / len(train_loader)
    
    return avg_loss, avg_dur, avg_prior, avg_diff

def validate():
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            x = batch["x"].to(device)
            x_lengths = batch["x_lengths"].to(device)
            y = batch["y"].to(device)
            y_lengths = batch["y_lengths"].to(device)
            
            dur_loss, prior_loss, diff_loss, _ = model(
                x=x,
                x_lengths=x_lengths,
                y=y,
                y_lengths=y_lengths,
                cond=None
            )
            
            loss = dur_loss + prior_loss + diff_loss
            total_loss += loss.item()
    
    return total_loss / len(valid_loader)

# ====================
# 5. MAIN TRAINING
# ====================
print("\nStarting training...")
best_val_loss = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    print(f"{'='*50}")
    
    # Train
    train_loss, train_dur, train_prior, train_diff = train_epoch(epoch)
    
    # Validate
    val_loss = validate()
    
    # Step scheduler
    scheduler.step()
    
    # Log epoch results
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {train_loss:.4f} (Dur: {train_dur:.4f}, Prior: {train_prior:.4f}, Diff: {train_diff:.4f})")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    # Save latest
    torch.save(checkpoint, Path(SAVE_DIR) / "latest.pt")
    
    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(checkpoint, Path(SAVE_DIR) / "best.pt")
        print(f"  ✓ New best model saved! (Val loss: {val_loss:.4f})")
    
    # Save periodic checkpoints
    if epoch % 10 == 0:
        torch.save(checkpoint, Path(SAVE_DIR) / f"epoch_{epoch}.pt")

print("\nTraining completed!")