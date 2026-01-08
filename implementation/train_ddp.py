import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# modules
from .config import (
    SEED, TRAIN_SPLIT_PATH, VALID_SPLIT_PATH, LJSPEECH_WAV_PATH,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, GRAD_CLIP, SAVE_DIR, LOG_INTERVAL,
    encoder_params, duration_predictor_params, decoder_params, cfm_params, data_stats
)
from .data import LJSpeechDataset, matcha_collate_fn
from .matchatts import MatchaTTS
from .utils import symbols


def setup_ddp():
    """Initialize distributed + set correct GPU for this process."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)

    seed_everything(SEED + dist.get_rank())  # small offset per rank helps determinism issues

    if is_main_process():
        print(f"DDP initialized. World size={dist.get_world_size()}")
        os.makedirs(SAVE_DIR, exist_ok=True)

    # ====================
    # 1. CREATE DATASETS
    # ====================
    if is_main_process():
        print("Loading datasets...")

    train_dataset = LJSpeechDataset(TRAIN_SPLIT_PATH, LJSPEECH_WAV_PATH)
    valid_dataset = LJSpeechDataset(VALID_SPLIT_PATH, LJSPEECH_WAV_PATH)

    USE_DEBUG_DATASET = True
    NUM_DEBUG_SAMPLES = 4096

    if USE_DEBUG_DATASET:
        if is_main_process():
            print(f"\n⚠️  MODE DEBUG ACTIVÉ : On ne garde que {NUM_DEBUG_SAMPLES} exemples !")
        indices_train = torch.arange(min(NUM_DEBUG_SAMPLES, len(train_dataset)))
        train_dataset = torch.utils.data.Subset(train_dataset, indices_train)

        indices_valid = torch.arange(min(16, len(valid_dataset)))
        valid_dataset = torch.utils.data.Subset(valid_dataset, indices_valid)

    # Distributed samplers (IMPORTANT)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False, drop_last=False)

    # Per-process batch size (global batch = BATCH_SIZE * world_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=matcha_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler,
        collate_fn=matcha_collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    if is_main_process():
        print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")

    # ====================
    # 2. CREATE MODEL
    # ====================
    if is_main_process():
        print("Initializing model...")

    model = MatchaTTS(
        n_vocab=len(symbols),
        n_feats=encoder_params.n_feats,
        encoder_params=encoder_params,
        duration_predictor_params=duration_predictor_params,
        decoder_params=decoder_params,
        cfm_params=cfm_params,
        mel_mean=data_stats.mel_mean,
        mel_std=data_stats.mel_std,
        prior_loss=True
    ).to(device)

    # Wrap with DDP
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # set True only if you get "unused params" errors
    )

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # ====================
    # 3. OPTIMIZER
    # ====================
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # ====================
    # Helpers: reduce losses across ranks
    # ====================
    def ddp_mean(x: torch.Tensor) -> float:
        """Average a scalar tensor across all ranks, return python float."""
        x = x.detach()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / dist.get_world_size()
        return x.item()

    # ====================
    # 4. TRAINING LOOP
    # ====================
    def train_epoch(epoch: int):
        model.train()
        train_sampler.set_epoch(epoch)

        total = 0.0
        total_dur = 0.0
        total_prior = 0.0
        total_diff = 0.0

        progress = train_loader
        if is_main_process():
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", unit="batch")

        for batch_idx, batch in enumerate(progress):
            x = batch["x"].to(device, non_blocking=True)
            x_lengths = batch["x_lengths"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            y_lengths = batch["y_lengths"].to(device, non_blocking=True)

            dur_loss, prior_loss, diff_loss, _ = model(
                x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, cond=None
            )

            # In DDP these should already be per-rank scalars; ensure they are scalars
            loss = dur_loss + prior_loss + diff_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # average metrics across ranks for logging
            loss_v = ddp_mean(loss)
            dur_v = ddp_mean(dur_loss)
            prior_v = ddp_mean(prior_loss)
            diff_v = ddp_mean(diff_loss)

            total += loss_v
            total_dur += dur_v
            total_prior += prior_v
            total_diff += diff_v

            if is_main_process() and batch_idx % LOG_INTERVAL == 0:
                progress.set_postfix({
                    "Loss": f"{loss_v:.3f}",
                    "Dur": f"{dur_v:.3f}",
                    "Prior": f"{prior_v:.3f}",
                    "Diff": f"{diff_v:.3f}",
                })

        n = len(train_loader)
        return total / n, total_dur / n, total_prior / n, total_diff / n

    @torch.no_grad()
    def validate():
        model.eval()
        total = 0.0

        for batch in valid_loader:
            x = batch["x"].to(device, non_blocking=True)
            x_lengths = batch["x_lengths"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            y_lengths = batch["y_lengths"].to(device, non_blocking=True)

            dur_loss, prior_loss, diff_loss, _ = model(
                x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, cond=None
            )
            loss = dur_loss + prior_loss + diff_loss
            total += ddp_mean(loss)

        return total / len(valid_loader)

    # ====================
    # 5. MAIN TRAINING
    # ====================
    if is_main_process():
        print("\nStarting training...")

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        if is_main_process():
            print(f"\n{'='*50}\nEpoch {epoch}/{NUM_EPOCHS}\n{'='*50}")

        train_loss, train_dur, train_prior, train_diff = train_epoch(epoch)
        val_loss = validate()
        scheduler.step()

        if is_main_process():
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (Dur: {train_dur:.4f}, Prior: {train_prior:.4f}, Diff: {train_diff:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

            # Save checkpoints (IMPORTANT: model.module is the real model under DDP)
            state_dict = model.module.state_dict()
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            torch.save(checkpoint, Path(SAVE_DIR) / "latest.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, Path(SAVE_DIR) / "best.pt")
                print(f"  ✓ New best model saved! (Val loss: {val_loss:.4f})")

            if epoch % 10 == 0:
                torch.save(checkpoint, Path(SAVE_DIR) / f"epoch_{epoch}.pt")

        # Sync before next epoch (keeps ranks aligned)
        dist.barrier()

    if is_main_process():
        print("\nTraining completed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
