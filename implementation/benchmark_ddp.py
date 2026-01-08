import os, time
from pathlib import Path
import itertools

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .config import (
    SEED, TRAIN_SPLIT_PATH, LJSPEECH_WAV_PATH,
    BATCH_SIZE,
    encoder_params, duration_predictor_params, decoder_params, cfm_params, data_stats
)
from .data import LJSpeechDataset, matcha_collate_fn
from .matchatts import MatchaTTS
from .utils import symbols


# -------------------------
# DDP helpers
# -------------------------
def ddp_init():
    """If launched with torchrun, init DDP; otherwise run single process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank
    else:
        return False, 0

def ddp_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def ddp_world():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def is_main():
    return ddp_rank() == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# Benchmark settings
# -------------------------
USE_DEBUG_DATASET = True
NUM_DEBUG_SAMPLES = 1024

WARMUP_STEPS = 5
MEASURE_STEPS = 30  # augmente à 100 si tu veux une moyenne plus stable


def main():
    ddp_on, local_rank = ddp_init()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(SEED + ddp_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + ddp_rank())

    if is_main():
        print(f"DDP: {ddp_on}, world_size={ddp_world()}, device={device}")
        print(f"Per-GPU batch_size = {BATCH_SIZE}  => Global batch = {BATCH_SIZE * ddp_world()}")

    # -------------------------
    # Data
    # -------------------------
    dataset = LJSpeechDataset(TRAIN_SPLIT_PATH, LJSPEECH_WAV_PATH)
    if USE_DEBUG_DATASET:
        dataset = torch.utils.data.Subset(dataset, torch.arange(min(NUM_DEBUG_SAMPLES, len(dataset))))

    if ddp_on:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=matcha_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # -------------------------
    # Model
    # -------------------------
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

    if ddp_on:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # -------------------------
    # Benchmark loop (train-like)
    # -------------------------
    model.train()
    if ddp_on:
        sampler.set_epoch(0)

    global_batch = BATCH_SIZE * ddp_world()

    it = itertools.cycle(loader)

    # Warmup
    for step in range(WARMUP_STEPS):
        batch = next(it)
        x = batch["x"].to(device, non_blocking=True)
        x_lengths = batch["x_lengths"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        y_lengths = batch["y_lengths"].to(device, non_blocking=True)

        dur_loss, prior_loss, diff_loss, _ = model(x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, cond=None)
        loss = dur_loss + prior_loss + diff_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure
    samples = 0
    frames = 0  # optionnel: compter les frames mel (sum(y_lengths))
    t0 = time.time()

    for step in range(MEASURE_STEPS):
        batch = next(it)
        x = batch["x"].to(device, non_blocking=True)
        x_lengths = batch["x_lengths"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        y_lengths = batch["y_lengths"].to(device, non_blocking=True)

        dur_loss, prior_loss, diff_loss, _ = model(x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, cond=None)
        loss = dur_loss + prior_loss + diff_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        samples += global_batch
        # frames par rank = sum(y_lengths), mais pour un vrai frames/sec global il faudrait all_reduce
        frames += int(y_lengths.sum().item())

    barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0

    # Gather dt? (on prend dt du rank0, suffisant pour comparer)
    if is_main():
        print("\n===== BENCH RESULTS =====")
        print(f"Warmup steps: {WARMUP_STEPS}, Measure steps: {MEASURE_STEPS}")
        print(f"Time: {dt:.3f} s")
        print(f"Throughput: {samples/dt:.2f} samples/s (global)")
        print(f"Approx: {frames/dt:.2f} mel-frames/s (rank0 only, rough)")
        print("=========================\n")

    cleanup()


if __name__ == "__main__":
    main()
