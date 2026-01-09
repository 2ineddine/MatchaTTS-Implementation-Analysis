"""
Generate audio samples for survey evaluation
Creates 4 random samples with 4 different configurations each (16 total audio files)
"""

import torch
import torchaudio
import random
from pathlib import Path
import sys
import os

# Change to inference directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add implementation to path
sys.path.insert(0, "../implementation")

from data import process_text_single
from matchatts import MatchaTTS
from text import symbols
from config import encoder_params, duration_predictor_params, decoder_params, cfm_params, data_stats

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUSTOM_CHECKPOINT = "../checkpoints/best.pt"
ORIGINAL_CHECKPOINT = "../models/matcha_ljspeech.ckpt"
VOCODER_PATH = "../vocoders/hifigan_T2_v1"
TEST_FILELIST = "../datasets/data_files/test_filelist.txt"
OUTPUT_DIR = Path("../survey/audio_samples")
SEED = 42

# Configurations to test
CONFIGS = [
    {"temp": 0.667, "steps": 10, "name": "steps10"},
    {"temp": 0.667, "steps": 20, "name": "steps20"},
]

# ============================================================
# LOAD MODELS
# ============================================================
print("="*60)
print("LOADING MODELS")
print("="*60)

# Load Vocoder
print("Loading vocoder...")
sys.path.insert(0, str(Path("../Matcha-TTS").resolve()))

from matcha.hifigan.models import Generator as HiFiGAN
from matcha.hifigan.config import v1
from matcha.hifigan.env import AttrDict

h = AttrDict(v1)
vocoder = HiFiGAN(h).to(DEVICE)
vocoder.load_state_dict(torch.load(VOCODER_PATH, map_location=DEVICE)['generator'])
vocoder.eval()
vocoder.remove_weight_norm()
print("✓ Vocoder loaded")

# Load Custom Model
print("Loading custom model...")
custom_model = MatchaTTS(
    n_vocab=len(symbols),
    n_feats=encoder_params.n_feats,
    encoder_params=encoder_params,
    duration_predictor_params=duration_predictor_params,
    decoder_params=decoder_params,
    cfm_params=cfm_params,
    mel_mean=data_stats.mel_mean,
    mel_std=data_stats.mel_std,
    prior_loss=True
).to(DEVICE)
ckpt = torch.load(CUSTOM_CHECKPOINT, map_location=DEVICE)
custom_model.load_state_dict(ckpt['model_state_dict'])
custom_model.eval()
print(f"✓ Custom model loaded (Epoch {ckpt['epoch']})")

# Load Original Model
print("Loading original model...")
sys.path.insert(0, '../Matcha-TTS')
from matcha.models.matcha_tts import MatchaTTS as OriginalMatchaTTS
import lightning as L

original_model = OriginalMatchaTTS.load_from_checkpoint(
    ORIGINAL_CHECKPOINT,
    map_location=DEVICE
)
original_model.eval()
print("✓ Original model loaded")

print()

# ============================================================
# SELECT RANDOM SAMPLES
# ============================================================
print("="*60)
print("SELECTING SAMPLES")
print("="*60)

random.seed(SEED)
with open(TEST_FILELIST, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Select 4 random samples
selected_samples = random.sample(lines, 4)

print(f"Selected {len(selected_samples)} samples:")
for i, line in enumerate(selected_samples, 1):
    parts = line.strip().split('|')
    filename = parts[0]
    text = parts[1]
    print(f"  {i}. {filename}: '{text[:50]}...'")
print()

# ============================================================
# GENERATE AUDIO
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("GENERATING AUDIO SAMPLES")
print("="*60)

metadata = []

for sample_idx, line in enumerate(selected_samples, 1):
    parts = line.strip().split('|')
    filename = parts[0]
    text = parts[1]

    print(f"\nSample {sample_idx}: {filename}")
    print(f"Text: '{text}'")
    print("-"*60)

    sample_data = {
        'sample_id': sample_idx,
        'filename': filename,
        'text': text,
        'audios': []
    }

    # Prepare input for custom model
    x = process_text_single(text).unsqueeze(0).to(DEVICE)
    x_lengths = torch.LongTensor([x.shape[1]]).to(DEVICE)

    for config in CONFIGS:
        temp = config['temp']
        steps = config['steps']
        config_name = config['name']

        print(f"  Config: temp={temp}, steps={steps}")

        # Generate with Custom Model
        with torch.no_grad():
            output_custom = custom_model.synthesise(
                x, x_lengths,
                n_timesteps=steps,
                temperature=temp
            )
            mel_custom = output_custom['mel']
            wav_custom = vocoder(mel_custom).squeeze().cpu()

        # Generate with Original Model
        with torch.no_grad():
            output_original = original_model.synthesise(
                x, x_lengths,
                n_timesteps=steps,
                temperature=temp
            )
            mel_original = output_original['mel']
            wav_original = vocoder(mel_original).squeeze().cpu()

        # Save custom audio
        custom_filename = f"sample{sample_idx}_custom_{config_name}.wav"
        custom_path = OUTPUT_DIR / custom_filename
        torchaudio.save(custom_path, wav_custom.unsqueeze(0), 22050)

        # Save original audio
        original_filename = f"sample{sample_idx}_original_{config_name}.wav"
        original_path = OUTPUT_DIR / original_filename
        torchaudio.save(original_path, wav_original.unsqueeze(0), 22050)

        sample_data['audios'].append({
            'config': config_name,
            'temp': temp,
            'steps': steps,
            'custom_file': custom_filename,
            'original_file': original_filename
        })

        print(f"    ✓ Generated: {custom_filename} & {original_filename}")

    metadata.append(sample_data)

print()
print("="*60)
print("GENERATION COMPLETE")
print("="*60)
print(f"Total audio files generated: {len(selected_samples) * len(CONFIGS) * 2}")
print(f"Saved to: {OUTPUT_DIR}")
print()

# ============================================================
# SAVE METADATA
# ============================================================
import json

metadata_path = OUTPUT_DIR / "metadata.json"
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

# ============================================================
# PRINT SUMMARY
# ============================================================
print()
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Samples: {len(selected_samples)}")
print(f"Configurations per sample: {len(CONFIGS)}")
print(f"Models: 2 (Custom + Original)")
print(f"Total audio files: {len(selected_samples) * len(CONFIGS) * 2}")
print()
print("Configurations:")
for config in CONFIGS:
    print(f"  - Temperature: {config['temp']}, Steps: {config['steps']}")
print()
print("Next steps:")
print("1. Update survey HTML to use these audio files")
print("2. Update Google Sheets form with new samples")
print("="*60)
