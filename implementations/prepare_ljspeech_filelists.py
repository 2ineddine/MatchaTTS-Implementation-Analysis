"""
Prepare LJSpeech filelists for training
Splits the dataset into train/validation sets
"""

import random
from pathlib import Path

# Configuration
METADATA_PATH = "data/LJSpeech-1.1/metadata.csv"
TRAIN_FILELIST = "data/train_filelist.txt"
VALID_FILELIST = "data/valid_filelist.txt"
VALID_RATIO = 0.05  # 5% for validation
SEED = 42

def main():
    random.seed(SEED)

    # Read metadata
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse metadata: filename|original_text|normalized_text
    data = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) == 3:
            filename, _, normalized_text = parts
            # Format: wavs/filename.wav|text|speaker_id
            audio_path = f"wavs/{filename}.wav"
            speaker_id = "0"  # LJSpeech is single speaker
            data.append(f"{audio_path}|{normalized_text}|{speaker_id}")

    print(f"Total samples: {len(data)}")

    # Shuffle and split
    random.shuffle(data)
    n_valid = int(len(data) * VALID_RATIO)
    valid_data = data[:n_valid]
    train_data = data[n_valid:]

    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")

    # Create data directory if needed
    Path("data").mkdir(exist_ok=True)

    # Write train filelist
    with open(TRAIN_FILELIST, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_data))

    # Write valid filelist
    with open(VALID_FILELIST, 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_data))

    print(f"\nFilelists created:")
    print(f"  {TRAIN_FILELIST}")
    print(f"  {VALID_FILELIST}")

    # Show examples
    print("\nExample entries:")
    print(f"Train: {train_data[0]}")
    print(f"Valid: {valid_data[0]}")

if __name__ == "__main__":
    main()
