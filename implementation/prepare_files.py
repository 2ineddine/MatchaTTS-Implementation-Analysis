# Create and save the train Test Split for LJSpeech Dataset

__author__ = "Massyl A."



import random
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Configuration
from config import SEED, LJSPEECH_METADATA_PATH, SPLIT_PATH, LJSPEECH_WAV_PATH, data_stats
VALID_COUNT = 100
TEST_COUNT = 100



def main():
    random.seed(SEED)
    Path(SPLIT_PATH).mkdir(exist_ok=True)

    # 1. Read Metadata
    with open(LJSPEECH_METADATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2. Parse
    data = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) == 3:
            filename, _, normalized_text = parts
            # We save: wav_filename | text | speaker_id
            data.append(f"{filename}|{normalized_text}|0")

    print(f"Total samples found: {len(data)}")

    # 3. Shuffle and Split
    random.shuffle(data)
    
    # Extract Test and Valid sets from the front
    test_data = data[:TEST_COUNT]
    remaining = data[TEST_COUNT:]
    
    valid_data = remaining[:VALID_COUNT]
    train_data = remaining[VALID_COUNT:]

    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Test samples:  {len(test_data)}")

    # 4. Save
    def save_list(filename, data_list):
        path = Path(SPLIT_PATH) / filename
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data_list))
        print(f"Saved {path}")

    save_list("train_filelist.txt", train_data)
    save_list("valid_filelist.txt", valid_data)
    save_list("test_filelist.txt", test_data)

    # 5. Precompute Mels
    print("\n" + "="*60)
    print("PRECOMPUTING MELS...")
    print("="*60)

    from data import get_mel

    wavs_dir = Path(LJSPEECH_WAV_PATH)
    mels_dir = Path("datasets/LJSpeech-1.1/mels")
    mels_dir.mkdir(exist_ok=True)

    wav_files = sorted(wavs_dir.glob("*.wav"))
    print(f"Extracting mels from {len(wav_files)} files...")
    print(f"Output: {mels_dir}")
    print()

    for wav_path in tqdm(wav_files, desc="Extracting mels"):
        mel = get_mel(wav_path)
        mel_normalized = (mel - data_stats.mel_mean) / data_stats.mel_std
        mel_path = mels_dir / f"{wav_path.stem}.npy"
        np.save(mel_path, mel_normalized.numpy())

    print()
    print("="*60)
    print("✓ DONE! Dataset preparation complete.")
    print(f"  Filelists: {SPLIT_PATH}")
    print(f"  Precomputed mels: {mels_dir}")
    print()
    print("You can now start training with: python train.py")
    print("="*60)

if __name__ == "__main__":
    main()