import random
from pathlib import Path

# Configuration
from config import SEED, LJSPEECH_METADATA_PATH, SPLIT_PATH
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

if __name__ == "__main__":
    main()