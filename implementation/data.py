# Dataset loading and text processing

__author__ = "Massyl A."



import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .config import AudioConfig, LJSPEECH_WAV_PATH, TRAIN_SPLIT_PATH

from matcha.text import text_to_sequence
from matcha.utils.utils import intersperse



# Text processing for inference
@torch.inference_mode()
def process_text(text, device):
    
    # a. Convert text to phoneme IDs
    # 'english_cleaners2' handles number expansion and phonemization via espeak
    sequence = text_to_sequence(text, ['english_cleaners2'])[0]
    
    # b. Intersperse with '0' (blank token) for the flow matching stability
    sequence = intersperse(sequence, 0)
    
    # c. Convert to tensor
    x = torch.tensor(sequence, dtype=torch.long, device=device)[None] # Add batch dim

    # metadatas
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)

    # output
    output = {
        'text': text,
        'x': x,
        'x_lengths': x_lengths
    }
    
    return output



# Text processing for training
def process_text_single(text):
    
    # a. Convert text to phoneme IDs
    # 'english_cleaners2' handles number expansion and phonemization via espeak
    sequence = text_to_sequence(text, ['english_cleaners2'])[0]
    
    # b. Intersperse with '0' (put 0 between every token) for the flow matching stability
    sequence = intersperse(sequence, 0)
    
    # c. Convert to tensor
    x = torch.tensor(sequence, dtype=torch.long)
    
    return x



# Mel transform
def get_mel(wav_path, mel_transform):
    """
    Reads a wav file and converts it to a Log-Mel Spectrogram.
    """
    # 1. Load Audio
    audio, sr = torchaudio.load(wav_path)

    # 2. Resample for ssafety (LJSpeech is usually 22050Hz)
    if sr != AudioConfig.sample_rate:
        audio = torchaudio.transforms.Resample(sr, AudioConfig.sample_rate)(audio)

    # 3. Clip audio (safety against outliers)
    audio = torch.clamp(audio[0], -1.0, 1.0).unsqueeze(0)

    # 4. Convert to Mel Spectrogram
    mel = mel_transform(audio)

    # 5. Log Compression (Dynamic Range Compression)
    # We clamp min to 1e-5 to avoid log(0) = -infinity
    mel = torch.log(torch.clamp(mel, min=1e-5))
    
    # Shape: [Channels(80), Time]
    return mel.squeeze(0)



# Data Loader
class LJSpeechDataset(Dataset):
    def __init__(self, filelist_path, wavs_dir):
        """
        Args:
            filelist_path: Path to '.../train_filelist.txt'
            wavs_dir: Path to 'wavs' folder
        """
        self.wavs_dir = Path(wavs_dir)
        self.items = []

        # Parse the filelist
        # Line format: filename|text|speaker_id
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                self.items.append(parts)

        # initialize mel
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=AudioConfig.sample_rate,
                n_fft=AudioConfig.n_fft,
                win_length=AudioConfig.win_length,
                hop_length=AudioConfig.hop_length,
                f_min=AudioConfig.f_min,
                f_max=AudioConfig.f_max,
                n_mels=AudioConfig.n_mels,
                power=AudioConfig.power,
                center=False
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filename, text, _ = self.items[idx]

        # A. Process Text
        text_tensor = process_text_single(text)

        # B. Process Audio
        wav_path = self.wavs_dir / f"{filename}.wav"
        mel_tensor = get_mel(wav_path, self.mel_transform)

        return {
            "x": text_tensor,      # [Text_Length]
            "y": mel_tensor,       # [80, Audio_Length]
            "name": filename,
            "text": text
        }



# collate function for padding (a batch must have the same size)
def matcha_collate_fn(batch):
    """
    Audio files have different lengths, we must PAD them.
    """
    # 1. Sort batch by audio length (descending)
    # This helps CUDA kernels optimize processing
    batch.sort(key=lambda x: x['y'].shape[1], reverse=True)

    # 2. Extract items
    texts = [item['x'] for item in batch]
    mels = [item['y'] for item in batch]
    filenames = [item['name'] for item in batch]
    
    # 3. Get original lengths (needed for the model masking)
    x_lengths = torch.LongTensor([len(x) for x in texts])
    y_lengths = torch.LongTensor([y.shape[1] for y in mels])

    # 4. Pad Text
    # Pad with 0 (Blank token)
    x_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)

    # 5. Pad Mels
    # pad_sequence expects [Time, Dimensions], but Mels are [80, Time]
    # So we: Permute -> Pad -> Permute back
    mels_permuted = [mel.permute(1, 0) for mel in mels] # [Time, 80]
    
    # Padding value: -11.5 is roughly log(1e-5), representing silence in log-mel domain
    y_padded = torch.nn.utils.rnn.pad_sequence(mels_permuted, batch_first=True, padding_value=-11.5129)
    y_padded = y_padded.permute(0, 2, 1) # Back to [Batch, 80, Time]

    return {
        "x": x_padded,              # Model Input (Text IDs)
        "x_lengths": x_lengths,     # Input Lengths
        "y": y_padded,              # Model Target (Mel Spectrograms)
        "y_lengths": y_lengths,     # Target Lengths
        "names": filenames
    }





if __name__ == "__main__":

    # 2. Initialize Dataset
    train_dataset = LJSpeechDataset(TRAIN_SPLIT_PATH, LJSPEECH_WAV_PATH)

    # 3. Initialize Loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,           # How many files to process at once
        shuffle=True,            # Shuffle for training
        collate_fn=matcha_collate_fn, # Use our custom padding logic
        num_workers=4,           # Use multi-processing for loading
        pin_memory=True          # Faster transfer to GPU
    )

    # 4. Loop
    for batch in train_loader:
        print(batch['x'].shape) # Should be [32, Max_Text_Len]
        print(batch['y'].shape) # Should be [32, 80, Max_Audio_Len]
        break