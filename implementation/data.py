# Dataset loading and text processing

__author__ = "Massyl A."



import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from librosa.filters import mel as librosa_mel_fn

from config import AudioConfig, LJSPEECH_WAV_PATH, TRAIN_SPLIT_PATH, data_stats
from utils import intersperse
from text import text_to_sequence

# Global cache for mel basis and hann window (from original Matcha-TTS)
mel_basis = {}
hann_window = {}



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



# Mel extraction (using original Matcha-TTS method with librosa filterbank)
def get_mel(wav_path, mel_transform=None):
    """
    Reads a wav file and converts it to a Log-Mel Spectrogram.
    Uses the same method as original Matcha-TTS (librosa mel filterbank).
    """
    global mel_basis, hann_window

    # 1. Load Audio
    audio, sr = torchaudio.load(wav_path)

    # 2. Resample if needed
    if sr != AudioConfig.sample_rate:
        audio = torchaudio.transforms.Resample(sr, AudioConfig.sample_rate)(audio)

    # 3. Take first channel (mono)
    audio = audio[0]  # Shape: [Time]

    # 4. Create mel filterbank if not cached (using librosa like original)
    fmax_key = f"{AudioConfig.f_max}_{audio.device}"
    if fmax_key not in mel_basis:
        mel = librosa_mel_fn(
            sr=AudioConfig.sample_rate,
            n_fft=AudioConfig.n_fft,
            n_mels=AudioConfig.n_mels,
            fmin=AudioConfig.f_min,
            fmax=AudioConfig.f_max
        )
        mel_basis[fmax_key] = torch.from_numpy(mel).float().to(audio.device)
        hann_window[str(audio.device)] = torch.hann_window(AudioConfig.win_length).to(audio.device)

    # 5. STFT (manual, like original)
    spec = torch.stft(
        audio,
        n_fft=AudioConfig.n_fft,
        hop_length=AudioConfig.hop_length,
        win_length=AudioConfig.win_length,
        window=hann_window[str(audio.device)],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # 6. Magnitude
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)

    # 7. Apply mel filterbank
    spec = torch.matmul(mel_basis[fmax_key], spec)

    # 8. Log compression
    spec = torch.log(torch.clamp(spec, min=1e-5))

    # Shape: [80, Time]
    return spec



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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filename, text, _ = self.items[idx]

        # A. Process Text
        text_tensor = process_text_single(text)

        # B. Process Audio
        wav_path = self.wavs_dir / f"{filename}.wav"
        mel_tensor = get_mel(wav_path)

        # C. Normalize mel using dataset statistics
        # This ensures training mels have mean≈0, std≈1
        mel_tensor = (mel_tensor - data_stats.mel_mean) / data_stats.mel_std

        return {
            "x": text_tensor,      # [Text_Length]
            "y": mel_tensor,       # [80, Audio_Length] - NORMALIZED
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

    # A. Text to sequence test
    text = "If you want, I can give you a ready-to-use VS Code settings.json tuned for PyTorch + dynamic ML projects, so Pylance or Pyright is almost noise-free."
    sequence = text_to_sequence(text, ['english_cleaners2'])
    print(f"Text_to_sequence test : {sequence}")



    # B. Data loader Test

    # Initialize Dataset
    train_dataset = LJSpeechDataset(TRAIN_SPLIT_PATH, LJSPEECH_WAV_PATH)

    # Initialize Loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,           # How many files to process at once
        shuffle=True,            # Shuffle for training
        collate_fn=matcha_collate_fn, # Use our custom padding logic
        num_workers=4,           # Use multi-processing for loading
        pin_memory=True          # Faster transfer to GPU
    )

    # Loop
    for batch in train_loader:
        print(batch['x'].shape) # Should be [32, Max_Text_Len]
        print(batch['y'].shape) # Should be [32, 80, Max_Audio_Len]
        break