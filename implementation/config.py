# Constants, Parameters and Hyperparameters

from dataclasses import dataclass



# Random State
SEED = 42



# Paths 
LJSPEECH_WAV_PATH = r"datasets/LJSpeech-1.1/wavs"
LJSPEECH_METADATA_PATH = r"datasets/LJSpeech-1.1/metadata.csv"
SPLIT_PATH = r"datasets/data_files"  # Folder to save the filelists
TRAIN_SPLIT_PATH = r"datasets/data_files/train_filelist.txt"
VALID_SPLIT_PATH = r"datasets/data_files/valid_filelist.txt"
TEST_SPLIT_PATH = r"datasets/data_files/test_filelist.txt"


#Model_training parameters

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
GRAD_CLIP = 1.0
SAVE_DIR = "checkpoints"
LOG_INTERVAL = 100




# Audio config
@dataclass
class AudioConfig:
    sample_rate = 22050
    n_mels = 80
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    f_min = 0.0
    f_max = 8000.0
    power = 1.0
    

#  Encoder Config
@dataclass
class EncoderParams:
    encoder_type: str = "RoPE Encoder"
    n_vocab: int = 178
    n_feats: int = 80
    n_channels: int = 192
    filter_channels: int = 768
    filter_channels_dp: int = 256
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    prenet: bool = True

@dataclass
class DurationPredictorParams:
    filter_channels_dp: int = 256  # Same as EncoderParams
    kernel_size: int = 3
    p_dropout: float = 0.1  # Same as EncoderParams

@dataclass
class DataStatistics:
    mel_mean: float = -5.536622
    mel_std: float = 2.116101

@dataclass
class DecoderParams:
    channels: tuple = (256, 256)  # U-Net channels
    dropout: float = 0.05
    attention_head_dim: int = 64
    n_blocks: int = 1
    num_mid_blocks: int = 2
    num_heads: int = 2
    act_fn: str = "snakebeta"

@dataclass
class CFMParams:
    solver: str = "euler"
    sigma_min: float = 1e-4


# Create instances
encoder_params = EncoderParams()
duration_predictor_params = DurationPredictorParams()
audio_config = AudioConfig()
data_stats = DataStatistics()
decoder_params = DecoderParams()
cfm_params = CFMParams()
audio_config = AudioConfig()