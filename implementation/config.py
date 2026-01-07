# Constants, Parameters and Hyperparameters

# Random State
SEED = 42

# Paths 
LJSPEECH_METADATA_PATH = r"datasets/LJSpeech-1.1/metadata.csv"
SPLIT_PATH = r"datasets/data_files"  # Folder to save the filelists



# Audio config
class AudioConfig:
    sample_rate = 22050
    n_mels = 80
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    f_min = 0.0
    f_max = 8000.0
    