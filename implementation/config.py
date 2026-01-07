# Constants, Parameters and Hyperparameters



# Random State
SEED = 42



# Paths 
LJSPEECH_WAV_PATH = r"datasets/LJSpeech-1.1/wavs"
LJSPEECH_METADATA_PATH = r"datasets/LJSpeech-1.1/metadata.csv"
SPLIT_PATH = r"datasets/data_files"  # Folder to save the filelists
TRAIN_SPLIT_PATH = r"datasets/data_files/train_filelist.txt"
VALID_SPLIT_PATH = r"datasets/data_files/valid_filelist.txt"
TEST_SPLIT_PATH = r"datasets/data_files/test_filelist.txt"



# Audio config
class AudioConfig:
    sample_rate = 22050
    n_mels = 80
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    f_min = 0.0
    f_max = 8000.0
    power = 1.0
    