# Text processing utilities for Matcha-TTS

from matcha.text import text_to_sequence as _matcha_text_to_sequence
from matcha.text.symbols import symbols

# Vocabulary size
N_VOCAB = len(symbols)

# Process text: convert to phonemes and add blank tokens
def process_text(text, cleaner_names=['english_cleaners2']):
    # Convert text to phoneme IDs using matcha cleaners
    sequence, clean_text = _matcha_text_to_sequence(text, cleaner_names)

    # Add blank tokens between phonemes for better alignment
    result = [0] * (len(sequence) * 2 + 1)
    result[1::2] = sequence

    return result
