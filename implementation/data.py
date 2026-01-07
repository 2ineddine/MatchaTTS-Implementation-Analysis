# Dataset loading and text processing



import numpy as np
import torch

from matcha.text import text_to_sequence
from matcha.utils.utils import intersperse



# Text processing
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



# DataFeeder





if __name__ == "__main__":

    # device
    import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
    if torch.cuda.is_available() :
        print(torch.cuda.get_device_name(0))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {DEVICE}")

    # Test text processing

    text = "Hello World."
    output = process_text(text, DEVICE)
    
    print(f"Input text : {text}")
    print(f"Output sequence : {output['x']}")