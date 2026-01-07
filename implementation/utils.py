# Helper functions (MAS alignment, common layers)

import torch




def sequence_mask(length, max_length=None):
    """
    Creates a boolean mask to distinguish real data from padding in batched sequences.
    Author: Mouad
    Example: lengths=[5, 3, 7] produces a [3, 7] mask where True=valid, False=padding

    row 0: [0<5, 1<5, 2<5, 3<5, 4<5, 5<5, 6<5] = [T, T, T, T, T, F, F]
    row 1: [0<3, 1<3, 2<3, 3<3, 4<3, 5<3, 6<3] = [T, T, T, F, F, F, F]
    row 2: [0<7, 1<7, 2<7, 3<7, 4<7, 5<7, 6<7] = [T, T, T, T, T, T, T]
    """
    # Step 1: Determine the maximum length if not provided
    if max_length is None:
        max_length = length.max()

    # Step 2: Create position indices [0, 1, 2, 3, ..., max_length-1]
    # This represents "what column am I in?"
    positions = torch.arange(max_length, dtype=length.dtype, device=length.device)

    # Step 3: Reshape position indices into a row vector
    # From shape [max_length] to [1, max_length]
    # Example: [0,1,2,3,4,5,6] becomes [[0,1,2,3,4,5,6]]
    positions_row = positions.unsqueeze(0)

    # Step 4: Reshape lengths into a column vector
    # From shape [batch_size] to [batch_size, 1]
    # Example: [5,3,7] becomes [[5], [3], [7]]
    lengths_column = length.unsqueeze(1)

    # Step 5: Compare using broadcasting
    # PyTorch expands both to [batch_size, max_length] and compares element-wise
    # For each sequence: is this position < the sequence's actual length?
    mask = positions_row < lengths_column

    return mask


def duration_loss(logw, logw_, lengths):
    """
    Computes MSE loss for duration, normalized by the total number of valid frames.
    Author : Mouad
    Args:
        logw: Predicted log durations
        logw_: Target log durations
        lengths: Valid lengths of sequences
    """
    # Note: We assume logw and logw_ are masked (zeroed) outside valid regions
    # We divide by sum(lengths) which is the total count of valid phonemes in the batch.
    # we can see that when pad loss=0, and not counted on the normalisation by sum(lengths)
    #because we compute n losses, n valid pos ,and sum(lengths)=n
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss


def convert_pad_shape(pad_shape):
    """
    Helper for generate_path: Convert padding format for PyTorch's F.pad.

    Author: Mouad


    We usually think of padding as: [[Pad_Batch], [Pad_Height], [Pad_Width]].
    But PyTorch's F.pad function expects dimensions in REVERSE order,
    starting from the last dimension.

    Example Input (What we want):
        [[0, 0], [1, 0], [0, 0]]  -> (No pad on Batch, 1 pad on Top of Text, No pad on Audio)

    Process:
        1. Reverse: [[0, 0], [1, 0], [0, 0]] (Since it's symmetric here, looks same)
        2. Flatten: [0, 0, 1, 0, 0, 0]

    Output (What F.pad wants):
        [Left, Right, Top, Bottom, Front, Back]
    """
    # 1. Reverse the order of dimensions (Last dim comes first for F.pad)
    inverted_shape = pad_shape[::-1]

    # 2. Flatten the list of lists into a single list
    # [[A, B], [C, D]] -> [A, B, C, D]
    pad_shape = [item for sublist in inverted_shape for item in sublist]

    return pad_shape


def generate_path(duration, mask):
    """
    Generates the alignment path from durations using fast vectorization.
    Used during inference to convert predicted durations into an alignment matrix.

    Author: Mouad


    We want to turn durations [2, 3] into a matrix where:
    - Row 0 has two 1s.
    - Row 1 has three 1s (shifted to start after Row 0).

    1. CUMSUM: Calculate end times. [2, 3] -> [2, 5].
    2. STAIRCASE: Use sequence_mask to create cumulative blocks.
       Row 0 (Ends at 2): [1, 1, 0, 0, 0]
       Row 1 (Ends at 5): [1, 1, 1, 1, 1]
    3. SUBTRACTION: Retrieve individual durations by subtracting the previous row.
       Row 1 (Pure) = Row 1 (Cumulative) - Row 0 (Cumulative)
       [0, 0, 1, 1, 1] = [1, 1, 1, 1, 1] - [1, 1, 0, 0, 0]

    Args:
        duration: Int tensor [b, text_len] (How many frames per phoneme)
        mask: Binary mask [b, text_len, mel_len] (The valid rectangle)

    Returns:
        path: Binary alignment matrix [b, text_len, mel_len]
    """
    device = duration.device
    b, t_x, t_y = mask.shape

    # 1. Calculate cumulative duration (The "End Time" of each phoneme)
    # Ex: [2, 3, 1] -> [2, 5, 6]
    cum_duration = torch.cumsum(duration, 1)

    # 2. Flatten to use sequence_mask trick
    cum_duration_flat = cum_duration.view(b * t_x)

    # 3. Create the "Staircase" (Cumulative Masks)
    # Using the end times, we create rows of 1s starting from 0.
    # Row 0: 1 1 0 0 0 0 (Ends at 2)
    # Row 1: 1 1 1 1 1 0 (Ends at 5)
    # Row 2: 1 1 1 1 1 1 (Ends at 6)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)

    # 4. Reshape back to batch dimensions [Batch, Text, Mel]
    path = path.view(b, t_x, t_y)

    # 5. The "Difference" Trick (Isolation)
    # We shift the path down by 1 pixel along the text axis.
    # pad_shape arguments: [[Batch], [Text], [Mel]] -> We pad [1, 0] on Text.
    # This effectively moves "Row i-1" to position "Row i".
    path_shifted = F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]

    # Subtracting the shifted version removes the "history" of previous phonemes.
    # Current Row - Previous Row = Only the frames belonging to Current Phoneme.
    path = path - path_shifted

    # 6. Apply the global mask to clean up any padding artifacts
    path = path * mask

    return path


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    """
    Adjusts length to be compatible with U-Net downsampling.
    Author : Mouad
    Args:
        length: Sequence length
        num_downsamplings_in_unet: Number of downsampling layers

    Returns:
        Adjusted length (int)
    """
    factor = 2 ** num_downsamplings_in_unet
    adjusted_length = int(torch.ceil(torch.tensor(length) / factor) * factor)
    return adjusted_length








