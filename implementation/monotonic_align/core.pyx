# cython: language_level=3
__author__ = "jaywalnut310"
__comments__="Mouad"
"""
Monotonic Alignment Search (MAS) Core Engine.
This Cython module implements the Viterbi algorithm for finding the optimal path.

THE "COIN GAME" ANALOGY:
------------------------
Imagine a grid where every cell contains a pile of coins (Log-Probabilities).
- Rows (x) represent Text Phonemes.
- Cols (y) represent Audio Time Frames.

GOAL:
Start at the top-left (first phoneme, first frame) and reach the bottom-right
(last phoneme, last frame) collecting the maximum value of coins possible. 

MOVEMENT RULES (MONOTONICITY): taken from Glow TTS

Time always moves forward (to the right). For the text axis, you have only two valid moves:
1. STAY (Move Right): 
   (x, y-1) -> (x, y)
   Meaning: You are still pronouncing the SAME phoneme in the new time frame.

2. TRANSITION (Move Diagonal): 
   (x-1, y-1) -> (x, y)
   Meaning: You have moved to the NEXT phoneme in the new time frame.

FORBIDDEN MOVES:
- Skipping a row (cannot jump over a phoneme).
- Moving backward or strictly vertical (time must advance).
"""

import numpy as np
cimport cython
cimport numpy as np
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each(int[:, ::1] path, float[:, ::1] value, int t_x, int t_y, float max_neg_val) nogil:
    """
    Computes the optimal path for a SINGLE batch item (one sentence).
  
    Arguments:
        path: Output grid to store the binary path (0s and 1s).
        value: Input grid of log-probabilities (The "Coins").
               Modified in-place to store cumulative scores.
        t_x: Real length of text (number of phonemes).
        t_y: Real length of audio (number of frames).
        max_neg_val: -infinity.
    """
    cdef int x
    cdef int y
    cdef float v_prev  # Score coming from the diagonal (New Phoneme)
    cdef float v_cur  # Score coming from the left (Same Phoneme)
    cdef float tmp
    cdef int index = t_x - 1


    # Calculate the max score to reach every cell in the grid.

    for y in range(t_y):

        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):

            # Boundary Condition: Main Diagonal
            if x == y:
                v_cur = max_neg_val
            else:
                # Move 1: Coming from the same phoneme (Left neighbor)
                v_cur = value[x, y - 1]

            # Boundary Condition: First Phoneme
            if x == 0:
                if y == 0:
                    v_prev = 0.  # Start point
                else:
                    v_prev = max_neg_val
            else:
                # Move 2: Coming from the previous phoneme (Diagonal neighbor)
                v_prev = value[x - 1, y - 1]


            # Take the max of staying vs moving, and add the current cell's coins.
            # value[x, y] becomes the cumulative score.
            value[x, y] = max(v_cur, v_prev) + value[x, y]


    # Trace back the optimal path from the end to the start.

    for y in range(t_y - 1, -1, -1):
        # Mark the current cell as part of the optimal path
        path[index, y] = 1

        # Decide where we came from to get here:
        # If the score from the same phoneme (left) is lower than the previous phoneme (diagonal),
        # it means we must have transitioned.
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index = index - 1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(int[:, :, ::1] paths, float[:, :, ::1] values, int[::1] t_xs, int[::1] t_ys,
                          float max_neg_val=-1e9) nogil:
    """
    Batch Wrapper for Parallel Processing.
  
    Arguments:
        paths:  3D Tensor [Batch, Text, Mel] for results.
        values: 3D Tensor [Batch, Text, Mel] of log-probs.
        t_xs:   Array of real text lengths for each item.
        t_ys:   Array of real audio lengths for each item.
    """
    cdef int b = values.shape[0]
    cdef int i


    for i in prange(b, nogil=True):
        maximum_path_each(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)