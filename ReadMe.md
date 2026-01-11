# MatchaTTS Implementation and Analysis

**Authors:**

- Massyl Adjal (Project Lead & Coordinator)

- Yasser Bouhai

- Zineddine Bouhadjira

- Mohamed Mouad Boularas

**Institution:** Sorbonne Université — Master Ingénierie des Systèmes Intelligents

**Repository:** [https://github.com/2ineddine/MatchaTTS-Implementation-Analysis](https://www.google.com/search?q=https://github.com/2ineddine/MatchaTTS-Implementation-Analysis)

---

## Project Overview

This project focuses on the reproducibility and analysis of **Matcha-TTS**, a state-of-the-art non-autoregressive text-to-speech architecture based on **Optimal-Transport Conditional Flow Matching (OT-CFM)**.

Matcha-TTS is designed to synthesize high-quality mel-spectrograms efficiently. Unlike diffusion models that require many iterative steps, Matcha-TTS uses an ODE-based decoder to transform noise into speech representations along a straight trajectory.

We present a complete re-implementation of the model, featuring a Transformer-based text encoder with Rotational Position Embeddings (RoPE) and a lightweight 1D U-Net decoder. We evaluated our implementation against the official pre-trained checkpoint using the LJ Speech dataset.

---

## Architecture

The system functions sequentially through two main subsystems:

### 1. The Encoder (Text Processing)

- **Text Encoder:** Transforms raw text into contextualized embeddings using a stack of transformer blocks.
  
  - **Preprocessing:** Uses 1D convolutions (Prenet) for local feature extraction.
  
  - **Positional Encoding:** Implements **RoPE (Rotary Position Embedding)**, applied to half the embedding dimensions to preserve raw information.

- **Duration Predictor & Alignment:**
  
  - Uses **Monotonic Alignment Search (MAS)** during training to align phonemes to mel-frames.
  
  - A Duration Predictor network learns log-durations from these alignments to be used during inference.

### 2. The Decoder (Acoustic Generation)

- **Flow Matching:** Replaces standard diffusion with OT-CFM. It predicts a vector field $v_t$ to guide simple noise $x_0$ to a complex mel-spectrogram $x_1$.

- **1D U-Net Hybrid:** The decoder combines ResNet blocks (local features) with Transformer blocks (global context).

- **SnakeBeta Activation:** A key innovation found by the author is the use of SnakeBeta activation in Feed-Forward layers, which is periodic and better suited for audio waveform generation than ReLU.

---

## Dataset and Preprocessing

- **Dataset:** **LJ Speech** (approx. 24 hours of single-speaker English audio).

- **Text Processing:** IPA phonemization via `espeak-ng` with interspersing (blank tokens) to stabilize MAS.

- **Audio Processing:**
  
  - Sample Rate: 22,050 Hz.
  
  - STFT: 1024 FFT size, 256 hop length.
  
  - **Normalization:** Mel-spectrograms are normalized using dataset statistics. **This was found to be critical for training stability**.

---

## Experimental Results

We compared the Original Paper's reported results against our **Re-implementation (Original)** and a **Custom Variation**. All models were trained for 150 epochs on a 4-GPU cluster.

Quantitative Analysis

| **Metric**             | **Matcha (Paper)** | **Matcha (Retested)** | **Custom (Ours)**     |
| ---------------------- | ------------------ | --------------------- | --------------------- |
| **Parameters**         | 18.2M              | 18.2M                 | 18.2M                 |
| **RTF (GPU)**          | $0.038 \pm 0.019$  | **$0.019 \pm 0.008$** | **$0.018 \pm 0.008$** |
| **WER (%)**            | 2.09               | $4.03 \pm 6.72$       | $5.64 \pm 8.45$       |
| **Synthesis Time (s)** | -                  | $0.123 \pm 0.009$     | $0.110 \pm 0.007$     |

Subjective Evaluation (MOS)

*Mean Opinion Score evaluated by 31 participants.*

| **Model**                      | **MOS (3 samples)** |
| ------------------------------ | ------------------- |
| **Matcha (Paper)**             | **$3.84 \pm 0.08$** |
| **Original Re-implementation** | **$3.86 \pm 1.01$** |
| **Our Custom Model**           | $3.04 \pm 1.23$     |

### Key Findings

1. **Reproducibility:** Our re-implementation achieves a MOS of 3.86, virtually identical to the original paper's 3.84, confirming successful reproduction of audio quality.

2. **Efficiency:** Our implementations achieved an RTF of ~0.019, performing faster than the paper's reported 0.03829. Synthesis time scales quasi-linearly with text length.

3. **Critical Challenge:** We identified that **Mel Spectrogram Normalization** is mandatory. An initial version without it failed to converge, resulting in extremely high loss and artifacts.

---

## Implementation Details

### Versions Developed

1. **18M Parameter (Main):** The primary version used for all final results. Stable training and high quality. it is the current main branch.

2. **16M Parameter (Simplified):** A version with simplified transformer blocks in the decoder. It achieved comparable training dynamics to the 18M version.

3. **18M Parameter (from-scratch):** A version with our own implementation of all fundamental modules (For example : Attention blocks and other modules found in the difusion library). It achieved worse results than the other versions. This version can be found in the ZedBranch2 branch.

---

## References

1. **Original Paper:** Mehta, S., et al. "Matcha-TTS: A fast TTS architecture with conditional flow matching." arXiv:2309.03199 (2023).

2. **Implementation Report:** Adjal, M., Bouhai, Y., Bouhadjira, Z., Boularas, M.M. "MatchaTTS Implementation And Analysis." Sorbonne Université (2025-2026).