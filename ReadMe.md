# Implementation and Analysis of the MatchaTTS Paper


## Project Description

This repository contains our group's implementation and analysis of the Matcha-TTS model, as described in the paper "Matcha-TTS: A fast TTS architecture with conditional flow matching" by Shivam Mehta et al. Matcha-TTS is a non-autoregressive text-to-speech (TTS) system that leverages optimal-transport conditional flow matching (OT-CFM) for efficient, probabilistic generation of mel-spectrograms, enabling faster synthesis compared to traditional diffusion-based models like Grad-TTS.

As a group project by four students, we focus on reproducing the model's architecture, training it from scratch, and evaluating it against the original paper's metrics and the official pre-trained models. Our goal is to gain a deep understanding of flow-matching techniques in TTS while providing a clear, reproducible implementation for educational purposes.

Key features of Matcha-TTS (as per the paper):
- **Encoder-Decoder Architecture**: Text encoder with Rotary Positional Embeddings (RoPE) and a 1D U-Net decoder for flow prediction.
- **Flow Matching**: Uses OT-CFM for straight-line paths in probabilistic space, reducing inference steps (NFE) to as few as 2-10.
- **Vocoder Integration**: Paired with HiFi-GAN for waveform generation.
- **Datasets**: Primarily trained on LJ Speech for single-speaker TTS.

This project includes our re-implementation, scripts for training/evaluation, and comparative results.


## Objectives

- **Utilize Official Resources**: Leverage the official GitHub repository and pre-trained models to reproduce reported metrics.
- **Re-implement the Paper**: Build the model from scratch in PyTorch, following the architecture and training procedures described in the paper.


## References

[![GitHub license](https://img.shields.io/github/license/shivammehta25/Matcha-TTS)](https://github.com/shivammehta25/Matcha-TTS/blob/main/LICENSE)  
[![Paper](https://img.shields.io/badge/arXiv-2309.03199-b31b1b.svg)](https://arxiv.org/abs/2309.03199)  
[![Official Repo](https://img.shields.io/badge/GitHub-Official%20Repo-blue.svg)](https://github.com/shivammehta25/Matcha-TTS)

- **Paper**: Mehta, S., et al. "Matcha-TTS: A fast TTS architecture with conditional flow matching." arXiv preprint arXiv:2309.03199 (2023).
- **Official Repository**: [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
- **Demo**: [Hugging Face Space](https://huggingface.co/spaces/shivkanthb/Matcha-TTS)
- **Wiki/FAQs**: [Official Wiki](https://github.com/shivammehta25/Matcha-TTS/wiki)