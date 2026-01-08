# Main Model wrapper (Architecture + Loss)

__author__ = "Mouad"

import math
import random
import datetime as dt
import torch
import torch.nn as nn

# Assure-toi que ces imports pointent vers tes fichiers
from . import monotonic_align as monotonic_align
from .encoder import TextEncoder # not coded yet
from .flow_matching import CFM
from .utils import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask
)


# log = utils.get_pylogger(__name__)

class MatchaTTS(nn.Module):  # [MODIFIED] Inherits from nn.Module instead of BaseLightningClass
    def __init__(
            self,
            n_vocab,
            n_feats,
            encoder_params,  #  Accepts config object instead of dict/kwargs
            duration_predictor_params,  # Needed for TextEncoder init
            decoder_params,  #  Accepts config object
            cfm_params,  #Accepts config object
            mel_mean,  #  Passed directly instead of via data_statistics list
            mel_std,
            prior_loss=True,

    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_feats = n_feats
        self.prior_loss = prior_loss

        # modified because we are not using baselightning module
        self.register_buffer("mel_mean", torch.tensor(mel_mean).float())
        self.register_buffer("mel_std", torch.tensor(mel_std).float())

        self.encoder = TextEncoder(
            encoder_params.encoder_type,
            encoder_params,
            duration_predictor_params,  # class duration_predictor_params in config
            n_vocab,
        )

        self.decoder = CFM(
            in_channels=2 * encoder_params.n_feats,
            out_channel=encoder_params.n_feats,
            cfm_params=cfm_params,
            decoder_params=decoder_params,
        )

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, n_timesteps, temperature=1.0, length_scale=1.0):
        """
        Generates mel-spectrogram from text.
        (Logic remains mostly identical to original, just removed RTF logging for simplicity)
        """
        # For RTF computation
        t = dt.datetime.now()


        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }


    def forward(self, x, x_lengths, y, y_lengths, cond=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
        """

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)


        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()  # b, t_text, T_mel

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, cond=cond)

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        return dur_loss, prior_loss, diff_loss, attn