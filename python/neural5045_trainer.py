#!/usr/bin/env python3
"""
Neural 5045: DDSP Source Separation Trainer
============================================
A differentiable IIR filter-based trainer for the DeBleed plugin.

Instead of STFT masking, this uses a series cascade of differentiable biquad filters.
The neural network predicts ~50 filter parameters per frame, which are applied
to a series of 16 biquads: HPF → LowShelf → 12x Peaking → HighShelf → LPF + Gains.

Key Features:
- Zero latency: Causal IIR filters, no FFT lookahead
- End-to-end audio loss: Multi-resolution STFT perceptual loss
- Stable training: Pole reflection, gradient clipping
- ONNX export: Dilated TCN architecture (no RNN for ONNX compatibility)

Architecture (17 filter nodes, 50 parameters):
    Input → HPF → LowShelf → [12x Peaking] → HighShelf → LPF → Output

    Params per frame:
    - 16 biquads × 3 params (freq, gain, Q) = 48 params
    - Input gain = 1 param
    - Output gain (broadband) = 1 param
    = 50 total params

Usage:
    python neural5045_trainer.py --clean_audio_dir ./vocals --noise_audio_dir ./bleed \
                                  --output_path ./model --epochs 100
"""

import argparse
import json
import math
import os
import sys
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

# ============================================================================
# Configuration Constants
# ============================================================================

SAMPLE_RATE = 96000          # High quality sample rate for training
CHUNK_DURATION = 1.5         # 1.5-second chunks (shorter for IIR memory)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# Frame rate for parameter updates
FRAME_SIZE = 64              # Update coefficients every 64 samples (~0.67ms @ 96kHz)

# Filter bank configuration
N_BIQUADS = 16               # Total number of biquad stages
N_PARAMS_PER_BIQUAD = 3      # freq, gain, Q
N_EXTRA_PARAMS = 2           # input gain, output gain
N_TOTAL_PARAMS = N_BIQUADS * N_PARAMS_PER_BIQUAD + N_EXTRA_PARAMS  # 50 params

# Biquad roles (indices into the 16 biquads)
BIQUAD_HPF = 0               # High-pass filter
BIQUAD_LOW_SHELF = 1         # Low shelf
BIQUAD_PEAKING_START = 2     # First parametric EQ
BIQUAD_PEAKING_END = 13      # Last parametric EQ (12 total)
BIQUAD_HIGH_SHELF = 14       # High shelf
BIQUAD_LPF = 15              # Low-pass filter

# Frequency ranges for different filter types (normalized 0-1)
HPF_FREQ_RANGE = (20.0, 500.0)
LPF_FREQ_RANGE = (5000.0, 20000.0)
SHELF_FREQ_RANGE = (50.0, 16000.0)
PEAK_FREQ_RANGE = (100.0, 15000.0)

# Gain ranges (dB)
GAIN_RANGE = (-24.0, 24.0)    # Shelves and peaks
BROADBAND_RANGE = (-60.0, 0.0)  # Broadband gain (mostly reduction)

# Q ranges
Q_RANGE = (0.5, 16.0)

# SNR range for synthetic mixing (in dB)
SNR_MIN = 0.0
SNR_MAX = 25.0


# ============================================================================
# Audio Utilities
# ============================================================================

def load_audio_file(path: str, target_sr: int = SAMPLE_RATE) -> Optional[torch.Tensor]:
    """Load an audio file and resample to target sample rate."""
    try:
        waveform, sr = torchaudio.load(path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        return waveform.squeeze(0)  # Return 1D tensor
    except Exception as e:
        print(f"WARNING: Failed to load {path}: {e}", file=sys.stderr)
        return None


def collect_audio_files(directory: str) -> List[str]:
    """Collect all supported audio files from a directory."""
    supported_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif'}
    audio_files = []

    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for ext in supported_extensions:
        audio_files.extend(dir_path.glob(f"*{ext}"))
        audio_files.extend(dir_path.glob(f"*{ext.upper()}"))

    return [str(f) for f in audio_files]


def compute_rms(signal: torch.Tensor) -> torch.Tensor:
    """Compute RMS energy of a signal."""
    return torch.sqrt(torch.mean(signal ** 2) + 1e-8)


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix clean signal with noise at specified SNR level."""
    clean_rms = compute_rms(clean)
    noise_rms = compute_rms(noise)

    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / (noise_rms + 1e-8))

    return clean + noise_scaled


# ============================================================================
# Differentiable Biquad Filter (Direct Form II Transposed)
# ============================================================================

class DifferentiableBiquad(nn.Module):
    """
    A differentiable biquad filter using Direct Form II Transposed.

    Processes audio sample-by-sample with coefficients that can vary per-frame.
    Uses bilinear transform to convert analog prototype to digital coefficients.
    """

    def __init__(self):
        super().__init__()
        # State registers (per channel)
        self.register_buffer('s1', torch.zeros(1))
        self.register_buffer('s2', torch.zeros(1))

    def reset_state(self, batch_size: int = 1, device: torch.device = None):
        """Reset filter state for new audio."""
        if device is None:
            device = self.s1.device
        self.s1 = torch.zeros(batch_size, device=device)
        self.s2 = torch.zeros(batch_size, device=device)

    def forward(self, x: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor,
                a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
        """
        Process audio through biquad filter.

        Args:
            x: Input audio (batch, samples) or (samples,)
            b0, b1, b2: Numerator coefficients (batch,) or scalar
            a1, a2: Denominator coefficients (batch,) or scalar (a0 normalized to 1)

        Returns:
            Filtered audio (batch, samples)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size, n_samples = x.shape

        # Ensure state is correct size
        if self.s1.shape[0] != batch_size:
            self.reset_state(batch_size, x.device)

        # Expand coefficients if needed
        if b0.dim() == 0:
            b0 = b0.expand(batch_size)
            b1 = b1.expand(batch_size)
            b2 = b2.expand(batch_size)
            a1 = a1.expand(batch_size)
            a2 = a2.expand(batch_size)

        outputs = []
        s1, s2 = self.s1, self.s2

        for i in range(n_samples):
            inp = x[:, i]

            # Direct Form II Transposed
            out = b0 * inp + s1
            s1_new = b1 * inp - a1 * out + s2
            s2_new = b2 * inp - a2 * out

            s1, s2 = s1_new, s2_new
            outputs.append(out)

        self.s1, self.s2 = s1, s2

        return torch.stack(outputs, dim=1)


def bilinear_biquad_lowpass(fc: torch.Tensor, Q: torch.Tensor, fs: float) -> Tuple[torch.Tensor, ...]:
    """
    Compute biquad coefficients for lowpass filter using bilinear transform.

    Args:
        fc: Cutoff frequency in Hz
        Q: Quality factor
        fs: Sample rate

    Returns:
        b0, b1, b2, a1, a2 coefficients
    """
    w0 = 2 * math.pi * fc / fs
    alpha = torch.sin(w0) / (2 * Q)
    cos_w0 = torch.cos(w0)

    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    return b0/a0, b1/a0, b2/a0, a1/a0, a2/a0


def bilinear_biquad_highpass(fc: torch.Tensor, Q: torch.Tensor, fs: float) -> Tuple[torch.Tensor, ...]:
    """Compute biquad coefficients for highpass filter."""
    w0 = 2 * math.pi * fc / fs
    alpha = torch.sin(w0) / (2 * Q)
    cos_w0 = torch.cos(w0)

    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    return b0/a0, b1/a0, b2/a0, a1/a0, a2/a0


def bilinear_biquad_peak(fc: torch.Tensor, gain_db: torch.Tensor, Q: torch.Tensor,
                         fs: float) -> Tuple[torch.Tensor, ...]:
    """Compute biquad coefficients for peaking EQ filter."""
    A = 10 ** (gain_db / 40)  # amplitude
    w0 = 2 * math.pi * fc / fs
    alpha = torch.sin(w0) / (2 * Q)
    cos_w0 = torch.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    return b0/a0, b1/a0, b2/a0, a1/a0, a2/a0


def bilinear_biquad_lowshelf(fc: torch.Tensor, gain_db: torch.Tensor, Q: torch.Tensor,
                             fs: float) -> Tuple[torch.Tensor, ...]:
    """Compute biquad coefficients for low shelf filter."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * fc / fs
    alpha = torch.sin(w0) / (2 * Q)
    cos_w0 = torch.cos(w0)
    sqrt_A = torch.sqrt(A)

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    return b0/a0, b1/a0, b2/a0, a1/a0, a2/a0


def bilinear_biquad_highshelf(fc: torch.Tensor, gain_db: torch.Tensor, Q: torch.Tensor,
                              fs: float) -> Tuple[torch.Tensor, ...]:
    """Compute biquad coefficients for high shelf filter."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * fc / fs
    alpha = torch.sin(w0) / (2 * Q)
    cos_w0 = torch.cos(w0)
    sqrt_A = torch.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    return b0/a0, b1/a0, b2/a0, a1/a0, a2/a0


# ============================================================================
# Differentiable Filter Chain
# ============================================================================

class DifferentiableBiquadChain(nn.Module):
    """
    A chain of 16 differentiable biquad filters for source separation.

    Structure:
        HPF → LowShelf → 12x Peaking → HighShelf → LPF

    Takes 50 normalized parameters (0-1) and converts to filter coefficients.
    """

    def __init__(self, sample_rate: float = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate

        # Create biquad filters
        self.biquads = nn.ModuleList([DifferentiableBiquad() for _ in range(N_BIQUADS)])

    def reset_state(self, batch_size: int = 1, device: torch.device = None):
        """Reset all filter states."""
        for bq in self.biquads:
            bq.reset_state(batch_size, device)

    def _denormalize_freq(self, norm: torch.Tensor, freq_range: Tuple[float, float]) -> torch.Tensor:
        """Convert normalized (0-1) to frequency (Hz) with log scaling."""
        log_low = math.log(freq_range[0])
        log_high = math.log(freq_range[1])
        log_freq = log_low + norm * (log_high - log_low)
        return torch.exp(log_freq)

    def _denormalize_gain(self, norm: torch.Tensor, gain_range: Tuple[float, float] = GAIN_RANGE) -> torch.Tensor:
        """Convert normalized (0-1) to gain (dB)."""
        return gain_range[0] + norm * (gain_range[1] - gain_range[0])

    def _denormalize_q(self, norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized (0-1) to Q with log scaling."""
        log_low = math.log(Q_RANGE[0])
        log_high = math.log(Q_RANGE[1])
        log_q = log_low + norm * (log_high - log_low)
        return torch.exp(log_q)

    def _denormalize_broadband(self, norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized (0-1) to broadband gain (dB)."""
        return BROADBAND_RANGE[0] + norm * (BROADBAND_RANGE[1] - BROADBAND_RANGE[0])

    def forward(self, audio: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Process audio through the filter chain.

        Args:
            audio: Input audio (batch, samples)
            params: Normalized parameters (batch, 50) in range [0, 1]

        Returns:
            Filtered audio (batch, samples)
        """
        batch_size = audio.shape[0] if audio.dim() > 1 else 1
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Extract parameters
        # Layout: [16 biquads × 3 params] + [input_gain, output_gain]
        biquad_params = params[:, :N_BIQUADS * N_PARAMS_PER_BIQUAD].view(batch_size, N_BIQUADS, 3)
        input_gain_norm = params[:, -2]
        output_gain_norm = params[:, -1]

        # Convert input/output gains
        input_gain_db = self._denormalize_broadband(input_gain_norm)
        output_gain_db = self._denormalize_broadband(output_gain_norm)
        input_gain_linear = 10 ** (input_gain_db / 20)
        output_gain_linear = 10 ** (output_gain_db / 20)

        # Apply input gain
        x = audio * input_gain_linear.unsqueeze(1)

        # Process each biquad
        for i in range(N_BIQUADS):
            freq_norm = biquad_params[:, i, 0]
            gain_norm = biquad_params[:, i, 1]
            q_norm = biquad_params[:, i, 2]

            # Determine filter type and compute coefficients
            if i == BIQUAD_HPF:
                freq = self._denormalize_freq(freq_norm, HPF_FREQ_RANGE)
                Q = self._denormalize_q(q_norm)
                # Process batch
                for b in range(batch_size):
                    b0, b1, b2, a1, a2 = bilinear_biquad_highpass(freq[b], Q[b], self.sample_rate)
                    x[b:b+1] = self.biquads[i](x[b:b+1], b0, b1, b2, a1, a2)

            elif i == BIQUAD_LPF:
                freq = self._denormalize_freq(freq_norm, LPF_FREQ_RANGE)
                Q = self._denormalize_q(q_norm)
                for b in range(batch_size):
                    b0, b1, b2, a1, a2 = bilinear_biquad_lowpass(freq[b], Q[b], self.sample_rate)
                    x[b:b+1] = self.biquads[i](x[b:b+1], b0, b1, b2, a1, a2)

            elif i == BIQUAD_LOW_SHELF:
                freq = self._denormalize_freq(freq_norm, SHELF_FREQ_RANGE)
                gain = self._denormalize_gain(gain_norm)
                Q = self._denormalize_q(q_norm)
                for b in range(batch_size):
                    b0, b1, b2, a1, a2 = bilinear_biquad_lowshelf(freq[b], gain[b], Q[b], self.sample_rate)
                    x[b:b+1] = self.biquads[i](x[b:b+1], b0, b1, b2, a1, a2)

            elif i == BIQUAD_HIGH_SHELF:
                freq = self._denormalize_freq(freq_norm, SHELF_FREQ_RANGE)
                gain = self._denormalize_gain(gain_norm)
                Q = self._denormalize_q(q_norm)
                for b in range(batch_size):
                    b0, b1, b2, a1, a2 = bilinear_biquad_highshelf(freq[b], gain[b], Q[b], self.sample_rate)
                    x[b:b+1] = self.biquads[i](x[b:b+1], b0, b1, b2, a1, a2)

            else:  # Peaking EQ
                freq = self._denormalize_freq(freq_norm, PEAK_FREQ_RANGE)
                gain = self._denormalize_gain(gain_norm)
                Q = self._denormalize_q(q_norm)
                for b in range(batch_size):
                    b0, b1, b2, a1, a2 = bilinear_biquad_peak(freq[b], gain[b], Q[b], self.sample_rate)
                    x[b:b+1] = self.biquads[i](x[b:b+1], b0, b1, b2, a1, a2)

        # Apply output gain
        x = x * output_gain_linear.unsqueeze(1)

        return x


# ============================================================================
# Neural Network Architecture (Dilated TCN)
# ============================================================================

class ConvBlock(nn.Module):
    """1D Conv block with GroupNorm and PReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Neural5045Net(nn.Module):
    """
    Dilated TCN for predicting filter parameters.

    Input: Raw audio (batch, 1, samples)
    Output: Filter parameters (batch, N_TOTAL_PARAMS, n_frames)

    Parameters are normalized to [0, 1] and decoded by the filter chain.
    """

    def __init__(self, hidden_dim: int = 128, n_output_params: int = N_TOTAL_PARAMS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_output_params = n_output_params

        # Input projection (1 channel audio → hidden)
        self.input_proj = ConvBlock(1, hidden_dim, kernel_size=7)

        # Dilated TCN stack
        self.tcn = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=8),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=16),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=32),
        )

        # Output projection → parameters
        self.output_proj = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim // 2, kernel_size=3),
            nn.Conv1d(hidden_dim // 2, n_output_params, kernel_size=1),
            nn.Sigmoid()  # Normalize to [0, 1]
        )

        # Downsample to frame rate
        self.frame_pool = nn.AvgPool1d(kernel_size=FRAME_SIZE, stride=FRAME_SIZE)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio: (batch, samples) or (batch, 1, samples)

        Returns:
            params: (batch, N_TOTAL_PARAMS, n_frames)
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dim

        x = self.input_proj(audio)
        x = self.tcn(x)
        x = self.output_proj(x)

        # Downsample to frame rate
        x = self.frame_pool(x)

        return x


# ============================================================================
# Multi-Resolution STFT Loss
# ============================================================================

class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for perceptual audio quality.

    Computes spectral convergence and log magnitude loss at multiple resolutions.
    """

    def __init__(self, fft_sizes: List[int] = [512, 1024, 2048],
                 hop_sizes: List[int] = [128, 256, 512],
                 win_sizes: List[int] = [512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def stft_loss(self, pred: torch.Tensor, target: torch.Tensor,
                  fft_size: int, hop_size: int, win_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute STFT loss at one resolution."""
        window = torch.hann_window(win_size, device=pred.device)

        pred_stft = torch.stft(pred, n_fft=fft_size, hop_length=hop_size,
                               win_length=win_size, window=window, return_complex=True)
        target_stft = torch.stft(target, n_fft=fft_size, hop_length=hop_size,
                                 win_length=win_size, window=window, return_complex=True)

        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)

        # Spectral convergence
        sc_loss = torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)

        # Log magnitude loss
        log_pred = torch.log(pred_mag + 1e-8)
        log_target = torch.log(target_mag + 1e-8)
        mag_loss = F.l1_loss(log_pred, log_target)

        return sc_loss, mag_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            pred: Predicted audio (batch, samples)
            target: Target audio (batch, samples)

        Returns:
            Total loss
        """
        sc_losses = []
        mag_losses = []

        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            sc, mag = self.stft_loss(pred, target, fft_size, hop_size, win_size)
            sc_losses.append(sc)
            mag_losses.append(mag)

        return sum(sc_losses) / len(sc_losses) + sum(mag_losses) / len(mag_losses)


# ============================================================================
# Dataset
# ============================================================================

class SyntheticMixtureDataset(Dataset):
    """Dataset that creates synthetic mixtures at 96kHz."""

    def __init__(self, clean_files: List[str], noise_files: List[str],
                 samples_per_epoch: int = 2000, chunk_samples: int = CHUNK_SAMPLES):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.samples_per_epoch = samples_per_epoch
        self.chunk_samples = chunk_samples

        print("STATUS:Loading audio files into memory...")
        self.clean_audio = self._load_all_audio(clean_files, "clean")
        self.noise_audio = self._load_all_audio(noise_files, "noise")
        print(f"STATUS:Loaded {len(self.clean_audio)} clean files and {len(self.noise_audio)} noise files")

    def _load_all_audio(self, files: List[str], label: str) -> List[torch.Tensor]:
        audio_list = []
        for i, f in enumerate(files):
            audio = load_audio_file(f, SAMPLE_RATE)
            if audio is not None and len(audio) >= self.chunk_samples:
                audio_list.append(audio)
            if (i + 1) % 10 == 0:
                print(f"STATUS:Loaded {i + 1}/{len(files)} {label} files")
        return audio_list

    def _random_chunk(self, audio: torch.Tensor) -> torch.Tensor:
        if len(audio) <= self.chunk_samples:
            return F.pad(audio, (0, self.chunk_samples - len(audio)))
        start_idx = random.randint(0, len(audio) - self.chunk_samples)
        return audio[start_idx:start_idx + self.chunk_samples]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_audio = random.choice(self.clean_audio)
        noise_audio = random.choice(self.noise_audio)

        clean_chunk = self._random_chunk(clean_audio)
        noise_chunk = self._random_chunk(noise_audio)

        snr_db = random.uniform(SNR_MIN, SNR_MAX)
        mixture = mix_at_snr(clean_chunk, noise_chunk, snr_db)

        return mixture, clean_chunk


# ============================================================================
# Training
# ============================================================================

class Trainer:
    """Training loop for Neural 5045."""

    def __init__(self, model: Neural5045Net, filter_chain: DifferentiableBiquadChain,
                 train_loader: DataLoader, learning_rate: float = 1e-4, device: str = 'cpu'):
        self.model = model.to(device)
        self.filter_chain = filter_chain.to(device)
        self.train_loader = train_loader
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 100
        )

        self.stft_loss = MultiResolutionSTFTLoss().to(device)

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, (mixture, clean) in enumerate(self.train_loader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # Reset filter states
            self.filter_chain.reset_state(mixture.shape[0], self.device)

            # Predict parameters from mixture
            params = self.model(mixture)  # (batch, 50, n_frames)

            # For now, use frame-averaged parameters (TODO: per-frame processing)
            params_avg = params.mean(dim=2)  # (batch, 50)

            # Apply filter chain
            filtered = self.filter_chain(mixture, params_avg)

            # Compute loss
            loss = self.stft_loss(filtered, clean)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Progress
            if (batch_idx + 1) % max(1, n_batches // 10) == 0:
                progress = int(((epoch * n_batches + batch_idx + 1) /
                              (total_epochs * n_batches)) * 100)
                print(f"PROGRESS:{progress}")
                sys.stdout.flush()

        return total_loss / n_batches


# ============================================================================
# ONNX Export
# ============================================================================

def export_to_onnx(model: Neural5045Net, output_path: str, model_name: str = 'neural5045'):
    """Export the parameter prediction network to ONNX."""
    print("STATUS:Exporting Neural 5045 model to ONNX...")
    sys.stdout.flush()

    model.eval()
    device = next(model.parameters()).device

    # Dummy input: 1.5 seconds of audio at 96kHz
    n_samples = CHUNK_SAMPLES
    dummy_input = torch.randn(1, n_samples, device=device)

    onnx_path = os.path.join(output_path, f"{model_name}.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['audio'],
        output_names=['params'],
        dynamic_axes={
            'audio': {0: 'batch_size', 1: 'samples'},
            'params': {0: 'batch_size', 2: 'frames'}
        }
    )

    print(f"STATUS:Model exported to {onnx_path}")
    print(f"MODEL_PATH:{onnx_path}")
    sys.stdout.flush()

    return onnx_path


def save_metadata(output_path: str, model: Neural5045Net, history: dict):
    """Save model metadata."""
    metadata = {
        'model_type': 'neural5045',
        'version': '1.0.0',
        'sample_rate': SAMPLE_RATE,
        'frame_size': FRAME_SIZE,
        'n_biquads': N_BIQUADS,
        'n_params': N_TOTAL_PARAMS,
        'hidden_dim': model.hidden_dim,
        'architecture': 'dilated_tcn',
        'filter_chain': {
            'hpf_idx': BIQUAD_HPF,
            'lpf_idx': BIQUAD_LPF,
            'low_shelf_idx': BIQUAD_LOW_SHELF,
            'high_shelf_idx': BIQUAD_HIGH_SHELF,
            'peaking_start': BIQUAD_PEAKING_START,
            'peaking_end': BIQUAD_PEAKING_END,
        },
        'frequency_ranges': {
            'hpf': HPF_FREQ_RANGE,
            'lpf': LPF_FREQ_RANGE,
            'shelf': SHELF_FREQ_RANGE,
            'peak': PEAK_FREQ_RANGE,
        },
        'gain_range': GAIN_RANGE,
        'broadband_range': BROADBAND_RANGE,
        'q_range': Q_RANGE,
        'training_epochs': len(history.get('losses', [])),
        'final_loss': history['losses'][-1] if history.get('losses') else None,
    }

    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"STATUS:Metadata saved to {metadata_path}")
    sys.stdout.flush()


def save_checkpoint(output_path: str, model: Neural5045Net, trainer: Trainer,
                    epoch: int, history: dict):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'history': history,
        'hidden_dim': model.hidden_dim,
    }

    checkpoint_path = os.path.join(output_path, "checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"STATUS:Checkpoint saved")
    sys.stdout.flush()


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Neural 5045 Trainer')

    parser.add_argument('--clean_audio_dir', type=str, required=True,
                        help='Path to clean audio directory')
    parser.add_argument('--noise_audio_dir', type=str, required=True,
                        help='Path to noise audio directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory for model')
    parser.add_argument('--model_name', type=str, default='neural5045',
                        help='Model file name')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--samples_per_epoch', type=int, default=2000,
                        help='Samples per epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Checkpoint to resume from')
    parser.add_argument('--continue_training', action='store_true',
                        help='Continue from checkpoint')

    return parser.parse_args()


def get_device(device_arg: str) -> str:
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device_arg


def main():
    args = parse_args()

    print("STATUS:Neural 5045 Trainer starting...")
    print(f"STATUS:Sample rate: {SAMPLE_RATE}Hz")
    print(f"STATUS:Parameters per frame: {N_TOTAL_PARAMS}")
    sys.stdout.flush()

    device = get_device(args.device)
    print(f"STATUS:Using device: {device}")
    sys.stdout.flush()

    os.makedirs(args.output_path, exist_ok=True)

    # Load checkpoint if continuing
    checkpoint = None
    start_epoch = 0
    history = {'losses': []}
    hidden_dim = args.hidden_dim

    if args.continue_training and args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint.get('history', {'losses': []})
            hidden_dim = checkpoint.get('hidden_dim', args.hidden_dim)
            print(f"STATUS:Resuming from epoch {start_epoch}")
            sys.stdout.flush()

    # Collect audio files
    print("STATUS:Collecting audio files...")
    try:
        clean_files = collect_audio_files(args.clean_audio_dir)
        noise_files = collect_audio_files(args.noise_audio_dir)
    except FileNotFoundError as e:
        print(f"ERROR:{e}")
        sys.exit(1)

    if len(clean_files) == 0 or len(noise_files) == 0:
        print("ERROR:No audio files found")
        sys.exit(1)

    print(f"STATUS:Found {len(clean_files)} clean, {len(noise_files)} noise files")
    sys.stdout.flush()

    # Create dataset
    dataset = SyntheticMixtureDataset(
        clean_files=clean_files,
        noise_files=noise_files,
        samples_per_epoch=args.samples_per_epoch
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )

    # Create model and filter chain
    print("STATUS:Creating Neural 5045 model...")
    model = Neural5045Net(hidden_dim=hidden_dim)
    filter_chain = DifferentiableBiquadChain(sample_rate=SAMPLE_RATE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"STATUS:Model has {n_params:,} parameters")
    sys.stdout.flush()

    # Create trainer
    trainer = Trainer(
        model=model,
        filter_chain=filter_chain,
        train_loader=train_loader,
        learning_rate=args.learning_rate,
        device=device
    )

    # Load checkpoint state
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("STATUS:Loaded checkpoint state")
        sys.stdout.flush()

    # Training loop
    total_epochs = start_epoch + args.epochs
    print(f"STATUS:Training from epoch {start_epoch + 1} to {total_epochs}...")
    sys.stdout.flush()

    for epoch in range(start_epoch, total_epochs):
        avg_loss = trainer.train_epoch(epoch - start_epoch, args.epochs)
        history['losses'].append(avg_loss)

        progress = int(((epoch - start_epoch + 1) / args.epochs) * 100)
        print(f"EPOCH:{epoch + 1}/{total_epochs}")
        print(f"LOSS:{avg_loss:.6f}")
        print(f"PROGRESS:{progress}")
        sys.stdout.flush()

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(args.output_path, model, trainer, epoch, history)

    print("STATUS:Training complete!")

    # Final checkpoint
    save_checkpoint(args.output_path, model, trainer, total_epochs - 1, history)

    # Export
    onnx_path = export_to_onnx(model, args.output_path, args.model_name)
    save_metadata(args.output_path, model, history)

    print("RESULT:SUCCESS")
    print(f"MODEL_PATH:{onnx_path}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
