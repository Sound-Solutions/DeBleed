#!/usr/bin/env python3
"""
DeBleed Neural Gate Trainer
===========================
A headless CLI trainer for the DeBleed "Timbre-Aware Primary Source Enhancer."

Trains a lightweight DeepFilterNet-inspired mask estimation network on synthetic
mixtures of clean audio and noise, then exports to ONNX for real-time inference.

Architecture Note:
The ONNX model exports only the mask estimation network. STFT/iSTFT transforms
are handled in C++ for optimal real-time performance. The plugin will:
1. Compute STFT on input audio
2. Feed magnitude spectrogram to the ONNX model
3. Get mask output and apply to complex spectrogram
4. Compute iSTFT to reconstruct audio

Usage:
    python trainer.py --clean_audio_dir ./vocals --noise_audio_dir ./bleed \
                      --output_path ./model --epochs 50
"""

import argparse
import json
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

SAMPLE_RATE = 48000          # Standard professional audio rate
CHUNK_DURATION = 3.0         # 3-second chunks for training
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# =============================================================================
# Dual-Stream STFT Configuration
# =============================================================================
# Stream A: Fast/Temporal (transients, high-frequency detail)
# - 256-point FFT = 5.33ms window @ 48kHz
# - ~187 Hz per bin resolution
# - Good for transients and high-frequency content
#
# Stream B: Slow/Spectral (bass precision)
# - 2048-point FFT = 42.67ms window @ 48kHz
# - ~23.4 Hz per bin resolution
# - Critical for distinguishing 60Hz from 150Hz
#
# Both streams use the same hop size for frame alignment.
# =============================================================================

# Stream A: Fast STFT (transients/highs) - original parameters
N_FFT_A = 256
WIN_LENGTH_A = 256
N_FREQ_BINS_A = N_FFT_A // 2 + 1  # 129 frequency bins

# Stream B: Slow STFT (bass precision)
N_FFT_B = 2048
WIN_LENGTH_B = 2048
N_FREQ_BINS_B = N_FFT_B // 2 + 1  # 1025 frequency bins

# We only use the lower portion of Stream B (up to ~1.5kHz)
# This gives us fine bass resolution without redundant high-freq info
STREAM_B_BINS_USED = 128  # Bins 0-127 = 0 to ~1.5kHz @ 48kHz

# Common hop size for frame alignment
HOP_LENGTH = 128  # 2.67ms @ 48kHz

# Total input features to neural network
# Stream A: 129 bins (full spectrum, coarse)
# Stream B: 128 bins (bass detail, fine)
N_TOTAL_FEATURES = N_FREQ_BINS_A + STREAM_B_BINS_USED  # 257 features

# Legacy aliases for backward compatibility
N_FFT = N_FFT_A
WIN_LENGTH = WIN_LENGTH_A
N_FREQ_BINS = N_FREQ_BINS_A  # Stream A bins (for backward compatibility)

# Dual-output mask: 257 bins = 129 (Stream A) + 128 (Stream B bass)
N_OUTPUT_BINS = N_TOTAL_FEATURES  # 257 total output bins

# SNR range for synthetic mixing (in dB)
# Higher SNR = less aggressive model (preserves more vocal)
# Lower SNR = more aggressive model (cuts more bleed, but may damage vocal)
SNR_MIN = 3.0    # Was -5.0 - don't train on extreme bleed cases
SNR_MAX = 15.0   # Was 10.0 - train on "vocal is clearly louder" cases


# ============================================================================
# Audio Utilities
# ============================================================================

def load_audio_file(path: str, target_sr: int = SAMPLE_RATE) -> Optional[torch.Tensor]:
    """
    Load an audio file and resample to target sample rate.
    Returns mono audio as 1D tensor, or None if loading fails.
    """
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
    """
    Mix clean signal with noise at specified SNR level.
    SNR = 10 * log10(P_signal / P_noise)
    """
    clean_rms = compute_rms(clean)
    noise_rms = compute_rms(noise)

    # Compute required noise scaling
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / (noise_rms + 1e-8))

    return clean + noise_scaled


def compute_stft(audio: torch.Tensor, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                  win_length: int = WIN_LENGTH) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute STFT and return magnitude and phase.
    Input: audio (batch, samples) or (samples,)
    Output: magnitude (batch, freq, frames), phase (batch, freq, frames)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Compute STFT
    window = torch.hann_window(win_length, device=audio.device)
    stft_out = torch.stft(audio, n_fft=n_fft, hop_length=hop_length,
                          win_length=win_length, window=window,
                          return_complex=True, center=True)

    # Extract magnitude and phase
    magnitude = torch.abs(stft_out)
    phase = torch.angle(stft_out)

    return magnitude, phase


def compute_dual_stream_stft(audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute dual-stream STFT for enhanced frequency resolution.

    Stream A (Fast): 256-point FFT - good for transients, ~187Hz resolution
    Stream B (Slow): 2048-point FFT - bass precision, ~23Hz resolution

    Input: audio (batch, samples) or (samples,)
    Output:
        mag_a (batch, 129, frames) - Stream A magnitude (full spectrum)
        phase_a (batch, 129, frames) - Stream A phase
        mag_b_low (batch, 128, frames) - Stream B magnitude (bass bins only)
        phase_b (batch, 1025, frames) - Stream B phase (full, for reconstruction)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Stream A: Fast STFT (transients/highs)
    window_a = torch.hann_window(WIN_LENGTH_A, device=audio.device)
    stft_a = torch.stft(audio, n_fft=N_FFT_A, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH_A, window=window_a,
                        return_complex=True, center=True)
    mag_a = torch.abs(stft_a)
    phase_a = torch.angle(stft_a)

    # Stream B: Slow STFT (bass precision)
    window_b = torch.hann_window(WIN_LENGTH_B, device=audio.device)
    stft_b = torch.stft(audio, n_fft=N_FFT_B, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH_B, window=window_b,
                        return_complex=True, center=True)
    mag_b = torch.abs(stft_b)
    phase_b = torch.angle(stft_b)

    # Extract only the lower bins from Stream B (bass region)
    # Bins 0-127 cover 0 to ~1.5kHz with ~23Hz resolution
    mag_b_low = mag_b[:, :STREAM_B_BINS_USED, :]

    return mag_a, phase_a, mag_b_low, phase_b


def concatenate_dual_stream_features(mag_a: torch.Tensor, mag_b_low: torch.Tensor) -> torch.Tensor:
    """
    Concatenate dual-stream magnitude features for neural network input.

    Input:
        mag_a (batch, 129, frames) - Stream A full spectrum
        mag_b_low (batch, 128, frames) - Stream B bass bins

    Output:
        features (batch, 257, frames) - Concatenated features
            [0:129] = Stream A (coarse full spectrum)
            [129:257] = Stream B bass (fine low-frequency detail)
    """
    return torch.cat([mag_a, mag_b_low], dim=1)


def compute_istft(magnitude: torch.Tensor, phase: torch.Tensor, n_fft: int = N_FFT,
                   hop_length: int = HOP_LENGTH, win_length: int = WIN_LENGTH) -> torch.Tensor:
    """
    Compute inverse STFT from magnitude and phase.
    Input: magnitude (batch, freq, frames), phase (batch, freq, frames)
    Output: audio (batch, samples)
    """
    # Reconstruct complex spectrogram
    complex_spec = magnitude * torch.exp(1j * phase)

    # Compute iSTFT
    window = torch.hann_window(win_length, device=magnitude.device)
    audio = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window=window, center=True)

    return audio


# ============================================================================
# Neural Network Architecture
# ============================================================================

def get_valid_num_groups(num_channels: int, preferred_groups: int = 8) -> int:
    """Find a valid number of groups for GroupNorm that divides num_channels."""
    for g in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm and PReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        num_groups = get_valid_num_groups(out_channels, preferred_groups=8)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class MaskEstimator(nn.Module):
    """
    Dual-stream mask estimation network inspired by DeepFilterNet.

    Takes concatenated dual-stream magnitude features:
    - Stream A (129 bins): Full spectrum at ~187Hz resolution
    - Stream B (128 bins): Bass region at ~23Hz resolution

    Outputs a dual mask [0, 1] for both streams:
    - Output [0:129] = Stream A mask (highs, 187Hz resolution)
    - Output [129:257] = Stream B mask (bass, 23Hz resolution)

    Input: dual-stream features (batch, 257, frames)
           [0:129] = Stream A (coarse full spectrum)
           [129:257] = Stream B bass (fine low-frequency detail)
    Output: mask (batch, 257, frames) in range [0, 1]
    """

    def __init__(self, n_input_features: int = N_TOTAL_FEATURES,
                 n_output_bins: int = N_OUTPUT_BINS, hidden_dim: int = 64):
        super().__init__()

        self.n_input_features = n_input_features
        self.n_output_bins = n_output_bins
        self.hidden_dim = hidden_dim

        # Input normalization parameters (learned)
        self.input_norm = nn.GroupNorm(1, n_input_features)  # Instance norm equivalent

        # Encoder - compress dual-stream frequency information
        self.encoder = nn.Sequential(
            ConvBlock(n_input_features, hidden_dim * 2, kernel_size=5),
            ConvBlock(hidden_dim * 2, hidden_dim, kernel_size=5),
        )

        # Temporal modeling using dilated convolutions (ONNX-friendly alternative to GRU)
        temporal_groups = get_valid_num_groups(hidden_dim, preferred_groups=8)
        self.temporal = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(temporal_groups, hidden_dim),
            nn.PReLU(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(temporal_groups, hidden_dim),
            nn.PReLU(hidden_dim),
        )

        # Decoder - expand back to output frequency bins
        self.decoder = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim * 2, kernel_size=5),
            ConvBlock(hidden_dim * 2, n_output_bins, kernel_size=5),
        )

        # Final mask output
        self.mask_output = nn.Sequential(
            nn.Conv1d(n_output_bins, n_output_bins, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: dual-stream features (batch, 257, frames)
        Output: dual mask (batch, 257, frames) in range [0, 1]
               [0:129] = Stream A mask, [129:257] = Stream B bass mask
        """
        # Normalize input
        x = self.input_norm(features)

        # Encode
        x = self.encoder(x)

        # Temporal processing
        x = self.temporal(x)

        # Decode
        x = self.decoder(x)

        # Output mask
        mask = self.mask_output(x)

        return mask


# ============================================================================
# Dataset
# ============================================================================

class SyntheticMixtureDataset(Dataset):
    """
    Dataset that creates synthetic mixtures of clean and noise audio.

    For each sample:
    1. Randomly select a clean audio file and noise audio file
    2. Extract random 3-second chunks from each
    3. Mix at a random SNR between -5dB and +10dB
    4. Return (mixed_magnitude, clean_magnitude, ideal_mask)
    """

    def __init__(self, clean_files: List[str], noise_files: List[str],
                 samples_per_epoch: int = 1000, chunk_samples: int = CHUNK_SAMPLES):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.samples_per_epoch = samples_per_epoch
        self.chunk_samples = chunk_samples

        # Pre-load all audio files into memory for faster training
        print("STATUS:Loading audio files into memory...")
        self.clean_audio = self._load_all_audio(clean_files, "clean")
        self.noise_audio = self._load_all_audio(noise_files, "noise")
        print(f"STATUS:Loaded {len(self.clean_audio)} clean files and {len(self.noise_audio)} noise files")

    def _load_all_audio(self, files: List[str], label: str) -> List[torch.Tensor]:
        """Load all audio files into memory."""
        audio_list = []
        for i, f in enumerate(files):
            audio = load_audio_file(f)
            if audio is not None and len(audio) >= self.chunk_samples:
                audio_list.append(audio)
            if (i + 1) % 10 == 0:
                print(f"STATUS:Loaded {i + 1}/{len(files)} {label} files")
        return audio_list

    def _random_chunk(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract a random chunk from an audio tensor."""
        if len(audio) <= self.chunk_samples:
            return F.pad(audio, (0, self.chunk_samples - len(audio)))

        start_idx = random.randint(0, len(audio) - self.chunk_samples)
        return audio[start_idx:start_idx + self.chunk_samples]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Randomly select clean and noise files
        clean_audio = random.choice(self.clean_audio)
        noise_audio = random.choice(self.noise_audio)

        # Extract random chunks
        clean_chunk = self._random_chunk(clean_audio)
        noise_chunk = self._random_chunk(noise_audio)

        # Random SNR
        snr_db = random.uniform(SNR_MIN, SNR_MAX)

        # Create mixture
        mixture = mix_at_snr(clean_chunk, noise_chunk, snr_db)

        # Compute dual-stream spectrograms
        with torch.no_grad():
            # Clean signal - dual stream
            clean_mag_a, _, clean_mag_b_low, _ = compute_dual_stream_stft(clean_chunk)

            # Mixture signal - dual stream
            mix_mag_a, _, mix_mag_b_low, _ = compute_dual_stream_stft(mixture)

            # Concatenate dual-stream features for input
            mixture_features = concatenate_dual_stream_features(mix_mag_a, mix_mag_b_low)

            # Concatenate clean magnitudes for loss computation
            clean_mag_dual = torch.cat([clean_mag_a, clean_mag_b_low], dim=1)  # (1, 257, frames)

            # Compute dual-stream ideal ratio mask (IRM)
            # Stream A mask: for highs (129 bins)
            ideal_mask_a = torch.clamp(clean_mag_a / (mix_mag_a + 1e-8), 0.0, 1.0)
            # Stream B mask: for bass (128 bins)
            ideal_mask_b = torch.clamp(clean_mag_b_low / (mix_mag_b_low + 1e-8), 0.0, 1.0)
            # Concatenate: [Stream A (129) | Stream B bass (128)] = 257 bins
            ideal_mask = torch.cat([ideal_mask_a, ideal_mask_b], dim=1)

        return mixture_features.squeeze(0), clean_mag_dual.squeeze(0), ideal_mask.squeeze(0)


# ============================================================================
# Training
# ============================================================================

class Trainer:
    """Training loop with progress reporting."""

    def __init__(self, model: MaskEstimator, train_loader: DataLoader,
                 learning_rate: float = 1e-3, device: str = 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 100
        )

        self.l1_loss = nn.L1Loss()

        # Vocal preservation weight - penalizes over-cutting
        self.preservation_weight = 0.3

    def spectral_convergence_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Spectral convergence loss for better perceptual quality."""
        return torch.norm(target - pred, p='fro') / (torch.norm(target, p='fro') + 1e-8)

    def vocal_preservation_loss(self, pred_mask: torch.Tensor, clean_mag: torch.Tensor) -> torch.Tensor:
        """
        Penalize over-cutting where clean vocal signal is strong.

        When clean_mag is high, the mask should be close to 1.0 (don't cut).
        When clean_mag is low, we don't care as much what the mask does.

        This prevents the model from learning to cut everything aggressively.
        """
        # Normalize clean magnitude to [0, 1] range for weighting
        clean_max = clean_mag.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        clean_weight = clean_mag / (clean_max + 1e-8)

        # Where clean signal is strong, mask should be close to 1.0
        # Loss = weighted mean of (1 - mask) where weight is clean energy
        over_cut = (1.0 - pred_mask) * clean_weight

        return over_cut.mean()

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, (mixture_features, clean_mag_dual, ideal_mask) in enumerate(self.train_loader):
            # mixture_features: (batch, 257, frames) - dual-stream concatenated input
            # clean_mag_dual: (batch, 257, frames) - dual-stream clean magnitudes
            # ideal_mask: (batch, 257, frames) - dual-stream mask [Stream A | Stream B bass]

            mixture_features = mixture_features.to(self.device)
            clean_mag_dual = clean_mag_dual.to(self.device)
            ideal_mask = ideal_mask.to(self.device)

            # Forward pass - predict dual-stream mask from dual-stream features
            pred_mask = self.model(mixture_features)

            # Apply predicted mask to dual-stream magnitude
            pred_clean_mag = mixture_features * pred_mask

            # Split masks for separate loss computation
            pred_mask_a = pred_mask[:, :N_FREQ_BINS_A, :]  # Stream A (129 bins)
            pred_mask_b = pred_mask[:, N_FREQ_BINS_A:, :]  # Stream B bass (128 bins)
            ideal_mask_a = ideal_mask[:, :N_FREQ_BINS_A, :]
            ideal_mask_b = ideal_mask[:, N_FREQ_BINS_A:, :]

            # Compute losses for each stream
            mask_loss_a = self.l1_loss(pred_mask_a, ideal_mask_a)
            mask_loss_b = self.l1_loss(pred_mask_b, ideal_mask_b)
            # Combined mask loss (equal weight for both streams)
            mask_loss = mask_loss_a + mask_loss_b

            # Magnitude reconstruction loss
            mag_loss = self.l1_loss(pred_clean_mag, clean_mag_dual)
            sc_loss = self.spectral_convergence_loss(pred_clean_mag, clean_mag_dual)

            # Vocal preservation loss - penalize over-cutting where vocal is strong
            pres_loss = self.vocal_preservation_loss(pred_mask, clean_mag_dual)

            # Combined loss
            # - mask_loss: match the ideal mask
            # - mag_loss: reconstructed magnitude matches clean
            # - sc_loss: spectral convergence for perceptual quality
            # - pres_loss: don't over-cut where vocal is present
            loss = mask_loss + 0.5 * mag_loss + 0.1 * sc_loss + self.preservation_weight * pres_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Progress within epoch
            if (batch_idx + 1) % max(1, n_batches // 10) == 0:
                progress = int(((epoch * n_batches + batch_idx + 1) /
                              (total_epochs * n_batches)) * 100)
                print(f"PROGRESS:{progress}")
                sys.stdout.flush()

        return total_loss / n_batches

    def train(self, epochs: int) -> dict:
        """Full training loop."""
        print("STATUS:Starting training...")
        sys.stdout.flush()

        best_loss = float('inf')
        history = {'losses': []}

        for epoch in range(epochs):
            avg_loss = self.train_epoch(epoch, epochs)
            history['losses'].append(avg_loss)

            # Report progress
            progress = int(((epoch + 1) / epochs) * 100)
            print(f"EPOCH:{epoch + 1}/{epochs}")
            print(f"LOSS:{avg_loss:.6f}")
            print(f"PROGRESS:{progress}")
            sys.stdout.flush()

            if avg_loss < best_loss:
                best_loss = avg_loss

        print("STATUS:Training complete!")
        sys.stdout.flush()

        return history


# ============================================================================
# Checkpoint Save/Load
# ============================================================================

def save_checkpoint(output_path: str, model: MaskEstimator, trainer: 'Trainer',
                    epoch: int, history: dict, clean_files: List[str], noise_files: List[str]):
    """Save a training checkpoint for continuation."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'history': history,
        'clean_files': clean_files,
        'noise_files': noise_files,
        'hidden_dim': model.hidden_dim,
        'n_input_features': model.n_input_features,
        'n_output_bins': model.n_output_bins,
        'dual_stream': True,
    }

    checkpoint_path = os.path.join(output_path, "checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"STATUS:Checkpoint saved to {checkpoint_path}")
    sys.stdout.flush()


def load_checkpoint(checkpoint_path: str, device: str) -> dict:
    """Load a training checkpoint."""
    print(f"STATUS:Loading checkpoint from {checkpoint_path}...")
    sys.stdout.flush()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


# ============================================================================
# ONNX Export
# ============================================================================

def export_to_onnx(model: MaskEstimator, output_path: str,
                   model_name: str = 'model',
                   n_input_features: int = N_TOTAL_FEATURES, n_frames: int = 128):
    """
    Export the dual-stream mask estimation model to ONNX format.

    Input: Concatenated dual-stream features (batch, 257, frames)
        - [0:129] = Stream A (256-point FFT, coarse full spectrum)
        - [129:257] = Stream B bass (2048-point FFT, fine low-frequency)

    Output: Dual mask (batch, 257, frames)
        - [0:129] = Stream A mask (highs, 187Hz resolution)
        - [129:257] = Stream B mask (bass, 23Hz resolution)

    STFT/iSTFT is handled by the C++ plugin for efficiency.
    """
    print("STATUS:Exporting dual-stream model to ONNX...")
    sys.stdout.flush()

    model.eval()
    device = next(model.parameters()).device

    # Dummy input: (batch=1, n_input_features=257, frames)
    dummy_input = torch.randn(1, n_input_features, n_frames, device=device)

    onnx_path = os.path.join(output_path, f"{model_name}.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['dual_stream_features'],
        output_names=['mask'],
        dynamic_axes={
            'dual_stream_features': {0: 'batch_size', 2: 'frames'},
            'mask': {0: 'batch_size', 2: 'frames'}
        }
    )

    print(f"STATUS:Model exported to {onnx_path}")
    sys.stdout.flush()

    return onnx_path


def save_metadata(output_path: str, model: MaskEstimator, history: dict):
    """Save model metadata as JSON for dual-stream architecture."""
    metadata = {
        'sample_rate': SAMPLE_RATE,

        # Dual-stream STFT configuration
        'dual_stream': True,
        'dual_output': True,  # NEW: Model outputs 257 bins (both streams)
        'stream_a': {
            'n_fft': N_FFT_A,
            'win_length': WIN_LENGTH_A,
            'n_freq_bins': N_FREQ_BINS_A,
            'output_bins': N_FREQ_BINS_A,  # 129 bins in output mask
            'description': 'Fast/temporal stream for transients and high frequencies',
            'resolution_hz': SAMPLE_RATE / N_FFT_A
        },
        'stream_b': {
            'n_fft': N_FFT_B,
            'win_length': WIN_LENGTH_B,
            'n_freq_bins': N_FREQ_BINS_B,
            'bins_used': STREAM_B_BINS_USED,
            'output_bins': STREAM_B_BINS_USED,  # 128 bins in output mask
            'description': 'Slow/spectral stream for bass precision',
            'resolution_hz': SAMPLE_RATE / N_FFT_B
        },
        'hop_length': HOP_LENGTH,
        'n_input_features': N_TOTAL_FEATURES,
        'n_output_bins': N_OUTPUT_BINS,  # 257 total output bins

        # Legacy compatibility
        'n_fft': N_FFT_A,
        'win_length': WIN_LENGTH_A,
        'n_freq_bins': N_FREQ_BINS_A,  # Stream A bins only (for legacy)

        'chunk_samples': CHUNK_SAMPLES,
        'latency_samples': N_FFT_B,  # Latency determined by larger FFT
        'latency_ms': (N_FFT_B / SAMPLE_RATE) * 1000,
        'hidden_dim': model.hidden_dim,
        'final_loss': history['losses'][-1] if history['losses'] else None,
        'training_epochs': len(history['losses']),
        'version': '2.0.0',
        'model_type': 'dual_stream_mask_estimator',
        'input_format': 'dual_stream_magnitude_features',
        'output_format': 'mask'
    }

    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"STATUS:Metadata saved to {metadata_path}")
    sys.stdout.flush()

    return metadata_path


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DeBleed Neural Gate Trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--clean_audio_dir',
        type=str,
        required=True,
        help='Path to directory containing clean target audio files'
    )

    parser.add_argument(
        '--noise_audio_dir',
        type=str,
        required=True,
        help='Path to directory containing noise/bleed audio files'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to output directory for model and metadata'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='model',
        help='Name for the output model file (without .onnx extension)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Training batch size'
    )

    parser.add_argument(
        '--samples_per_epoch',
        type=int,
        default=8000,
        help='Number of synthetic samples per epoch'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-4,
        help='Learning rate'
    )

    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=256,
        help='Hidden dimension of the mask estimator'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for training'
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to checkpoint file to resume from'
    )

    parser.add_argument(
        '--continue_training',
        action='store_true',
        help='Continue training from checkpoint'
    )

    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Determine the best available device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg


def main():
    """Main entry point."""
    args = parse_args()

    print("STATUS:DeBleed Neural Gate Trainer starting...")
    print(f"STATUS:Clean audio dir: {args.clean_audio_dir}")
    print(f"STATUS:Noise audio dir: {args.noise_audio_dir}")
    print(f"STATUS:Output path: {args.output_path}")
    print(f"STATUS:Epochs: {args.epochs}")
    if args.continue_training:
        print(f"STATUS:Continuing from checkpoint: {args.checkpoint_path}")
    sys.stdout.flush()

    # Determine device
    device = get_device(args.device)
    print(f"STATUS:Using device: {device}")
    sys.stdout.flush()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Check if continuing from checkpoint
    checkpoint = None
    start_epoch = 0
    history = {'losses': []}
    hidden_dim = args.hidden_dim

    if args.continue_training and args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            checkpoint = load_checkpoint(args.checkpoint_path, device)
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint.get('history', {'losses': []})
            hidden_dim = checkpoint.get('hidden_dim', args.hidden_dim)
            print(f"STATUS:Resuming from epoch {start_epoch}")
            sys.stdout.flush()

    # Collect audio files
    print("STATUS:Collecting audio files...")
    sys.stdout.flush()

    try:
        clean_files = collect_audio_files(args.clean_audio_dir)
        noise_files = collect_audio_files(args.noise_audio_dir)
    except FileNotFoundError as e:
        print(f"ERROR:{e}")
        sys.exit(1)

    # If continuing, merge with previously used files (additive training)
    if checkpoint is not None:
        prev_clean = checkpoint.get('clean_files', [])
        prev_noise = checkpoint.get('noise_files', [])

        # Add new files that aren't already in the list
        existing_clean = set(prev_clean)
        existing_noise = set(prev_noise)

        for f in clean_files:
            if f not in existing_clean:
                prev_clean.append(f)
        for f in noise_files:
            if f not in existing_noise:
                prev_noise.append(f)

        clean_files = prev_clean
        noise_files = prev_noise
        print(f"STATUS:Merged training data - now {len(clean_files)} clean, {len(noise_files)} noise files")
        sys.stdout.flush()

    if len(clean_files) == 0:
        print(f"ERROR:No audio files found in {args.clean_audio_dir}")
        sys.exit(1)

    if len(noise_files) == 0:
        print(f"ERROR:No audio files found in {args.noise_audio_dir}")
        sys.exit(1)

    print(f"STATUS:Using {len(clean_files)} clean files and {len(noise_files)} noise files")
    sys.stdout.flush()

    # Create dataset and dataloader
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

    # Create model
    print("STATUS:Creating dual-stream model...")
    sys.stdout.flush()

    # Use checkpoint dimensions if available, otherwise use current config
    n_input_features = N_TOTAL_FEATURES
    n_output_bins = N_OUTPUT_BINS  # 257 bins (dual-output)
    if checkpoint is not None:
        n_input_features = checkpoint.get('n_input_features', N_TOTAL_FEATURES)
        n_output_bins = checkpoint.get('n_output_bins', N_OUTPUT_BINS)

    model = MaskEstimator(
        n_input_features=n_input_features,
        n_output_bins=n_output_bins,
        hidden_dim=hidden_dim
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"STATUS:Model has {n_params:,} parameters")
    print(f"STATUS:Input features: {n_input_features} (Stream A: {N_FREQ_BINS_A} + Stream B bass: {STREAM_B_BINS_USED})")
    print(f"STATUS:Output bins: {n_output_bins}")
    sys.stdout.flush()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        learning_rate=args.learning_rate,
        device=device
    )

    # Load checkpoint state if continuing
    if checkpoint is not None:
        # Check if checkpoint is from an older single-stream model
        if not checkpoint.get('dual_stream', False):
            print("WARNING: Checkpoint is from single-stream model. Starting fresh with dual-stream architecture.")
            sys.stdout.flush()
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("STATUS:Loaded model and optimizer state from checkpoint")
            sys.stdout.flush()

    # Train for remaining epochs
    total_epochs = start_epoch + args.epochs
    print(f"STATUS:Training from epoch {start_epoch + 1} to {total_epochs}...")
    sys.stdout.flush()

    for epoch in range(start_epoch, total_epochs):
        avg_loss = trainer.train_epoch(epoch - start_epoch, args.epochs)
        history['losses'].append(avg_loss)

        # Report progress
        progress = int(((epoch - start_epoch + 1) / args.epochs) * 100)
        print(f"EPOCH:{epoch + 1}/{total_epochs}")
        print(f"LOSS:{avg_loss:.6f}")
        print(f"PROGRESS:{progress}")
        sys.stdout.flush()

        # Save checkpoint every 10 epochs (for resume on crash/sleep)
        if (epoch + 1) % 10 == 0:
            save_checkpoint(args.output_path, model, trainer, epoch, history,
                           clean_files, noise_files)

    print("STATUS:Training complete!")
    sys.stdout.flush()

    # Save checkpoint for future continuation
    save_checkpoint(args.output_path, model, trainer, total_epochs - 1, history,
                   clean_files, noise_files)

    # Export to ONNX
    onnx_path = export_to_onnx(model, args.output_path, args.model_name)

    # Save metadata
    metadata_path = save_metadata(args.output_path, model, history)

    print("STATUS:All done!")
    print(f"RESULT:SUCCESS")
    print(f"MODEL_PATH:{onnx_path}")
    print(f"METADATA_PATH:{metadata_path}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
