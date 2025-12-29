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

# STFT parameters optimized for low latency (<5ms at 48kHz)
# Window size of 256 samples = 5.33ms at 48kHz
# Hop size of 128 samples = 2.67ms at 48kHz
N_FFT = 256
HOP_LENGTH = 128
WIN_LENGTH = 256
N_FREQ_BINS = N_FFT // 2 + 1  # 129 frequency bins

# SNR range for synthetic mixing (in dB)
SNR_MIN = -5.0
SNR_MAX = 10.0


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
    Lightweight mask estimation network inspired by DeepFilterNet.

    Takes magnitude spectrogram as input and outputs a mask [0, 1] for each
    time-frequency bin. The mask indicates how much of the target source
    is present vs. bleed.

    Input: magnitude spectrogram (batch, freq_bins, frames)
    Output: mask (batch, freq_bins, frames) in range [0, 1]
    """

    def __init__(self, n_freq_bins: int = N_FREQ_BINS, hidden_dim: int = 64):
        super().__init__()

        self.n_freq_bins = n_freq_bins
        self.hidden_dim = hidden_dim

        # Input normalization parameters (learned)
        self.input_norm = nn.GroupNorm(1, n_freq_bins)  # Instance norm equivalent

        # Encoder - compress frequency information
        self.encoder = nn.Sequential(
            ConvBlock(n_freq_bins, hidden_dim * 2, kernel_size=5),
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

        # Decoder - expand back to frequency bins
        self.decoder = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim * 2, kernel_size=5),
            ConvBlock(hidden_dim * 2, n_freq_bins, kernel_size=5),
        )

        # Final mask output
        self.mask_output = nn.Sequential(
            nn.Conv1d(n_freq_bins, n_freq_bins, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: magnitude spectrogram (batch, freq_bins, frames)
        Output: mask (batch, freq_bins, frames) in range [0, 1]
        """
        # Normalize input
        x = self.input_norm(magnitude)

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

        # Compute spectrograms
        with torch.no_grad():
            clean_mag, _ = compute_stft(clean_chunk)
            mixture_mag, _ = compute_stft(mixture)

            # Compute ideal ratio mask (IRM)
            ideal_mask = torch.clamp(clean_mag / (mixture_mag + 1e-8), 0.0, 1.0)

        return mixture_mag.squeeze(0), clean_mag.squeeze(0), ideal_mask.squeeze(0)


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

    def spectral_convergence_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Spectral convergence loss for better perceptual quality."""
        return torch.norm(target - pred, p='fro') / (torch.norm(target, p='fro') + 1e-8)

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, (mixture_mag, clean_mag, ideal_mask) in enumerate(self.train_loader):
            mixture_mag = mixture_mag.to(self.device)
            clean_mag = clean_mag.to(self.device)
            ideal_mask = ideal_mask.to(self.device)

            # Forward pass - predict mask
            pred_mask = self.model(mixture_mag)

            # Apply predicted mask
            pred_clean_mag = mixture_mag * pred_mask

            # Compute losses
            mask_loss = self.l1_loss(pred_mask, ideal_mask)
            mag_loss = self.l1_loss(pred_clean_mag, clean_mag)
            sc_loss = self.spectral_convergence_loss(pred_clean_mag, clean_mag)

            # Combined loss
            loss = mask_loss + 0.5 * mag_loss + 0.1 * sc_loss

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
# ONNX Export
# ============================================================================

def export_to_onnx(model: MaskEstimator, output_path: str,
                   n_freq_bins: int = N_FREQ_BINS, n_frames: int = 128):
    """
    Export the mask estimation model to ONNX format.

    The exported model takes magnitude spectrogram and returns a mask.
    STFT/iSTFT is handled by the C++ plugin for efficiency.
    """
    print("STATUS:Exporting model to ONNX...")
    sys.stdout.flush()

    model.eval()
    device = next(model.parameters()).device

    # Dummy input: (batch=1, freq_bins, frames)
    dummy_input = torch.randn(1, n_freq_bins, n_frames, device=device)

    onnx_path = os.path.join(output_path, "model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['magnitude'],
        output_names=['mask'],
        dynamic_axes={
            'magnitude': {0: 'batch_size', 2: 'frames'},
            'mask': {0: 'batch_size', 2: 'frames'}
        }
    )

    print(f"STATUS:Model exported to {onnx_path}")
    sys.stdout.flush()

    return onnx_path


def save_metadata(output_path: str, model: MaskEstimator, history: dict):
    """Save model metadata as JSON."""
    metadata = {
        'sample_rate': SAMPLE_RATE,
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'win_length': WIN_LENGTH,
        'n_freq_bins': N_FREQ_BINS,
        'chunk_samples': CHUNK_SAMPLES,
        'latency_samples': N_FFT,
        'latency_ms': (N_FFT / SAMPLE_RATE) * 1000,
        'hidden_dim': model.hidden_dim,
        'final_loss': history['losses'][-1] if history['losses'] else None,
        'training_epochs': len(history['losses']),
        'version': '1.0.0',
        'model_type': 'mask_estimator',
        'input_format': 'magnitude_spectrogram',
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
        '--epochs',
        type=int,
        default=50,
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
        default=1000,
        help='Number of synthetic samples per epoch'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )

    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=64,
        help='Hidden dimension of the mask estimator'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for training'
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
    sys.stdout.flush()

    # Determine device
    device = get_device(args.device)
    print(f"STATUS:Using device: {device}")
    sys.stdout.flush()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Collect audio files
    print("STATUS:Collecting audio files...")
    sys.stdout.flush()

    try:
        clean_files = collect_audio_files(args.clean_audio_dir)
        noise_files = collect_audio_files(args.noise_audio_dir)
    except FileNotFoundError as e:
        print(f"ERROR:{e}")
        sys.exit(1)

    if len(clean_files) == 0:
        print(f"ERROR:No audio files found in {args.clean_audio_dir}")
        sys.exit(1)

    if len(noise_files) == 0:
        print(f"ERROR:No audio files found in {args.noise_audio_dir}")
        sys.exit(1)

    print(f"STATUS:Found {len(clean_files)} clean files and {len(noise_files)} noise files")
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
    print("STATUS:Creating model...")
    sys.stdout.flush()

    model = MaskEstimator(
        n_freq_bins=N_FREQ_BINS,
        hidden_dim=args.hidden_dim
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"STATUS:Model has {n_params:,} parameters")
    sys.stdout.flush()

    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        learning_rate=args.learning_rate,
        device=device
    )

    history = trainer.train(epochs=args.epochs)

    # Export to ONNX
    onnx_path = export_to_onnx(model, args.output_path)

    # Save metadata
    metadata_path = save_metadata(args.output_path, model, history)

    print("STATUS:All done!")
    print(f"RESULT:SUCCESS")
    print(f"MODEL_PATH:{onnx_path}")
    print(f"METADATA_PATH:{metadata_path}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
