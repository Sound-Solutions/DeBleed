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

# Redirect stderr to stdout so plugin captures all output
sys.stderr = sys.stdout
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as tF

# Import SVF-based filter chain that matches C++ plugin exactly
from differentiable_svf import DifferentiableBiquadChain

# ============================================================================
# Configuration Constants
# ============================================================================

SAMPLE_RATE = 96000          # High quality sample rate for training
CHUNK_DURATION = 1.0         # 1.0-second chunks (balance of speed + quality)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# Frame rate for parameter updates
FRAME_SIZE = 2048            # Update coefficients every 2048 samples (~21ms @ 96kHz) - musical rate

# Filter bank configuration
N_BIQUADS = 16               # Total number of biquad stages
N_PARAMS_PER_BIQUAD = 3      # freq, gain, Q
N_EXTRA_PARAMS = 2           # input gain, output gain
N_TOTAL_PARAMS = N_BIQUADS * N_PARAMS_PER_BIQUAD + N_EXTRA_PARAMS  # 50 params


# ============================================================================
# Progress Reporter (file-based for reliable plugin communication)
# ============================================================================

class ProgressReporter:
    """
    Writes progress to both stdout and a JSON file.
    The plugin reads the JSON file for reliable progress updates.
    Also writes to a log file for detailed message history.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, "training_progress.json")
        self.log_file = os.path.join(output_dir, "training_log.txt")
        self.state = {
            "status": "initializing",
            "progress": 0,
            "epoch": 0,
            "total_epochs": 0,
            "loss": 0.0,
            "error": "",
            "model_path": ""
        }
        # Clear log file on start
        with open(self.log_file, 'w') as f:
            f.write("")

    def update(self, status: str = None, progress: int = None, epoch: int = None,
               total_epochs: int = None, loss: float = None, error: str = None,
               model_path: str = None):
        """Update progress state and write to file."""
        if status is not None:
            self.state["status"] = status
        if progress is not None:
            self.state["progress"] = progress
        if epoch is not None:
            self.state["epoch"] = epoch
        if total_epochs is not None:
            self.state["total_epochs"] = total_epochs
        if loss is not None:
            self.state["loss"] = loss
        if error is not None:
            self.state["error"] = error
        if model_path is not None:
            self.state["model_path"] = model_path

        # Write to file atomically (write to temp, then rename)
        try:
            temp_file = self.progress_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.state, f)
            os.replace(temp_file, self.progress_file)
        except Exception as e:
            pass  # Don't let file writing errors stop training

        # Also print to stdout and log file for terminal/plugin usage
        if status:
            print(f"STATUS:{status}")
            sys.stdout.flush()
            self._write_log(f"STATUS:{status}")

    def log(self, message: str):
        """Log a message to stdout and log file."""
        print(message)
        sys.stdout.flush()
        self._write_log(message)

    def _write_log(self, message: str):
        """Append message to log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(message + "\n")
        except:
            pass

    def set_error(self, error: str):
        """Set error state."""
        self.update(status=f"Error: {error}", error=error)
        print(f"ERROR:{error}")
        sys.stdout.flush()

    def set_complete(self, model_path: str):
        """Set completion state."""
        self.update(status="complete", progress=100, model_path=model_path)
        print(f"MODEL_PATH:{model_path}")
        print("RESULT:SUCCESS")
        sys.stdout.flush()


# Global progress reporter (initialized in main)
_progress_reporter: Optional[ProgressReporter] = None

def get_reporter() -> Optional[ProgressReporter]:
    return _progress_reporter

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

# SNR range for synthetic mixing (in dB) - wider range for real-world scenarios
# Negative SNR = noise louder than signal (challenging cases)
SNR_MIN = -5.0
SNR_MAX = 30.0

# Data augmentation: random gain on clean audio (dB)
AUGMENT_GAIN_MIN = -6.0
AUGMENT_GAIN_MAX = 6.0

# Validation split ratio
VALIDATION_SPLIT = 0.1  # 10% of files for validation


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


# ============================================================================
# Note: Old DifferentiableBiquad and bilinear_biquad_* functions removed.
# Now using DifferentiableBiquadChain from differentiable_svf.py which
# implements SVF TPT filters matching the C++ plugin exactly.
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
    """Dataset that creates synthetic mixtures at 96kHz with data augmentation."""

    def __init__(self, clean_files: List[str], noise_files: List[str],
                 samples_per_epoch: int = 2000, chunk_samples: int = CHUNK_SAMPLES,
                 augment: bool = True):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.samples_per_epoch = samples_per_epoch
        self.chunk_samples = chunk_samples
        self.augment = augment

        reporter = get_reporter()
        if reporter:
            reporter.log("STATUS:Loading audio files into memory...")
        else:
            print("STATUS:Loading audio files into memory...")
            sys.stdout.flush()

        self.clean_audio = self._load_all_audio(clean_files, "clean")
        self.noise_audio = self._load_all_audio(noise_files, "noise")

        msg = f"STATUS:Loaded {len(self.clean_audio)} clean files and {len(self.noise_audio)} noise files"
        if reporter:
            reporter.log(msg)
        else:
            print(msg)
            sys.stdout.flush()

    def _load_all_audio(self, files: List[str], label: str) -> List[torch.Tensor]:
        reporter = get_reporter()
        audio_list = []
        for i, f in enumerate(files):
            audio = load_audio_file(f, SAMPLE_RATE)
            if audio is not None and len(audio) >= self.chunk_samples:
                audio_list.append(audio)
            if (i + 1) % 10 == 0:
                msg = f"STATUS:Loaded {i + 1}/{len(files)} {label} files"
                if reporter:
                    reporter.log(msg)
                else:
                    print(msg)
                    sys.stdout.flush()
        return audio_list

    def _random_chunk(self, audio: torch.Tensor, max_retries: int = 10) -> torch.Tensor:
        """Select random chunk, preferring non-silent regions."""
        if len(audio) <= self.chunk_samples:
            return F.pad(audio, (0, self.chunk_samples - len(audio)))

        silence_threshold = 1e-3  # ~-60dB (safe: silence is -200dB, quiet vocals are -50dB)

        for _ in range(max_retries):
            start_idx = random.randint(0, len(audio) - self.chunk_samples)
            chunk = audio[start_idx:start_idx + self.chunk_samples]
            rms = torch.sqrt(torch.mean(chunk ** 2))
            if rms > silence_threshold:
                return chunk

        return chunk  # Fallback to last attempted chunk

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_audio = random.choice(self.clean_audio)
        noise_audio = random.choice(self.noise_audio)

        clean_chunk = self._random_chunk(clean_audio)
        noise_chunk = self._random_chunk(noise_audio)

        # Data augmentation: random gain on clean audio
        if self.augment:
            gain_db = random.uniform(AUGMENT_GAIN_MIN, AUGMENT_GAIN_MAX)
            gain_linear = 10 ** (gain_db / 20)
            clean_chunk = clean_chunk * gain_linear

        snr_db = random.uniform(SNR_MIN, SNR_MAX)
        mixture = mix_at_snr(clean_chunk, noise_chunk, snr_db)

        return mixture, clean_chunk


# ============================================================================
# Training
# ============================================================================

class Trainer:
    """Training loop for Neural 5045."""

    def __init__(self, model: Neural5045Net, filter_chain: DifferentiableBiquadChain,
                 train_loader: DataLoader, val_loader: DataLoader = None,
                 learning_rate: float = 1e-4, device: str = 'cpu'):
        # Hybrid device setup: neural network on GPU (if available), filter chain on CPU
        # This is because torchaudio.functional.lfilter is 530x slower on MPS than CPU
        self.nn_device = device
        self.filter_device = 'cpu'  # Always CPU for filter chain

        # Check if we can use MPS for neural network
        if device == 'cpu' and torch.backends.mps.is_available():
            self.nn_device = 'mps'
            print("STATUS:Using hybrid mode: Neural network on MPS, filters on CPU")
            sys.stdout.flush()

        self.model = model.to(self.nn_device)
        self.filter_chain = filter_chain.to(self.filter_device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device  # Keep for compatibility

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 100
        )

        self.stft_loss = MultiResolutionSTFTLoss().to(self.filter_device)

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        """Train for one epoch with hybrid device support."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, (mixture, clean) in enumerate(self.train_loader):
            # Keep clean on filter device (CPU) for loss computation
            clean = clean.to(self.filter_device)

            # Move mixture to neural network device for parameter prediction
            mixture_nn = mixture.to(self.nn_device)

            # Predict parameters from mixture (on MPS if available)
            params = self.model(mixture_nn)  # (batch, 50, n_frames)

            # Move params back to filter device (keep per-frame for dynamic EQ)
            params_cpu = params.to(self.filter_device)
            mixture_cpu = mixture.to(self.filter_device)

            # Reset filter states and apply filter chain with per-frame params (on CPU)
            self.filter_chain.reset_state(mixture_cpu.shape[0], self.filter_device)
            filtered = self.filter_chain(mixture_cpu, params_cpu)

            # Compute loss (on CPU)
            loss = self.stft_loss(filtered, clean)

            # Backward - gradients flow back through the device transfer
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Progress - update every 20 batches or 5% of epoch
            if (batch_idx + 1) % max(1, min(20, n_batches // 20)) == 0:
                progress = int(((epoch * n_batches + batch_idx + 1) /
                              (total_epochs * n_batches)) * 100)
                avg_loss = total_loss / (batch_idx + 1)
                print(f"PROGRESS:{progress}")
                print(f"LOSS:{avg_loss:.6f}")
                print(f"STATUS:Epoch {epoch+1}/{total_epochs} batch {batch_idx+1}/{n_batches}")
                sys.stdout.flush()

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss."""
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for mixture, clean in self.val_loader:
            clean = clean.to(self.filter_device)
            mixture_nn = mixture.to(self.nn_device)

            params = self.model(mixture_nn)
            params_cpu = params.to(self.filter_device)
            mixture_cpu = mixture.to(self.filter_device)

            self.filter_chain.reset_state(mixture_cpu.shape[0], self.filter_device)
            filtered = self.filter_chain(mixture_cpu, params_cpu)

            loss = self.stft_loss(filtered, clean)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0


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
    # Use 3D input [batch, channels, samples] to match C++ plugin expectations
    n_samples = CHUNK_SAMPLES
    dummy_input = torch.randn(1, 1, n_samples, device=device)

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
            'audio': {0: 'batch_size', 2: 'samples'},
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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--samples_per_epoch', type=int, default=1000,
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
        # NOTE: MPS (Apple Metal) is extremely slow for lfilter operations
        # (530x slower than CPU), so we default to CPU for this model
        return 'cpu'
    return device_arg


def main():
    global _progress_reporter
    args = parse_args()

    # Create output directory first so we can write progress file
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize progress reporter
    _progress_reporter = ProgressReporter(args.output_path)
    reporter = _progress_reporter

    reporter.update(status="Neural 5045 Trainer starting...")
    reporter.update(status=f"Sample rate: {SAMPLE_RATE}Hz")
    reporter.update(status=f"Parameters per frame: {N_TOTAL_PARAMS}")

    device = get_device(args.device)
    reporter.update(status=f"Using device: {device}")

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
            reporter.update(status=f"Resuming from epoch {start_epoch}")

    # Collect audio files
    reporter.update(status="Collecting audio files...")
    try:
        clean_files = collect_audio_files(args.clean_audio_dir)
        noise_files = collect_audio_files(args.noise_audio_dir)
    except FileNotFoundError as e:
        reporter.set_error(str(e))
        sys.exit(1)

    if len(clean_files) == 0 or len(noise_files) == 0:
        reporter.set_error("No audio files found")
        sys.exit(1)

    reporter.update(status=f"Found {len(clean_files)} clean, {len(noise_files)} noise files")

    # Split clean files into train/validation sets
    random.shuffle(clean_files)
    n_val = max(1, int(len(clean_files) * VALIDATION_SPLIT))
    val_clean_files = clean_files[:n_val]
    train_clean_files = clean_files[n_val:]
    reporter.update(status=f"Split: {len(train_clean_files)} train, {len(val_clean_files)} val clean files")

    # Create training dataset (this loads all audio into memory)
    reporter.update(status=f"Loading {len(train_clean_files) + len(noise_files)} audio files into memory...")
    train_dataset = SyntheticMixtureDataset(
        clean_files=train_clean_files,
        noise_files=noise_files,
        samples_per_epoch=args.samples_per_epoch,
        augment=True  # Data augmentation for training
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )

    # Create validation dataset (no augmentation)
    reporter.update(status=f"Loading {len(val_clean_files)} validation files...")
    val_dataset = SyntheticMixtureDataset(
        clean_files=val_clean_files,
        noise_files=noise_files,
        samples_per_epoch=max(100, args.samples_per_epoch // 10),  # Smaller validation set
        augment=False  # No augmentation for validation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )

    # Create model and filter chain
    reporter.update(status="Creating Neural 5045 model...")
    model = Neural5045Net(hidden_dim=hidden_dim)
    filter_chain = DifferentiableBiquadChain(sample_rate=SAMPLE_RATE)

    n_params = sum(p.numel() for p in model.parameters())
    reporter.update(status=f"Model has {n_params:,} parameters")

    # Estimate training time
    batches_per_epoch = args.samples_per_epoch // args.batch_size
    total_batches = batches_per_epoch * args.epochs
    total_epochs = start_epoch + args.epochs
    reporter.update(status=f"Training {args.epochs} epochs x {batches_per_epoch} batches",
                   total_epochs=total_epochs)
    reporter.update(status="Pro quality training - this may take 1-2+ hours...")

    # Create trainer
    trainer = Trainer(
        model=model,
        filter_chain=filter_chain,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=device
    )

    # Load checkpoint state
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        reporter.update(status="Loaded checkpoint state")

    # Training loop
    reporter.update(status=f"Training from epoch {start_epoch + 1} to {total_epochs}...")

    for epoch in range(start_epoch, total_epochs):
        progress = int(((epoch - start_epoch) / args.epochs) * 100)
        reporter.update(status=f"Epoch {epoch + 1}/{total_epochs}",
                       epoch=epoch + 1, progress=progress)

        avg_loss = trainer.train_epoch(epoch - start_epoch, args.epochs)
        history['losses'].append(avg_loss)

        # Run validation
        val_loss = trainer.validate()
        if 'val_losses' not in history:
            history['val_losses'] = []
        history['val_losses'].append(val_loss)

        progress = int(((epoch - start_epoch + 1) / args.epochs) * 100)
        reporter.update(status=f"Epoch {epoch + 1}/{total_epochs} - Train: {avg_loss:.4f}, Val: {val_loss:.4f}",
                       epoch=epoch + 1, progress=progress, loss=avg_loss)

        # Also print legacy format for stdout parsing
        print(f"EPOCH:{epoch + 1}/{total_epochs}")
        print(f"LOSS:{avg_loss:.6f}")
        print(f"VAL_LOSS:{val_loss:.6f}")
        print(f"PROGRESS:{progress}")
        sys.stdout.flush()

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(args.output_path, model, trainer, epoch, history)
            reporter.update(status=f"Checkpoint saved at epoch {epoch + 1}")

    reporter.update(status="Training complete! Exporting model...")

    # Final checkpoint
    save_checkpoint(args.output_path, model, trainer, total_epochs - 1, history)

    # Export
    onnx_path = export_to_onnx(model, args.output_path, args.model_name)
    save_metadata(args.output_path, model, history)

    reporter.set_complete(onnx_path)


if __name__ == '__main__':
    main()
