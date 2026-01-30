#!/usr/bin/env python3
"""
DeBleed v2 Trainer - Vocal Preservation First
==============================================

New architecture:
- 6-band dynamic EQ (learned frequencies, VAD-gated gains)
- Simple RMS-based VAD (zero latency at runtime)
- User-controllable expander (not learned, just runtime)
- Differentiable smoothing during training

The model learns:
1. 6 problem frequencies (where bleed lives)
2. 6 Q values (how wide each problem area is)
3. 6 max cut depths (how much to cut when safe)

At runtime (zero latency):
- VAD detects vocal presence (RMS-based)
- Each band's gain = lerp(max_cut, 0dB, vad_confidence)
- Smooth interpolation prevents clicks
- Optional user expander on top

Key principle: NEVER damage the vocal. Only cut during silence.
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
import math

# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE = 48000
CHUNK_DURATION = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# Number of EQ bands to learn
N_BANDS = 6

# Number of VAD spectral bands (for learned weighting)
N_VAD_BANDS = 8

# VAD band center frequencies (Hz) - log-spaced from 200Hz to 8kHz
VAD_BAND_FREQUENCIES = [200.0, 400.0, 800.0, 1600.0, 2500.0, 4000.0, 6000.0, 8000.0]

# Frequency range for learnable bands (Hz)
FREQ_MIN = 80.0
FREQ_MAX = 12000.0

# Q range
Q_MIN = 0.5
Q_MAX = 8.0

# Max cut range (dB) - negative values only
MAX_CUT_MIN = -18.0  # Deepest allowed cut
MAX_CUT_MAX = 0.0    # No cut

# SNR range - higher = less aggressive model (preserves more vocal)
SNR_MIN = 5.0   # Vocal is louder than bleed
SNR_MAX = 20.0  # Vocal is much louder

# Smoothing time constant (samples) - matches runtime
SMOOTHING_SAMPLES = int(0.050 * SAMPLE_RATE)  # 50ms

# Frame size for processing
FRAME_SIZE = 256  # ~5.3ms at 48kHz


# ============================================================================
# Differentiable Smoothing
# ============================================================================

class DifferentiableEMA(nn.Module):
    """
    Exponential Moving Average that's differentiable for backprop.
    Matches the runtime smoothing behavior.
    """
    def __init__(self, smoothing_samples: int = SMOOTHING_SAMPLES):
        super().__init__()
        # Coefficient for exponential smoothing
        # coeff = 1 - exp(-1 / tau) where tau is time constant in samples
        self.register_buffer('coeff', torch.tensor(1.0 - math.exp(-1.0 / smoothing_samples)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal EMA smoothing along the time dimension.
        Input: (batch, features, time)
        Output: (batch, features, time) - smoothed
        """
        batch, features, time = x.shape
        output = torch.zeros_like(x)

        # Initialize with first value
        state = x[:, :, 0]
        output[:, :, 0] = state

        # Causal smoothing (can't see future)
        for t in range(1, time):
            state = self.coeff * x[:, :, t] + (1 - self.coeff) * state
            output[:, :, t] = state

        return output


# ============================================================================
# Differentiable Bandpass Filter for Spectral VAD
# ============================================================================

class DifferentiableBandpass(nn.Module):
    """
    Differentiable bandpass filter using frequency-domain multiplication.
    More efficient than IIR for training, matches IIR behavior at runtime.
    """
    def __init__(self, center_freq: float, q: float = 1.4, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.center_freq = center_freq
        self.q = q
        self.sample_rate = sample_rate

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply bandpass filter in frequency domain.
        Input: (batch, samples)
        Output: (batch, samples)
        """
        # FFT
        n_fft = audio.shape[-1]
        spectrum = torch.fft.rfft(audio)

        # Create bandpass response
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / self.sample_rate).to(audio.device)

        # Gaussian bandpass (smooth, differentiable)
        bandwidth = self.center_freq / self.q
        response = torch.exp(-0.5 * ((freqs - self.center_freq) / bandwidth) ** 2)

        # Apply filter
        filtered_spectrum = spectrum * response

        # IFFT
        return torch.fft.irfft(filtered_spectrum, n=n_fft)


class SpectralVAD(nn.Module):
    """
    Spectral Voice Activity Detection with learned frequency band weights.

    Uses bandpass filters to compute per-band energy, then applies learned
    weights to distinguish vocal timbre from bleed timbre.

    The weights are trained to be HIGH for vocal-dominant frequencies
    (formants ~800Hz-3kHz) and LOW for bleed-dominant frequencies
    (cymbals, snare bleed, etc).

    At runtime: uses IIR bandpass filters (zero latency).
    For training: uses FFT-based filters (differentiable).
    """
    def __init__(self, n_bands: int = N_VAD_BANDS,
                 band_frequencies: List[float] = VAD_BAND_FREQUENCIES,
                 frame_size: int = FRAME_SIZE):
        super().__init__()
        self.n_bands = n_bands
        self.band_frequencies = band_frequencies
        self.frame_size = frame_size

        # Create bandpass filters for each frequency band
        self.bandpass_filters = nn.ModuleList([
            DifferentiableBandpass(freq, q=1.4) for freq in band_frequencies
        ])

        # Learnable weights for each band (initialized uniform)
        # These will be trained to emphasize vocal frequencies
        self.band_weights_raw = nn.Parameter(torch.ones(n_bands))

        # Learnable threshold (in dB)
        self.threshold_db = nn.Parameter(torch.tensor(-35.0))

        # Learnable soft knee width (dB)
        self.knee_db = nn.Parameter(torch.tensor(15.0))

        # Smoothing for VAD output
        self.smoother = DifferentiableEMA(smoothing_samples=int(0.020 * SAMPLE_RATE))  # 20ms

    def get_normalized_weights(self) -> torch.Tensor:
        """Get weights normalized to sum to 1 (softmax)."""
        return F.softmax(self.band_weights_raw, dim=0)

    def compute_band_energies(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute per-band energy for input audio.
        Input: (batch, samples)
        Output: (batch, n_bands, frames)
        """
        batch, samples = audio.shape
        n_frames = samples // self.frame_size

        band_energies = []

        for bp_filter in self.bandpass_filters:
            # Apply bandpass filter
            filtered = bp_filter(audio)  # (batch, samples)

            # Compute RMS energy per frame
            filtered_frames = filtered[:, :n_frames * self.frame_size].view(batch, n_frames, self.frame_size)
            rms = torch.sqrt(torch.mean(filtered_frames ** 2, dim=-1) + 1e-8)  # (batch, frames)
            band_energies.append(rms)

        # Stack: (batch, n_bands, frames)
        return torch.stack(band_energies, dim=1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute VAD confidence from audio using spectral weighting.
        Input: (batch, samples)
        Output: (batch, 1, frames) - confidence 0-1
        """
        # Get per-band energies
        band_energies = self.compute_band_energies(audio)  # (batch, n_bands, frames)

        # Apply learned weights
        weights = self.get_normalized_weights()  # (n_bands,)
        weighted_energy = torch.sum(band_energies * weights.view(1, -1, 1), dim=1)  # (batch, frames)

        # Convert to dB
        energy_db = 20.0 * torch.log10(weighted_energy + 1e-8)

        # Soft threshold with sigmoid (differentiable)
        confidence = torch.sigmoid((energy_db - self.threshold_db) / (self.knee_db.abs() + 0.1))

        # Add channel dimension and smooth
        confidence = confidence.unsqueeze(1)  # (batch, 1, frames)
        confidence = self.smoother(confidence)

        return confidence

    def compute_discrimination_loss(self, clean_audio: torch.Tensor,
                                     noise_audio: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary loss to train weights to discriminate vocal from bleed.

        We want: high weighted energy for vocal, low for bleed.
        Loss = -log(vocal_energy / (bleed_energy + eps))

        This encourages weights that are high for vocal-dominant bands
        and low for bleed-dominant bands.
        """
        # Compute band energies for clean and noise
        clean_energies = self.compute_band_energies(clean_audio)  # (batch, n_bands, frames)
        noise_energies = self.compute_band_energies(noise_audio)  # (batch, n_bands, frames)

        # Average across frames
        clean_avg = clean_energies.mean(dim=-1)  # (batch, n_bands)
        noise_avg = noise_energies.mean(dim=-1)  # (batch, n_bands)

        # Apply weights
        weights = self.get_normalized_weights()  # (n_bands,)
        clean_weighted = torch.sum(clean_avg * weights, dim=1)  # (batch,)
        noise_weighted = torch.sum(noise_avg * weights, dim=1)  # (batch,)

        # Discrimination loss: want high clean/noise ratio
        # Use log ratio for numerical stability
        discrimination = torch.log(clean_weighted + 1e-6) - torch.log(noise_weighted + 1e-6)

        # We want to maximize discrimination, so return negative
        return -discrimination.mean()

    def export_weights(self) -> dict:
        """Export learned weights for C++ runtime."""
        weights = self.get_normalized_weights().detach().cpu().numpy().tolist()
        return {
            'band_frequencies': self.band_frequencies,
            'band_weights': weights,
            'threshold_db': self.threshold_db.item(),
            'knee_db': abs(self.knee_db.item()),
        }


# ============================================================================
# Differentiable Biquad EQ
# ============================================================================

class DifferentiableBiquadBand(nn.Module):
    """
    Single parametric EQ band with differentiable coefficient computation.

    Learns: center frequency, Q, max cut depth
    At runtime: gain is modulated by VAD (0dB when vocal, max_cut when silent)
    """
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate

        # Learnable parameters (normalized 0-1, denormalized in forward)
        self.freq_norm = nn.Parameter(torch.tensor(0.5))  # -> FREQ_MIN to FREQ_MAX
        self.q_norm = nn.Parameter(torch.tensor(0.5))     # -> Q_MIN to Q_MAX
        self.max_cut_norm = nn.Parameter(torch.tensor(0.5))  # -> MAX_CUT_MIN to MAX_CUT_MAX

    def get_freq(self) -> torch.Tensor:
        """Get denormalized frequency (Hz)"""
        # Log scale for frequency
        log_min = math.log(FREQ_MIN)
        log_max = math.log(FREQ_MAX)
        return torch.exp(log_min + torch.sigmoid(self.freq_norm) * (log_max - log_min))

    def get_q(self) -> torch.Tensor:
        """Get denormalized Q"""
        # Log scale for Q
        log_min = math.log(Q_MIN)
        log_max = math.log(Q_MAX)
        return torch.exp(log_min + torch.sigmoid(self.q_norm) * (log_max - log_min))

    def get_max_cut(self) -> torch.Tensor:
        """Get denormalized max cut (dB, negative)"""
        # Linear scale, clamped negative
        return MAX_CUT_MIN + torch.sigmoid(self.max_cut_norm) * (MAX_CUT_MAX - MAX_CUT_MIN)

    def compute_coefficients(self, gain_db: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute biquad coefficients for peaking EQ.

        Input: gain_db (batch, frames) - current gain in dB
        Output: b0, b1, b2, a1, a2 coefficients (batch, frames)
        """
        freq = self.get_freq()
        q = self.get_q()

        # Normalize frequency
        w0 = 2.0 * math.pi * freq / self.sample_rate

        # Compute intermediate values
        A = torch.pow(10.0, gain_db / 40.0)  # sqrt of linear gain
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        alpha = sin_w0 / (2.0 * q)

        # Peaking EQ coefficients
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A

        # Normalize
        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0

        return b0, b1, b2, a1, a2


class DifferentiableDynamicEQ(nn.Module):
    """
    6-band dynamic EQ where gains are modulated by VAD confidence.

    When VAD=1 (vocal present): all bands at 0dB (transparent)
    When VAD=0 (silence): bands at their learned max_cut depths
    """
    def __init__(self, n_bands: int = N_BANDS, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.n_bands = n_bands
        self.sample_rate = sample_rate

        # Create learnable bands
        self.bands = nn.ModuleList([
            DifferentiableBiquadBand(sample_rate) for _ in range(n_bands)
        ])

        # Smoothing for gain changes
        self.gain_smoother = DifferentiableEMA(smoothing_samples=SMOOTHING_SAMPLES)

    def forward(self, audio: torch.Tensor, vad_confidence: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic EQ to audio based on VAD confidence.

        Input:
            audio: (batch, samples)
            vad_confidence: (batch, 1, frames) - 0=silence, 1=vocal
        Output:
            processed: (batch, samples)
        """
        batch, samples = audio.shape
        n_frames = vad_confidence.shape[-1]
        frame_size = samples // n_frames

        # Process each band
        output = audio.clone()

        for band in self.bands:
            # Compute gain per frame: lerp between max_cut (silence) and 0dB (vocal)
            max_cut = band.get_max_cut()  # scalar, dB (negative)

            # gain_db = vad * 0 + (1 - vad) * max_cut = (1 - vad) * max_cut
            gain_db = (1.0 - vad_confidence) * max_cut  # (batch, 1, frames)

            # Smooth the gain changes
            gain_db = self.gain_smoother(gain_db)  # (batch, 1, frames)

            # Upsample gain to sample rate (simple repeat)
            gain_db_upsampled = gain_db.repeat_interleave(frame_size, dim=-1)  # (batch, 1, samples)
            gain_db_upsampled = gain_db_upsampled[:, 0, :samples]  # (batch, samples)

            # Apply time-varying biquad filter
            # For efficiency, we process in frames with constant gain
            output = self._apply_biquad_timevarying(output, band, gain_db_upsampled)

        return output

    def _apply_biquad_timevarying(self, audio: torch.Tensor, band: DifferentiableBiquadBand,
                                   gain_db: torch.Tensor) -> torch.Tensor:
        """
        Apply biquad filter with time-varying gain.

        For differentiability, we use a sample-by-sample implementation.
        This is slow but correct for training. Runtime uses optimized C++.
        """
        batch, samples = audio.shape

        # Get fixed parameters
        freq = band.get_freq()
        q = band.get_q()

        # Pre-compute fixed parts
        w0 = 2.0 * math.pi * freq / self.sample_rate
        cos_w0 = math.cos(w0.item())
        sin_w0 = math.sin(w0.item())

        output = torch.zeros_like(audio)

        # State variables (per batch)
        x1 = torch.zeros(batch, device=audio.device)
        x2 = torch.zeros(batch, device=audio.device)
        y1 = torch.zeros(batch, device=audio.device)
        y2 = torch.zeros(batch, device=audio.device)

        # Process sample by sample (slow but differentiable)
        for n in range(samples):
            # Get gain for this sample
            g = gain_db[:, n]  # (batch,)

            # Compute A from gain
            A = torch.pow(10.0, g / 40.0)  # (batch,)

            # Compute coefficients
            alpha = sin_w0 / (2.0 * q.item())

            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_w0
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha / A

            # Normalize
            b0 = b0 / a0
            b1 = b1 / a0
            b2 = b2 / a0
            a1 = a1 / a0
            a2 = a2 / a0

            # Apply filter
            x0 = audio[:, n]
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

            output[:, n] = y0

            # Update state
            x2 = x1
            x1 = x0
            y2 = y1
            y1 = y0

        return output

    def get_band_params(self) -> List[dict]:
        """Get current band parameters for export."""
        params = []
        for i, band in enumerate(self.bands):
            params.append({
                'freq_hz': band.get_freq().item(),
                'q': band.get_q().item(),
                'max_cut_db': band.get_max_cut().item(),
            })
        return params


# ============================================================================
# Combined Model
# ============================================================================

class DeBleedV2Model(nn.Module):
    """
    Complete DeBleed v2 model:
    - Spectral VAD with learned frequency weights detects vocal presence
    - Dynamic EQ cuts only when safe (VAD says silence)
    - All processing is differentiable for training
    - Zero latency at runtime
    """
    def __init__(self, n_bands: int = N_BANDS, n_vad_bands: int = N_VAD_BANDS,
                 sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.vad = SpectralVAD(n_bands=n_vad_bands,
                               band_frequencies=VAD_BAND_FREQUENCIES,
                               frame_size=FRAME_SIZE)
        self.dynamic_eq = DifferentiableDynamicEQ(n_bands=n_bands, sample_rate=sample_rate)

    def forward(self, audio: torch.Tensor, return_vad: bool = False):
        """
        Process audio through VAD-gated dynamic EQ.

        Input: audio (batch, samples)
        Output: processed (batch, samples), optionally vad_confidence
        """
        # Get VAD confidence (uses spectral weighting)
        vad_confidence = self.vad(audio)  # (batch, 1, frames)

        # Apply dynamic EQ
        processed = self.dynamic_eq(audio, vad_confidence)

        if return_vad:
            return processed, vad_confidence
        return processed

    def compute_vad_discrimination_loss(self, clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss to train VAD weights."""
        return self.vad.compute_discrimination_loss(clean, noise)

    def export_params(self) -> dict:
        """Export learned parameters for C++ runtime."""
        vad_params = self.vad.export_weights()
        return {
            'spectral_vad': vad_params,  # Includes band_frequencies, band_weights, threshold_db, knee_db
            'dynamic_eq_bands': self.dynamic_eq.get_band_params(),
            'smoothing_samples': SMOOTHING_SAMPLES,
            'frame_size': FRAME_SIZE,
            'sample_rate': SAMPLE_RATE,
        }


# ============================================================================
# Dataset
# ============================================================================

def load_audio_file(path: str, target_sr: int = SAMPLE_RATE) -> Optional[torch.Tensor]:
    """Load audio file and resample to target sample rate."""
    try:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0)
    except Exception as e:
        print(f"WARNING: Failed to load {path}: {e}", file=sys.stderr)
        return None


def collect_audio_files(directory: str) -> List[str]:
    """Collect all supported audio files from directory."""
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
    """Compute RMS energy."""
    return torch.sqrt(torch.mean(signal ** 2) + 1e-8)


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix clean signal with noise at specified SNR."""
    clean_rms = compute_rms(clean)
    noise_rms = compute_rms(noise)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / (noise_rms + 1e-8))
    return clean + noise_scaled


class DeBleedDataset(Dataset):
    """Dataset that creates synthetic mixtures for training."""

    def __init__(self, clean_files: List[str], noise_files: List[str],
                 samples_per_epoch: int = 1000, chunk_samples: int = CHUNK_SAMPLES):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.samples_per_epoch = samples_per_epoch
        self.chunk_samples = chunk_samples

        print("STATUS:Loading audio files...")
        self.clean_audio = self._load_all_audio(clean_files, "clean")
        self.noise_audio = self._load_all_audio(noise_files, "noise")
        print(f"STATUS:Loaded {len(self.clean_audio)} clean, {len(self.noise_audio)} noise files")

    def _load_all_audio(self, files: List[str], label: str) -> List[torch.Tensor]:
        audio_list = []
        for i, f in enumerate(files):
            audio = load_audio_file(f)
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean = self._random_chunk(random.choice(self.clean_audio))
        noise = self._random_chunk(random.choice(self.noise_audio))
        snr_db = random.uniform(SNR_MIN, SNR_MAX)
        mixture = mix_at_snr(clean, noise, snr_db)
        return mixture, clean, noise


# ============================================================================
# Training
# ============================================================================

class Trainer:
    """Trainer with vocal-preservation-first loss and spectral VAD learning."""

    def __init__(self, model: DeBleedV2Model, train_loader: DataLoader,
                 learning_rate: float = 1e-3, device: str = 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 100
        )

        # Loss weights - VOCAL PRESERVATION IS #1 PRIORITY
        self.w_preservation = 1.0    # Highest weight
        self.w_bleed_reduction = 0.1  # Low weight - only cut when safe
        self.w_sparsity = 0.05       # Encourage minimal cuts
        self.w_vad_discrimination = 0.3  # Train VAD to distinguish vocal from bleed

    def compute_loss(self, mixture: torch.Tensor, clean: torch.Tensor,
                     noise: torch.Tensor, processed: torch.Tensor,
                     vad_confidence: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute vocal-preservation-first loss with VAD discrimination.

        Key insight: We want to preserve the vocal ALWAYS, and only
        reduce bleed when VAD says it's safe (confidence < threshold).
        Additionally, we train the VAD weights to maximize vocal/bleed discrimination.
        """
        # 1. VOCAL PRESERVATION LOSS (most important)
        # When VAD is high (vocal present), output should match original mixture
        # This prevents ANY damage to the vocal
        vad_weight = vad_confidence.mean(dim=-1, keepdim=True)  # (batch, 1, 1)
        vad_weight = vad_weight.squeeze()  # (batch,)

        # L1 loss between processed and mixture, weighted by VAD
        # High VAD = heavy penalty for changing the signal
        preservation_loss = torch.mean(
            vad_weight.unsqueeze(-1) * torch.abs(processed - mixture)
        )

        # 2. BLEED REDUCTION LOSS (secondary)
        # When VAD is low (silence), processed should be closer to clean than mixture was
        # But we weight this much lower
        silence_weight = 1.0 - vad_weight  # (batch,)

        # Only encourage bleed reduction during silence
        mixture_to_clean = torch.abs(mixture - clean).mean(dim=-1)  # (batch,)
        processed_to_clean = torch.abs(processed - clean).mean(dim=-1)  # (batch,)

        # Reward if processed is closer to clean than mixture was
        improvement = mixture_to_clean - processed_to_clean  # positive = good
        bleed_loss = -torch.mean(silence_weight * improvement)  # negative because we want to maximize improvement

        # 3. SPARSITY LOSS - encourage minimal cuts
        # Penalize large max_cut values
        max_cuts = torch.stack([band.get_max_cut() for band in self.model.dynamic_eq.bands])
        sparsity_loss = torch.mean(torch.abs(max_cuts))  # Smaller cuts = lower loss

        # 4. VAD DISCRIMINATION LOSS - train spectral weights to distinguish vocal from bleed
        # This is the key addition for spectral VAD
        vad_discrimination_loss = self.model.compute_vad_discrimination_loss(clean, noise)

        # Combined loss
        total_loss = (
            self.w_preservation * preservation_loss +
            self.w_bleed_reduction * bleed_loss +
            self.w_sparsity * sparsity_loss +
            self.w_vad_discrimination * vad_discrimination_loss
        )

        metrics = {
            'preservation': preservation_loss.item(),
            'bleed_reduction': bleed_loss.item(),
            'sparsity': sparsity_loss.item(),
            'vad_discrimination': vad_discrimination_loss.item(),
            'total': total_loss.item(),
        }

        return total_loss, metrics

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, (mixture, clean, noise) in enumerate(self.train_loader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)
            noise = noise.to(self.device)

            # Forward pass
            processed, vad_confidence = self.model(mixture, return_vad=True)

            # Compute loss
            loss, metrics = self.compute_loss(mixture, clean, noise, processed, vad_confidence)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Progress reporting
            if (batch_idx + 1) % max(1, n_batches // 10) == 0:
                progress = int(((epoch * n_batches + batch_idx + 1) / (total_epochs * n_batches)) * 100)
                print(f"PROGRESS:{progress}")
                sys.stdout.flush()

        return total_loss / n_batches

    def train(self, epochs: int) -> dict:
        print("STATUS:Starting training...")
        sys.stdout.flush()

        history = {'losses': []}

        for epoch in range(epochs):
            avg_loss = self.train_epoch(epoch, epochs)
            history['losses'].append(avg_loss)

            # Report progress
            print(f"EPOCH:{epoch + 1}/{epochs}")
            print(f"LOSS:{avg_loss:.6f}")

            # Print learned parameters
            params = self.model.export_params()

            # VAD spectral weights
            vad_params = params['spectral_vad']
            print(f"STATUS:VAD threshold: {vad_params['threshold_db']:.1f}dB, knee: {vad_params['knee_db']:.1f}dB")
            weights_str = ", ".join([f"{w:.2f}" for w in vad_params['band_weights']])
            freqs_str = ", ".join([f"{int(f)}Hz" for f in vad_params['band_frequencies']])
            print(f"STATUS:VAD bands: {freqs_str}")
            print(f"STATUS:VAD weights: {weights_str}")

            # Dynamic EQ bands
            for i, band in enumerate(params['dynamic_eq_bands']):
                print(f"STATUS:EQ Band {i+1}: {band['freq_hz']:.0f}Hz, Q={band['q']:.1f}, cut={band['max_cut_db']:.1f}dB")

            sys.stdout.flush()

        print("STATUS:Training complete!")
        return history


# ============================================================================
# Export
# ============================================================================

def export_model(model: DeBleedV2Model, output_path: str, model_name: str = 'debleed_v2'):
    """Export learned parameters to JSON for C++ runtime."""
    params = model.export_params()

    # Save as JSON
    json_path = os.path.join(output_path, f"{model_name}_params.json")
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"STATUS:Parameters exported to {json_path}")
    print(f"MODEL_PATH:{json_path}")

    # Also save PyTorch checkpoint for future training
    checkpoint_path = os.path.join(output_path, f"{model_name}_checkpoint.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': params,
    }, checkpoint_path)

    print(f"STATUS:Checkpoint saved to {checkpoint_path}")

    return json_path


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='DeBleed v2 Trainer')
    parser.add_argument('--clean_audio_dir', type=str, required=True)
    parser.add_argument('--noise_audio_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='debleed_v2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--samples_per_epoch', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
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

    print("STATUS:DeBleed v2 Trainer (Vocal Preservation First)")
    print(f"STATUS:Clean audio: {args.clean_audio_dir}")
    print(f"STATUS:Noise audio: {args.noise_audio_dir}")
    print(f"STATUS:Output: {args.output_path}")
    sys.stdout.flush()

    device = get_device(args.device)
    print(f"STATUS:Using device: {device}")

    os.makedirs(args.output_path, exist_ok=True)

    # Collect audio files
    clean_files = collect_audio_files(args.clean_audio_dir)
    noise_files = collect_audio_files(args.noise_audio_dir)

    if not clean_files or not noise_files:
        print("ERROR:No audio files found")
        sys.exit(1)

    # Create dataset
    dataset = DeBleedDataset(
        clean_files=clean_files,
        noise_files=noise_files,
        samples_per_epoch=args.samples_per_epoch
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Create model
    print(f"STATUS:Creating model with {N_BANDS} bands...")
    model = DeBleedV2Model(n_bands=N_BANDS)

    # Train
    trainer = Trainer(model, train_loader, learning_rate=args.learning_rate, device=device)
    history = trainer.train(args.epochs)

    # Export
    export_model(model, args.output_path, args.model_name)

    print("STATUS:All done!")
    print("RESULT:SUCCESS")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
