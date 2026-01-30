#!/usr/bin/env python3
"""
Differentiable SVF Filter Chain - V2 using lfilter
===================================================
Uses torchaudio.functional.lfilter for vectorized processing.
Converts SVF TPT coefficients to Direct Form II for lfilter compatibility.
"""

import torch
import torch.nn as nn
import torchaudio.functional as AF
import math
from typing import Tuple

# Configuration matching C++
N_BIQUADS = 16
N_PARAMS_PER_BIQUAD = 3
N_EXTRA_PARAMS = 2
N_TOTAL_PARAMS = N_BIQUADS * N_PARAMS_PER_BIQUAD + N_EXTRA_PARAMS

# Filter indices
BIQUAD_HPF = 0
BIQUAD_LOW_SHELF = 1
BIQUAD_PEAKING_START = 2
BIQUAD_PEAKING_END = 13
BIQUAD_HIGH_SHELF = 14
BIQUAD_LPF = 15

# Ranges
HPF_FREQ_RANGE = (20.0, 500.0)
LPF_FREQ_RANGE = (5000.0, 20000.0)
SHELF_FREQ_RANGE = (50.0, 16000.0)
PEAK_FREQ_RANGE = (100.0, 15000.0)
GAIN_RANGE = (-24.0, 24.0)
BROADBAND_RANGE = (-60.0, 0.0)
Q_RANGE = (0.5, 16.0)

FRAME_SIZE = 2048


class DifferentiableBiquadChain(nn.Module):
    """
    Differentiable biquad chain using torchaudio.lfilter for speed.
    """

    def __init__(self, sample_rate: float = 96000.0):
        super().__init__()
        self.sample_rate = sample_rate

    def reset_state(self, batch_size: int = 1, device=None):
        """Reset filter states."""
        pass  # lfilter handles state internally per call

    def _denorm_freq(self, norm, freq_range):
        log_low = math.log(freq_range[0])
        log_high = math.log(freq_range[1])
        return torch.exp(log_low + norm * (log_high - log_low))

    def _denorm_gain(self, norm, gain_range=GAIN_RANGE):
        return gain_range[0] + norm * (gain_range[1] - gain_range[0])

    def _denorm_q(self, norm):
        log_low = math.log(Q_RANGE[0])
        log_high = math.log(Q_RANGE[1])
        return torch.exp(log_low + norm * (log_high - log_low))

    def _biquad_coeffs_hp(self, fc, Q):
        """Highpass biquad coefficients (Direct Form II)."""
        w0 = 2 * math.pi * fc / self.sample_rate
        alpha = torch.sin(w0) / (2 * Q)
        cos_w0 = torch.cos(w0)

        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

        return b0/a0, b1/a0, b2/a0, a1/a0, a2/a0

    def _biquad_coeffs_lp(self, fc, Q):
        """Lowpass biquad coefficients."""
        w0 = 2 * math.pi * fc / self.sample_rate
        alpha = torch.sin(w0) / (2 * Q)
        cos_w0 = torch.cos(w0)

        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

        return b0/a0, b1/a0, b2/a0, a1/a0, a2/a0

    def _biquad_coeffs_peak(self, fc, gain_db, Q):
        """Peaking EQ biquad coefficients."""
        A = torch.pow(10.0, gain_db / 40.0)
        w0 = 2 * math.pi * fc / self.sample_rate
        alpha = torch.sin(w0) / (2 * Q)
        cos_w0 = torch.cos(w0)

        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

        return b0/a0, b1/a0, b2/a0, a1/a0, a2/a0

    def _biquad_coeffs_lowshelf(self, fc, gain_db, Q):
        """Low shelf biquad coefficients."""
        A = torch.pow(10.0, gain_db / 40.0)
        w0 = 2 * math.pi * fc / self.sample_rate
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

    def _biquad_coeffs_highshelf(self, fc, gain_db, Q):
        """High shelf biquad coefficients."""
        A = torch.pow(10.0, gain_db / 40.0)
        w0 = 2 * math.pi * fc / self.sample_rate
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

    def _apply_biquad_batch(self, x, b0, b1, b2, a1, a2):
        """
        Apply biquad filter to batched audio using lfilter.

        Args:
            x: (batch, samples)
            b0, b1, b2, a1, a2: (batch,) coefficients

        Returns:
            Filtered audio (batch, samples)
        """
        batch_size = x.shape[0]
        outputs = []

        # lfilter expects (batch, samples) and coefficient tensors
        # Process each batch item (lfilter doesn't support batched coeffs directly)
        for i in range(batch_size):
            a_coeffs = torch.stack([torch.ones_like(a1[i]), a1[i], a2[i]])
            b_coeffs = torch.stack([b0[i], b1[i], b2[i]])
            out = AF.lfilter(x[i:i+1], a_coeffs, b_coeffs, clamp=False)
            outputs.append(out)

        return torch.cat(outputs, dim=0)

    def forward(self, audio: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Process audio through the filter chain."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        batch_size, n_samples = audio.shape
        device = audio.device

        if params.dim() == 2:
            params = params.unsqueeze(2)

        n_frames_params = params.shape[2]
        n_frames_audio = (n_samples + FRAME_SIZE - 1) // FRAME_SIZE

        output_frames = []

        for frame_idx in range(n_frames_audio):
            start = frame_idx * FRAME_SIZE
            end = min(start + FRAME_SIZE, n_samples)

            if start >= n_samples:
                break

            frame_audio = audio[:, start:end]
            param_idx = min(frame_idx, n_frames_params - 1)
            frame_params = params[:, :, param_idx]

            # Extract parameters
            biquad_params = frame_params[:, :N_BIQUADS * N_PARAMS_PER_BIQUAD].view(batch_size, N_BIQUADS, 3)
            input_gain_norm = frame_params[:, -2]
            output_gain_norm = frame_params[:, -1]

            # Convert gains
            input_gain_db = self._denorm_gain(input_gain_norm, BROADBAND_RANGE)
            output_gain_db = self._denorm_gain(output_gain_norm, BROADBAND_RANGE)
            input_gain = torch.pow(10.0, input_gain_db / 20.0)
            output_gain = torch.pow(10.0, output_gain_db / 20.0)

            # Apply input gain
            x = frame_audio * input_gain.unsqueeze(1)

            # Process through each filter
            for i in range(N_BIQUADS):
                freq_norm = biquad_params[:, i, 0]
                gain_norm = biquad_params[:, i, 1]
                q_norm = biquad_params[:, i, 2]

                if i == BIQUAD_HPF:
                    fc = self._denorm_freq(freq_norm, HPF_FREQ_RANGE)
                    Q = self._denorm_q(q_norm)
                    b0, b1, b2, a1, a2 = self._biquad_coeffs_hp(fc, Q)
                elif i == BIQUAD_LPF:
                    fc = self._denorm_freq(freq_norm, LPF_FREQ_RANGE)
                    Q = self._denorm_q(q_norm)
                    b0, b1, b2, a1, a2 = self._biquad_coeffs_lp(fc, Q)
                elif i == BIQUAD_LOW_SHELF:
                    fc = self._denorm_freq(freq_norm, SHELF_FREQ_RANGE)
                    gain_db = self._denorm_gain(gain_norm, GAIN_RANGE)
                    Q = self._denorm_q(q_norm)
                    b0, b1, b2, a1, a2 = self._biquad_coeffs_lowshelf(fc, gain_db, Q)
                elif i == BIQUAD_HIGH_SHELF:
                    fc = self._denorm_freq(freq_norm, SHELF_FREQ_RANGE)
                    gain_db = self._denorm_gain(gain_norm, GAIN_RANGE)
                    Q = self._denorm_q(q_norm)
                    b0, b1, b2, a1, a2 = self._biquad_coeffs_highshelf(fc, gain_db, Q)
                else:  # Peaking
                    fc = self._denorm_freq(freq_norm, PEAK_FREQ_RANGE)
                    gain_db = self._denorm_gain(gain_norm, GAIN_RANGE)
                    Q = self._denorm_q(q_norm)
                    b0, b1, b2, a1, a2 = self._biquad_coeffs_peak(fc, gain_db, Q)

                x = self._apply_biquad_batch(x, b0, b1, b2, a1, a2)

            # Apply output gain
            x = x * output_gain.unsqueeze(1)
            output_frames.append(x)

        return torch.cat(output_frames, dim=1)


def benchmark():
    """Benchmark the lfilter-based chain."""
    import time

    print("Benchmarking lfilter-based DifferentiableBiquadChain...")

    chain = DifferentiableBiquadChain(sample_rate=96000)

    batch_size = 2
    n_samples = 96000  # 1 second
    audio = torch.randn(batch_size, n_samples)
    n_frames = n_samples // FRAME_SIZE
    params = torch.rand(batch_size, N_TOTAL_PARAMS, n_frames)

    # Warmup
    print("Warmup...")
    _ = chain(audio[:1, :4096], params[:1, :, :2])

    # Benchmark forward
    print("Benchmark forward pass...")
    start = time.time()
    output = chain(audio, params)
    fwd_time = time.time() - start
    print(f"Forward: {fwd_time:.3f}s ({1.0/fwd_time:.2f}x realtime)")

    # Benchmark backward
    print("Benchmark backward pass...")
    audio2 = torch.randn(batch_size, n_samples, requires_grad=True)
    params2 = torch.rand(batch_size, N_TOTAL_PARAMS, n_frames, requires_grad=True)

    start = time.time()
    output = chain(audio2, params2)
    loss = output.mean()
    loss.backward()
    bwd_time = time.time() - start
    print(f"Forward+Backward: {bwd_time:.3f}s ({1.0/bwd_time:.2f}x realtime)")

    print(f"Gradients exist: audio={audio2.grad is not None}, params={params2.grad is not None}")
    print("Done!")


if __name__ == '__main__':
    benchmark()
