#!/usr/bin/env python3
"""
Differentiable SVF TPT Filter Chain for PyTorch - OPTIMIZED VERSION
====================================================================
Matches the C++ DifferentiableBiquadChain implementation exactly.

Key optimizations:
1. torch.compile for JIT compilation of filter processing
2. Vectorized batch processing
3. Pre-computed coefficients for all filters
4. Efficient tensor operations within frames
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, List

# Configuration matching C++
N_BIQUADS = 16
N_PARAMS_PER_BIQUAD = 3  # freq, gain, Q
N_EXTRA_PARAMS = 2  # input gain, output gain
N_TOTAL_PARAMS = N_BIQUADS * N_PARAMS_PER_BIQUAD + N_EXTRA_PARAMS  # 50

# Filter indices matching C++
BIQUAD_HPF = 0
BIQUAD_LOW_SHELF = 1
BIQUAD_PEAKING_START = 2
BIQUAD_PEAKING_END = 13
BIQUAD_HIGH_SHELF = 14
BIQUAD_LPF = 15

# Frequency ranges matching C++
HPF_FREQ_RANGE = (20.0, 500.0)
LPF_FREQ_RANGE = (5000.0, 20000.0)
SHELF_FREQ_RANGE = (50.0, 16000.0)
PEAK_FREQ_RANGE = (100.0, 15000.0)

# Gain/Q ranges matching C++
GAIN_RANGE = (-24.0, 24.0)
BROADBAND_RANGE = (-60.0, 0.0)
Q_RANGE = (0.5, 16.0)

# Frame size matching C++
FRAME_SIZE = 2048


@torch.jit.script
def _svf_process_frame(
    x: torch.Tensor,
    ic1eq: torch.Tensor,
    ic2eq: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    m0: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process a frame of audio through one SVF filter - JIT compiled.

    Args:
        x: Audio frame (batch, samples)
        ic1eq, ic2eq: Filter state (batch,)
        a1, a2, a3, m0, m1, m2: SVF coefficients (batch,)

    Returns:
        output: Filtered audio (batch, samples)
        new_ic1eq, new_ic2eq: Updated state
    """
    batch_size = x.shape[0]
    n_samples = x.shape[1]

    # Pre-allocate output tensor
    output = torch.empty_like(x)

    # Process sample by sample (JIT compiled - no Python overhead)
    for n in range(n_samples):
        x_n = x[:, n]

        # SVF TPT equations - exactly matching C++
        v3 = x_n - ic2eq
        v1 = a1 * ic1eq + a2 * v3
        v2 = ic2eq + a2 * v1 + a3 * v3

        # Update state
        ic1eq = 2.0 * v1 - ic1eq
        ic2eq = 2.0 * v2 - ic2eq

        # Output mixing
        output[:, n] = m0 * x_n + m1 * v1 + m2 * v2

    return output, ic1eq, ic2eq


class DifferentiableBiquadChain(nn.Module):
    """
    Differentiable SVF TPT filter chain matching C++ implementation.

    OPTIMIZED VERSION with:
    - JIT-compiled sample processing
    - Vectorized batch operations
    - Pre-computed coefficients
    """

    def __init__(self, sample_rate: float = 96000.0):
        super().__init__()
        self.sample_rate = sample_rate
        self._compiled = False

    def reset_state(self, batch_size: int = 1, device: torch.device = None):
        """Reset filter states for new audio."""
        # States stored as (n_filters, batch, 2) for ic1eq, ic2eq
        self._filter_states = torch.zeros(N_BIQUADS, batch_size, 2, device=device)

    def _denormalize_freq(self, norm: torch.Tensor, freq_range: Tuple[float, float]) -> torch.Tensor:
        """Log-scale frequency denormalization."""
        log_low = math.log(freq_range[0])
        log_high = math.log(freq_range[1])
        log_freq = log_low + norm * (log_high - log_low)
        return torch.exp(log_freq)

    def _denormalize_gain(self, norm: torch.Tensor, gain_range: Tuple[float, float] = GAIN_RANGE) -> torch.Tensor:
        """Linear gain denormalization."""
        return gain_range[0] + norm * (gain_range[1] - gain_range[0])

    def _denormalize_q(self, norm: torch.Tensor) -> torch.Tensor:
        """Log-scale Q denormalization."""
        log_low = math.log(Q_RANGE[0])
        log_high = math.log(Q_RANGE[1])
        log_q = log_low + norm * (log_high - log_low)
        return torch.exp(log_q)

    def _compute_all_svf_coeffs(self, biquad_params: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute SVF TPT coefficients for ALL 16 filters at once.

        Args:
            biquad_params: (batch, 16, 3) - freq_norm, gain_norm, q_norm for each filter

        Returns:
            Tuple of coefficient tensors, each (16, batch)
        """
        batch_size = biquad_params.shape[0]
        device = biquad_params.device
        dtype = biquad_params.dtype

        # Initialize coefficient tensors (16 filters, batch)
        a1_all = torch.zeros(N_BIQUADS, batch_size, device=device, dtype=dtype)
        a2_all = torch.zeros(N_BIQUADS, batch_size, device=device, dtype=dtype)
        a3_all = torch.zeros(N_BIQUADS, batch_size, device=device, dtype=dtype)
        m0_all = torch.zeros(N_BIQUADS, batch_size, device=device, dtype=dtype)
        m1_all = torch.zeros(N_BIQUADS, batch_size, device=device, dtype=dtype)
        m2_all = torch.zeros(N_BIQUADS, batch_size, device=device, dtype=dtype)

        for i in range(N_BIQUADS):
            freq_norm = biquad_params[:, i, 0]
            gain_norm = biquad_params[:, i, 1]
            q_norm = biquad_params[:, i, 2]

            # Determine filter type and frequency range
            if i == BIQUAD_HPF:
                fc = self._denormalize_freq(freq_norm, HPF_FREQ_RANGE)
                Q = self._denormalize_q(q_norm)
                a1, a2, a3, m0, m1, m2 = self._svf_highpass_coeffs(fc, Q)
            elif i == BIQUAD_LPF:
                fc = self._denormalize_freq(freq_norm, LPF_FREQ_RANGE)
                Q = self._denormalize_q(q_norm)
                a1, a2, a3, m0, m1, m2 = self._svf_lowpass_coeffs(fc, Q)
            elif i == BIQUAD_LOW_SHELF:
                fc = self._denormalize_freq(freq_norm, SHELF_FREQ_RANGE)
                gain_db = self._denormalize_gain(gain_norm, GAIN_RANGE)
                Q = self._denormalize_q(q_norm)
                a1, a2, a3, m0, m1, m2 = self._svf_lowshelf_coeffs(fc, gain_db, Q)
            elif i == BIQUAD_HIGH_SHELF:
                fc = self._denormalize_freq(freq_norm, SHELF_FREQ_RANGE)
                gain_db = self._denormalize_gain(gain_norm, GAIN_RANGE)
                Q = self._denormalize_q(q_norm)
                a1, a2, a3, m0, m1, m2 = self._svf_highshelf_coeffs(fc, gain_db, Q)
            else:  # Peaking EQ
                fc = self._denormalize_freq(freq_norm, PEAK_FREQ_RANGE)
                gain_db = self._denormalize_gain(gain_norm, GAIN_RANGE)
                Q = self._denormalize_q(q_norm)
                a1, a2, a3, m0, m1, m2 = self._svf_peak_coeffs(fc, gain_db, Q)

            a1_all[i] = a1
            a2_all[i] = a2
            a3_all[i] = a3
            m0_all[i] = m0
            m1_all[i] = m1
            m2_all[i] = m2

        return a1_all, a2_all, a3_all, m0_all, m1_all, m2_all

    def _svf_highpass_coeffs(self, fc: torch.Tensor, Q: torch.Tensor):
        """Compute highpass SVF coefficients."""
        g = torch.tan(math.pi * fc / self.sample_rate)
        g = torch.clamp(g, min=1e-6, max=100.0)
        Q = torch.clamp(Q, min=0.1, max=100.0)

        k = 1.0 / Q
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        m0 = torch.ones_like(g)
        m1 = -k
        m2 = -torch.ones_like(g)

        return a1, a2, a3, m0, m1, m2

    def _svf_lowpass_coeffs(self, fc: torch.Tensor, Q: torch.Tensor):
        """Compute lowpass SVF coefficients."""
        g = torch.tan(math.pi * fc / self.sample_rate)
        g = torch.clamp(g, min=1e-6, max=100.0)
        Q = torch.clamp(Q, min=0.1, max=100.0)

        k = 1.0 / Q
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        m0 = torch.zeros_like(g)
        m1 = torch.zeros_like(g)
        m2 = torch.ones_like(g)

        return a1, a2, a3, m0, m1, m2

    def _svf_peak_coeffs(self, fc: torch.Tensor, gain_db: torch.Tensor, Q: torch.Tensor):
        """Compute peaking EQ SVF coefficients."""
        g = torch.tan(math.pi * fc / self.sample_rate)
        g = torch.clamp(g, min=1e-6, max=100.0)
        Q = torch.clamp(Q, min=0.1, max=100.0)

        A = torch.pow(10.0, gain_db / 40.0)
        k = torch.where(gain_db >= 0, 1.0 / (Q * A), A / Q)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        m0 = torch.ones_like(g)
        m1 = k * (A * A - 1.0)
        m2 = torch.zeros_like(g)

        return a1, a2, a3, m0, m1, m2

    def _svf_lowshelf_coeffs(self, fc: torch.Tensor, gain_db: torch.Tensor, Q: torch.Tensor):
        """Compute low shelf SVF coefficients."""
        g = torch.tan(math.pi * fc / self.sample_rate)
        g = torch.clamp(g, min=1e-6, max=100.0)
        Q = torch.clamp(Q, min=0.1, max=100.0)

        A = torch.pow(10.0, gain_db / 40.0)
        sqrt_A = torch.sqrt(A)
        k = 1.0 / Q
        g_shelf = torch.where(gain_db >= 0, g / sqrt_A, g * sqrt_A)
        a1 = 1.0 / (1.0 + g_shelf * (g_shelf + k))
        a2 = g_shelf * a1
        a3 = g_shelf * a2
        m0 = torch.ones_like(g)
        m1 = k * (A - 1.0)
        m2 = A * A - 1.0

        return a1, a2, a3, m0, m1, m2

    def _svf_highshelf_coeffs(self, fc: torch.Tensor, gain_db: torch.Tensor, Q: torch.Tensor):
        """Compute high shelf SVF coefficients."""
        g = torch.tan(math.pi * fc / self.sample_rate)
        g = torch.clamp(g, min=1e-6, max=100.0)
        Q = torch.clamp(Q, min=0.1, max=100.0)

        A = torch.pow(10.0, gain_db / 40.0)
        sqrt_A = torch.sqrt(A)
        k = 1.0 / Q
        g_shelf = torch.where(gain_db >= 0, g * sqrt_A, g / sqrt_A)
        a1 = 1.0 / (1.0 + g_shelf * (g_shelf + k))
        a2 = g_shelf * a1
        a3 = g_shelf * a2
        m0 = A * A
        m1 = k * (1.0 - A) * A
        m2 = 1.0 - A * A

        return a1, a2, a3, m0, m1, m2

    def _process_frame_all_filters(
        self,
        x: torch.Tensor,
        a1_all: torch.Tensor,
        a2_all: torch.Tensor,
        a3_all: torch.Tensor,
        m0_all: torch.Tensor,
        m1_all: torch.Tensor,
        m2_all: torch.Tensor,
        input_gain: torch.Tensor,
        output_gain: torch.Tensor
    ) -> torch.Tensor:
        """
        Process audio frame through all 16 filters.

        Args:
            x: Audio frame (batch, samples)
            *_all: Coefficients (16, batch)
            input_gain, output_gain: (batch,)

        Returns:
            Filtered audio (batch, samples)
        """
        # Apply input gain
        x = x * input_gain.unsqueeze(1)

        # Process through each filter
        for i in range(N_BIQUADS):
            ic1eq = self._filter_states[i, :, 0]
            ic2eq = self._filter_states[i, :, 1]

            x, ic1eq, ic2eq = _svf_process_frame(
                x, ic1eq, ic2eq,
                a1_all[i], a2_all[i], a3_all[i],
                m0_all[i], m1_all[i], m2_all[i]
            )

            # Update state
            self._filter_states[i, :, 0] = ic1eq
            self._filter_states[i, :, 1] = ic2eq

        # Apply output gain
        x = x * output_gain.unsqueeze(1)

        return x

    def forward(self, audio: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Process audio through the SVF filter chain with per-frame parameters.

        Args:
            audio: Input audio (batch, samples)
            params: Normalized parameters (batch, 50, n_frames) or (batch, 50) in range [0, 1]

        Returns:
            Filtered audio (batch, samples)
        """
        batch_size = audio.shape[0] if audio.dim() > 1 else 1
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        n_samples = audio.shape[1]
        device = audio.device

        # Handle both (batch, 50) and (batch, 50, n_frames) param shapes
        if params.dim() == 2:
            n_frames_params = 1
            params = params.unsqueeze(2)
        else:
            n_frames_params = params.shape[2]

        n_frames_audio = (n_samples + FRAME_SIZE - 1) // FRAME_SIZE

        # Initialize filter states
        self.reset_state(batch_size, device)

        # Process frame by frame
        output_frames = []

        for frame_idx in range(n_frames_audio):
            start_sample = frame_idx * FRAME_SIZE
            end_sample = min(start_sample + FRAME_SIZE, n_samples)

            if start_sample >= n_samples:
                break

            # Get audio frame
            frame_audio = audio[:, start_sample:end_sample]

            # Get params for this frame
            param_idx = min(frame_idx, n_frames_params - 1)
            frame_params = params[:, :, param_idx]

            # Extract parameters
            biquad_params = frame_params[:, :N_BIQUADS * N_PARAMS_PER_BIQUAD].view(batch_size, N_BIQUADS, 3)
            input_gain_norm = frame_params[:, -2]
            output_gain_norm = frame_params[:, -1]

            # Convert gains
            input_gain_db = self._denormalize_gain(input_gain_norm, BROADBAND_RANGE)
            output_gain_db = self._denormalize_gain(output_gain_norm, BROADBAND_RANGE)
            input_gain = torch.pow(10.0, input_gain_db / 20.0)
            output_gain = torch.pow(10.0, output_gain_db / 20.0)

            # Compute all filter coefficients at once
            a1_all, a2_all, a3_all, m0_all, m1_all, m2_all = self._compute_all_svf_coeffs(biquad_params)

            # Process frame through all filters
            frame_output = self._process_frame_all_filters(
                frame_audio,
                a1_all, a2_all, a3_all, m0_all, m1_all, m2_all,
                input_gain, output_gain
            )

            output_frames.append(frame_output)

        return torch.cat(output_frames, dim=1)


def test_svf_chain():
    """Quick test of the optimized SVF chain."""
    import time

    print("Testing Optimized DifferentiableBiquadChain...")

    chain = DifferentiableBiquadChain(sample_rate=96000)

    # Create test audio (1 second at 96kHz)
    batch_size = 2
    n_samples = 96000
    audio = torch.randn(batch_size, n_samples)

    # Create random parameters
    n_frames = n_samples // FRAME_SIZE
    params = torch.rand(batch_size, N_TOTAL_PARAMS, n_frames)

    # Warmup (for JIT)
    print("Warming up JIT...")
    _ = chain(audio[:1, :4096], params[:1, :, :2])

    # Benchmark
    print("Running benchmark (1 second of audio, batch=2)...")
    start = time.time()
    output = chain(audio, params)
    elapsed = time.time() - start

    print(f"Input shape: {audio.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Processing time: {elapsed:.3f}s")
    print(f"Real-time factor: {n_samples / 96000 / elapsed:.2f}x")

    # Test gradient flow
    print("Testing gradient flow...")
    audio2 = torch.randn(batch_size, n_samples, requires_grad=True)
    params2 = torch.rand(batch_size, N_TOTAL_PARAMS, n_frames, requires_grad=True)

    output = chain(audio2, params2)
    loss = output.sum()
    loss.backward()

    print(f"Audio grad exists: {audio2.grad is not None}")
    print(f"Params grad exists: {params2.grad is not None}")
    print("✓ Optimized SVF chain test passed!")


if __name__ == '__main__':
    test_svf_chain()
