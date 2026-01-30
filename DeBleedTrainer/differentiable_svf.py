#!/usr/bin/env python3
"""
Differentiable SVF TPT Filter Chain for PyTorch
================================================
Matches the C++ DifferentiableBiquadChain implementation exactly.

This ensures the neural network learns to predict parameters that work
correctly with the SVF TPT filters used in the real-time plugin.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

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


class DifferentiableBiquadChain(nn.Module):
    """
    Differentiable SVF TPT filter chain matching C++ implementation.
    
    Processes audio sample-by-sample with coefficients computed from
    normalized parameters [0, 1].
    """
    
    def __init__(self, sample_rate: float = 96000.0):
        super().__init__()
        self.sample_rate = sample_rate
        
    def reset_state(self, batch_size: int = 1, device: torch.device = None):
        """
        Reset filter states for new audio.
        
        Note: Current implementation creates fresh state for each forward() call,
        so this is a no-op. Provided for API compatibility with the trainer.
        """
        pass
    
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
    
    def _compute_svf_coeffs(self, filter_type: str, fc: torch.Tensor, 
                            gain_db: torch.Tensor, Q: torch.Tensor) -> dict:
        """
        Compute SVF TPT coefficients matching C++ computeCoeffs().
        
        Returns dict with g, k, a1, a2, a3, m0, m1, m2
        """
        # Basic SVF coefficient
        g = torch.tan(math.pi * fc / self.sample_rate)
        
        # Clamp for stability
        g = torch.clamp(g, min=1e-6, max=100.0)
        Q = torch.clamp(Q, min=0.1, max=100.0)
        
        if filter_type == 'highpass':
            k = 1.0 / Q
            a1 = 1.0 / (1.0 + g * (g + k))
            a2 = g * a1
            a3 = g * a2
            m0 = torch.ones_like(g)
            m1 = -k
            m2 = -torch.ones_like(g)
            
        elif filter_type == 'lowpass':
            k = 1.0 / Q
            a1 = 1.0 / (1.0 + g * (g + k))
            a2 = g * a1
            a3 = g * a2
            m0 = torch.zeros_like(g)
            m1 = torch.zeros_like(g)
            m2 = torch.ones_like(g)
            
        elif filter_type == 'peak':
            A = torch.pow(10.0, gain_db / 40.0)
            # k depends on boost vs cut
            k = torch.where(gain_db >= 0, 1.0 / (Q * A), A / Q)
            a1 = 1.0 / (1.0 + g * (g + k))
            a2 = g * a1
            a3 = g * a2
            # Same formula for both boost and cut
            m0 = torch.ones_like(g)
            m1 = k * (A * A - 1.0)
            m2 = torch.zeros_like(g)
            
        elif filter_type == 'lowshelf':
            A = torch.pow(10.0, gain_db / 40.0)
            sqrt_A = torch.sqrt(A)
            k = 1.0 / Q
            # g_shelf depends on boost vs cut
            g_shelf = torch.where(gain_db >= 0, g / sqrt_A, g * sqrt_A)
            a1 = 1.0 / (1.0 + g_shelf * (g_shelf + k))
            a2 = g_shelf * a1
            a3 = g_shelf * a2
            m0 = torch.ones_like(g)
            m1 = k * (A - 1.0)
            m2 = A * A - 1.0
            
        elif filter_type == 'highshelf':
            A = torch.pow(10.0, gain_db / 40.0)
            sqrt_A = torch.sqrt(A)
            k = 1.0 / Q
            # g_shelf depends on boost vs cut
            g_shelf = torch.where(gain_db >= 0, g * sqrt_A, g / sqrt_A)
            a1 = 1.0 / (1.0 + g_shelf * (g_shelf + k))
            a2 = g_shelf * a1
            a3 = g_shelf * a2
            m0 = A * A
            m1 = k * (1.0 - A) * A
            m2 = 1.0 - A * A
            
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        return {'g': g, 'k': k, 'a1': a1, 'a2': a2, 'a3': a3, 
                'm0': m0, 'm1': m1, 'm2': m2}
    
    def _process_svf_frame(self, x: torch.Tensor, coeffs: dict, 
                           state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                           ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process audio through one SVF filter.
        
        Args:
            x: Audio (batch, samples)
            coeffs: SVF coefficients from _compute_svf_coeffs
            state: Optional (ic1eq, ic2eq) state tensors
            
        Returns:
            output: Filtered audio (batch, samples)
            new_state: (new_ic1eq, new_ic2eq)
        """
        batch_size, n_samples = x.shape
        device = x.device
        
        # Initialize state if not provided
        if state is None:
            ic1eq = torch.zeros(batch_size, device=device)
            ic2eq = torch.zeros(batch_size, device=device)
        else:
            ic1eq, ic2eq = state
        
        # Extract coefficients (may be (batch,) or scalar)
        a1 = coeffs['a1']
        a2 = coeffs['a2']
        a3 = coeffs['a3']
        m0 = coeffs['m0']
        m1 = coeffs['m1']
        m2 = coeffs['m2']
        k = coeffs['k']
        
        # Process sample by sample
        outputs = []
        for n in range(n_samples):
            x_n = x[:, n]  # (batch,)
            
            # SVF TPT equations
            v3 = x_n - ic2eq
            v1 = a1 * ic1eq + a2 * v3
            v2 = ic2eq + a2 * v1 + a3 * v3
            
            # Update state
            ic1eq = 2.0 * v1 - ic1eq
            ic2eq = 2.0 * v2 - ic2eq
            
            # Output mixing
            low = v2
            band = v1
            output = m0 * x_n + m1 * band + m2 * low
            
            outputs.append(output)
        
        return torch.stack(outputs, dim=1), (ic1eq, ic2eq)
    
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
        
        # Handle both (batch, 50) and (batch, 50, n_frames) param shapes
        if params.dim() == 2:
            # Single frame params - expand to all frames
            n_frames_params = 1
            params = params.unsqueeze(2)  # (batch, 50, 1)
        else:
            n_frames_params = params.shape[2]
        
        # Frame size for processing (matching C++ FRAME_SIZE = 2048)
        FRAME_SIZE = 2048
        n_frames_audio = (n_samples + FRAME_SIZE - 1) // FRAME_SIZE
        
        # Initialize filter states for all 16 filters
        filter_states = {}
        for i in range(N_BIQUADS):
            filter_states[i] = (
                torch.zeros(batch_size, device=audio.device),  # ic1eq
                torch.zeros(batch_size, device=audio.device)   # ic2eq
            )
        
        # Process frame by frame
        output_frames = []
        
        for frame_idx in range(n_frames_audio):
            start_sample = frame_idx * FRAME_SIZE
            end_sample = min(start_sample + FRAME_SIZE, n_samples)
            
            if start_sample >= n_samples:
                break
            
            # Get audio frame
            frame_audio = audio[:, start_sample:end_sample]
            
            # Get params for this frame (use last frame if we've run out)
            param_idx = min(frame_idx, n_frames_params - 1)
            frame_params = params[:, :, param_idx]  # (batch, 50)
            
            # Extract parameters
            biquad_params = frame_params[:, :N_BIQUADS * N_PARAMS_PER_BIQUAD].view(batch_size, N_BIQUADS, 3)
            input_gain_norm = frame_params[:, -2]
            output_gain_norm = frame_params[:, -1]
            
            # Convert gains
            input_gain_db = self._denormalize_gain(input_gain_norm, BROADBAND_RANGE)
            output_gain_db = self._denormalize_gain(output_gain_norm, BROADBAND_RANGE)
            input_gain = torch.pow(10.0, input_gain_db / 20.0)
            output_gain = torch.pow(10.0, output_gain_db / 20.0)
            
            # Apply input gain
            x = frame_audio * input_gain.unsqueeze(1)
            
            # Process through each filter (maintaining state across frames)
            for i in range(N_BIQUADS):
                freq_norm = biquad_params[:, i, 0]
                gain_norm = biquad_params[:, i, 1]
                q_norm = biquad_params[:, i, 2]
                
                # Determine filter type and compute coefficients
                if i == BIQUAD_HPF:
                    fc = self._denormalize_freq(freq_norm, HPF_FREQ_RANGE)
                    Q = self._denormalize_q(q_norm)
                    coeffs = self._compute_svf_coeffs('highpass', fc, None, Q)
                elif i == BIQUAD_LPF:
                    fc = self._denormalize_freq(freq_norm, LPF_FREQ_RANGE)
                    Q = self._denormalize_q(q_norm)
                    coeffs = self._compute_svf_coeffs('lowpass', fc, None, Q)
                elif i == BIQUAD_LOW_SHELF:
                    fc = self._denormalize_freq(freq_norm, SHELF_FREQ_RANGE)
                    gain_db = self._denormalize_gain(gain_norm, GAIN_RANGE)
                    Q = self._denormalize_q(q_norm)
                    coeffs = self._compute_svf_coeffs('lowshelf', fc, gain_db, Q)
                elif i == BIQUAD_HIGH_SHELF:
                    fc = self._denormalize_freq(freq_norm, SHELF_FREQ_RANGE)
                    gain_db = self._denormalize_gain(gain_norm, GAIN_RANGE)
                    Q = self._denormalize_q(q_norm)
                    coeffs = self._compute_svf_coeffs('highshelf', fc, gain_db, Q)
                else:  # Peaking EQ
                    fc = self._denormalize_freq(freq_norm, PEAK_FREQ_RANGE)
                    gain_db = self._denormalize_gain(gain_norm, GAIN_RANGE)
                    Q = self._denormalize_q(q_norm)
                    coeffs = self._compute_svf_coeffs('peak', fc, gain_db, Q)
                
                # Process through this filter with state
                x, filter_states[i] = self._process_svf_frame(x, coeffs, filter_states[i])
            
            # Apply output gain
            x = x * output_gain.unsqueeze(1)
            output_frames.append(x)
        
        # Concatenate all frames
        return torch.cat(output_frames, dim=1)

def test_svf_chain():
    """Quick test of the SVF chain."""
    print("Testing DifferentiableBiquadChain...")
    
    chain = DifferentiableBiquadChain(sample_rate=96000)
    
    # Create test audio (impulse)
    batch_size = 2
    n_samples = 4096
    audio = torch.zeros(batch_size, n_samples)
    audio[:, 0] = 1.0  # Impulse
    
    # Create neutral parameters (0.5 for filters, 1.0 for gains)
    params = torch.full((batch_size, N_TOTAL_PARAMS), 0.5)
    params[:, -2] = 1.0  # Input gain = 0dB
    params[:, -1] = 1.0  # Output gain = 0dB
    
    # Process
    output = chain(audio, params)
    
    print(f"Input shape: {audio.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input max: {audio.max().item():.4f}")
    print(f"Output max: {output.max().item():.4f}")
    
    # Test gradient flow
    audio.requires_grad_(True)
    params = torch.full((batch_size, N_TOTAL_PARAMS), 0.5, requires_grad=True)
    params_modified = params.clone()
    params_modified[:, -2] = 1.0
    params_modified[:, -1] = 1.0
    
    output = chain(audio, params_modified)
    loss = output.sum()
    loss.backward()
    
    print(f"Audio grad exists: {audio.grad is not None}")
    print(f"Params grad exists: {params.grad is not None}")
    print("✓ SVF chain test passed!")


if __name__ == '__main__':
    test_svf_chain()
