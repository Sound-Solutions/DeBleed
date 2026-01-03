#!/usr/bin/env python3
"""
SVF TPT Frequency Response Validator - FIXED VERSION
=====================================================
Tests that the biquad coefficient calculations produce correct frequency responses.
"""

import numpy as np
from scipy import signal
from typing import Tuple
import sys

SAMPLE_RATE = 96000.0

# ============================================================================
# Reference implementations using scipy (known-correct)
# ============================================================================

def scipy_peak(fc: float, gain_db: float, Q: float, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Reference peaking EQ using scipy."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])

def scipy_lowshelf(fc: float, gain_db: float, Q: float, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Reference low shelf using scipy."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)
    
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])

def scipy_highshelf(fc: float, gain_db: float, Q: float, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Reference high shelf using scipy."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)
    
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])

def scipy_highpass(fc: float, Q: float, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Reference highpass using scipy."""
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    
    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])

def scipy_lowpass(fc: float, Q: float, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Reference lowpass using scipy."""
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    
    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


# ============================================================================
# FIXED SVF TPT Implementation (mirroring corrected C++)
# ============================================================================

def svf_peak_coeffs(fc: float, gain_db: float, Q: float, fs: float) -> dict:
    """FIXED: Same formula for both boost and cut."""
    g = np.tan(np.pi * fc / fs)
    A = 10 ** (gain_db / 40)
    
    if gain_db >= 0:
        k = 1.0 / (Q * A)
    else:
        k = A / Q
    
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    
    # FIXED: A^2-1 is negative for cuts (A < 1), positive for boosts
    m0 = 1.0
    m1 = k * (A * A - 1.0)
    m2 = 0.0
    
    return {'g': g, 'k': k, 'A': A, 'a1': a1, 'a2': a2, 'a3': a3, 'm0': m0, 'm1': m1, 'm2': m2}


def svf_lowshelf_coeffs(fc: float, gain_db: float, Q: float, fs: float) -> dict:
    """FIXED: Low shelf with corrected cut formula."""
    g = np.tan(np.pi * fc / fs)
    A = 10 ** (gain_db / 40)
    sqrt_A = np.sqrt(A)
    k = 1.0 / Q
    
    if gain_db >= 0:
        g_shelf = g / sqrt_A
    else:
        g_shelf = g * sqrt_A
    
    a1 = 1.0 / (1.0 + g_shelf * (g_shelf + k))
    a2 = g_shelf * a1
    a3 = g_shelf * a2
    
    # FIXED: Same formula for both
    m0 = 1.0
    m1 = k * (A - 1.0)
    m2 = A * A - 1.0
    
    return {'g': g, 'k': k, 'A': A, 'a1': a1, 'a2': a2, 'a3': a3, 'm0': m0, 'm1': m1, 'm2': m2}


def svf_highshelf_coeffs(fc: float, gain_db: float, Q: float, fs: float) -> dict:
    """FIXED: High shelf with corrected cut formula."""
    g = np.tan(np.pi * fc / fs)
    A = 10 ** (gain_db / 40)
    sqrt_A = np.sqrt(A)
    k = 1.0 / Q
    
    if gain_db >= 0:
        g_shelf = g * sqrt_A
    else:
        g_shelf = g / sqrt_A
    
    a1 = 1.0 / (1.0 + g_shelf * (g_shelf + k))
    a2 = g_shelf * a1
    a3 = g_shelf * a2
    
    # FIXED: Same formula for both
    m0 = A * A
    m1 = k * (1.0 - A) * A
    m2 = 1.0 - A * A
    
    return {'g': g, 'k': k, 'A': A, 'a1': a1, 'a2': a2, 'a3': a3, 'm0': m0, 'm1': m1, 'm2': m2}


def svf_highpass_coeffs(fc: float, Q: float, fs: float) -> dict:
    """SVF highpass coefficients."""
    g = np.tan(np.pi * fc / fs)
    k = 1.0 / Q
    
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    
    m0 = 1.0
    m1 = -k
    m2 = -1.0
    
    return {'g': g, 'k': k, 'A': 1.0, 'a1': a1, 'a2': a2, 'a3': a3, 'm0': m0, 'm1': m1, 'm2': m2}


def svf_lowpass_coeffs(fc: float, Q: float, fs: float) -> dict:
    """SVF lowpass coefficients."""
    g = np.tan(np.pi * fc / fs)
    k = 1.0 / Q
    
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    
    m0 = 0.0
    m1 = 0.0
    m2 = 1.0
    
    return {'g': g, 'k': k, 'A': 1.0, 'a1': a1, 'a2': a2, 'a3': a3, 'm0': m0, 'm1': m1, 'm2': m2}


def svf_process_sample(x: float, coeffs: dict, state: dict) -> Tuple[float, dict]:
    """Process one sample through SVF TPT filter."""
    k = coeffs['k']
    a1, a2, a3 = coeffs['a1'], coeffs['a2'], coeffs['a3']
    m0, m1, m2 = coeffs['m0'], coeffs['m1'], coeffs['m2']
    
    ic1eq = state.get('ic1eq', 0.0)
    ic2eq = state.get('ic2eq', 0.0)
    
    v3 = x - ic2eq
    v1 = a1 * ic1eq + a2 * v3
    v2 = ic2eq + a2 * v1 + a3 * v3
    
    new_ic1eq = 2.0 * v1 - ic1eq
    new_ic2eq = 2.0 * v2 - ic2eq
    
    low = v2
    band = v1
    
    output = m0 * x + m1 * band + m2 * low
    
    return output, {'ic1eq': new_ic1eq, 'ic2eq': new_ic2eq}


def get_svf_frequency_response(coeffs: dict, freqs: np.ndarray, fs: float) -> np.ndarray:
    """Get frequency response of SVF filter by processing impulse."""
    n_samples = int(fs * 0.1)
    impulse = np.zeros(n_samples)
    impulse[0] = 1.0
    
    output = np.zeros(n_samples)
    state = {'ic1eq': 0.0, 'ic2eq': 0.0}
    
    for i in range(n_samples):
        output[i], state = svf_process_sample(impulse[i], coeffs, state)
    
    fft_out = np.fft.rfft(output)
    fft_freqs = np.fft.rfftfreq(n_samples, 1/fs)
    magnitudes = np.interp(freqs, fft_freqs, np.abs(fft_out))
    
    return magnitudes


def test_filter(name: str, scipy_func, svf_func, params: dict, fs: float = SAMPLE_RATE) -> bool:
    """Test a filter by comparing scipy reference to SVF implementation."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Params: {params}")
    print(f"{'='*60}")
    
    if 'gain_db' in params:
        b, a = scipy_func(params['fc'], params['gain_db'], params['Q'], fs)
    else:
        b, a = scipy_func(params['fc'], params['Q'], fs)
    
    freqs = np.logspace(np.log10(20), np.log10(fs/2 * 0.95), 500)
    w, h_scipy = signal.freqz(b, a, worN=freqs, fs=fs)
    scipy_mag_db = 20 * np.log10(np.abs(h_scipy) + 1e-10)
    
    if 'gain_db' in params:
        svf_coeffs = svf_func(params['fc'], params['gain_db'], params['Q'], fs)
    else:
        svf_coeffs = svf_func(params['fc'], params['Q'], fs)
    
    svf_mag = get_svf_frequency_response(svf_coeffs, freqs, fs)
    svf_mag_db = 20 * np.log10(svf_mag + 1e-10)
    
    error_db = svf_mag_db - scipy_mag_db
    max_error = np.max(np.abs(error_db))
    rms_error = np.sqrt(np.mean(error_db**2))
    max_err_idx = np.argmax(np.abs(error_db))
    max_err_freq = freqs[max_err_idx]
    
    # Find target frequency response
    target_freq_idx = np.argmin(np.abs(freqs - params['fc']))
    expected_gain = params.get('gain_db', 0)
    actual_svf_gain = svf_mag_db[target_freq_idx]
    actual_scipy_gain = scipy_mag_db[target_freq_idx]
    
    print(f"Expected gain at fc: {expected_gain:.1f} dB")
    print(f"Scipy gain at fc: {actual_scipy_gain:.2f} dB")
    print(f"SVF gain at fc: {actual_svf_gain:.2f} dB")
    print(f"Max error: {max_error:.4f} dB at {max_err_freq:.0f} Hz")
    print(f"RMS error: {rms_error:.4f} dB")
    
    if max_error > 1.0:
        print("⚠️  WARNING: Error exceeds 1 dB threshold!")
        return False
    else:
        print("✓ PASS: Error within tolerance")
        return True


def run_all_tests():
    """Run all frequency response tests."""
    print("\n" + "="*70)
    print("SVF TPT Frequency Response Validation (FIXED)")
    print("="*70)
    
    results = []
    
    # Peak EQ tests
    peak_tests = [
        {'fc': 1000, 'gain_db': 6, 'Q': 1.0},
        {'fc': 1000, 'gain_db': 12, 'Q': 2.0},
        {'fc': 1000, 'gain_db': -6, 'Q': 1.0},
        {'fc': 1000, 'gain_db': -12, 'Q': 2.0},
        {'fc': 5000, 'gain_db': 6, 'Q': 4.0},
        {'fc': 200, 'gain_db': -12, 'Q': 0.7},
    ]
    
    for params in peak_tests:
        passed = test_filter("Peak EQ", scipy_peak, svf_peak_coeffs, params)
        results.append(('Peak', params, passed))
    
    # Shelf tests
    lowshelf_tests = [
        {'fc': 200, 'gain_db': 6, 'Q': 0.707},
        {'fc': 200, 'gain_db': -6, 'Q': 0.707},
    ]
    
    for params in lowshelf_tests:
        passed = test_filter("Low Shelf", scipy_lowshelf, svf_lowshelf_coeffs, params)
        results.append(('LowShelf', params, passed))
    
    highshelf_tests = [
        {'fc': 8000, 'gain_db': 6, 'Q': 0.707},
        {'fc': 8000, 'gain_db': -6, 'Q': 0.707},
    ]
    
    for params in highshelf_tests:
        passed = test_filter("High Shelf", scipy_highshelf, svf_highshelf_coeffs, params)
        results.append(('HighShelf', params, passed))
    
    # HPF/LPF tests
    hpf_tests = [{'fc': 100, 'Q': 0.707}]
    lpf_tests = [{'fc': 10000, 'Q': 0.707}]
    
    for params in hpf_tests:
        passed = test_filter("Highpass", scipy_highpass, svf_highpass_coeffs, params)
        results.append(('HPF', params, passed))
    
    for params in lpf_tests:
        passed = test_filter("Lowpass", scipy_lowpass, svf_lowpass_coeffs, params)
        results.append(('LPF', params, passed))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r[2])
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed < total:
        print("\nFailed tests:")
        for name, params, p in results:
            if not p:
                print(f"  - {name}: {params}")
        return False
    
    print("\n✓ All tests passed!")
    return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
