"""
Channel Models
==============

    AWGN      : additive white Gaussian noise
    Multipath : tapped-delay-line with specified taps and delays
    Rayleigh  : flat-fading with complex Gaussian envelope
    Rician    : fading with a line-of-sight component (K-factor)
"""
from __future__ import annotations
import numpy as np


def awgn(signal: np.ndarray, snr_db: float, signal_power: float | None = None,
         complex_noise: bool | None = None) -> np.ndarray:
    """
    Add white Gaussian noise at the specified SNR in dB.

    If signal is complex, noise is complex (equal power in real & imag).
    signal_power: override measurement (useful when signal is zero-padded).
    """
    if complex_noise is None:
        complex_noise = np.iscomplexobj(signal)

    if signal_power is None:
        signal_power = np.mean(np.abs(signal) ** 2)

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    if complex_noise:
        noise = (np.random.normal(0, 1, len(signal))
                 + 1j * np.random.normal(0, 1, len(signal))) * np.sqrt(noise_power / 2)
    else:
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    return signal + noise


def multipath_channel(signal: np.ndarray,
                      taps: list[tuple[int, complex]]) -> np.ndarray:
    """
    Apply a static (time-invariant) multipath channel.
    taps: list of (delay_in_samples, complex_gain).
    Example: [(0, 1.0), (5, 0.5), (12, 0.2)] -- 3-tap channel.
    """
    max_delay = max(d for d, _ in taps)
    h = np.zeros(max_delay + 1, dtype=complex)
    for delay, gain in taps:
        h[delay] = gain
    return np.convolve(signal, h, mode="full")[:len(signal)]


def rayleigh_fading(signal: np.ndarray,
                    coherence_samples: int | None = None) -> np.ndarray:
    """
    Flat Rayleigh fading. If coherence_samples is None, the fade is constant
    for the whole block. Otherwise generate a piecewise-constant fade that
    changes every coherence_samples samples (realistic for block fading).
    """
    if coherence_samples is None:
        h = (np.random.normal() + 1j * np.random.normal()) / np.sqrt(2)
        return signal * h

    n_blocks = int(np.ceil(len(signal) / coherence_samples))
    fades = (np.random.normal(size=n_blocks)
             + 1j * np.random.normal(size=n_blocks)) / np.sqrt(2)
    # Expand each fade across its block
    h = np.repeat(fades, coherence_samples)[:len(signal)]
    return signal * h


def rician_fading(signal: np.ndarray, K_db: float,
                  coherence_samples: int | None = None) -> np.ndarray:
    """
    Rician fading with K-factor in dB. K = P_LOS / P_scatter.
    K_db -> infinity  : AWGN-like (pure LOS)
    K_db -> -infinity : Rayleigh
    """
    K = 10 ** (K_db / 10)
    los = np.sqrt(K / (K + 1))
    scatter_std = np.sqrt(1 / (2 * (K + 1)))

    if coherence_samples is None:
        h = los + scatter_std * (np.random.normal() + 1j * np.random.normal())
        return signal * h

    n_blocks = int(np.ceil(len(signal) / coherence_samples))
    scatter = (np.random.normal(size=n_blocks)
               + 1j * np.random.normal(size=n_blocks)) * scatter_std
    fades = los + scatter
    h = np.repeat(fades, coherence_samples)[:len(signal)]
    return signal * h


def frequency_offset(signal: np.ndarray, fs: float, f_off: float) -> np.ndarray:
    """Apply a carrier-frequency offset of f_off Hz (complex baseband)."""
    t = np.arange(len(signal)) / fs
    return signal * np.exp(1j * 2 * np.pi * f_off * t)


def phase_noise(signal: np.ndarray, std_rad: float) -> np.ndarray:
    """Apply a random Wiener-process phase drift."""
    phi = np.cumsum(np.random.normal(0, std_rad, len(signal)))
    return signal * np.exp(1j * phi)
