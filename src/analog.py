"""
Analog Modulation
=================

Classic analog schemes, implemented from first principles:

    AM    (Amplitude Modulation with carrier):  s(t) = A_c [1 + mu*m(t)] cos(2*pi*fc*t)
    DSB-SC (Double-SideBand Suppressed Carrier): s(t) = A_c * m(t) * cos(2*pi*fc*t)
    SSB   (Single SideBand, upper):              s(t) = m(t)cos(wc t) - m_hat(t)sin(wc t)
    FM    (Frequency Modulation):                s(t) = A_c cos(2*pi*fc*t + 2*pi*kf*integral(m))

Each modulator has a matching demodulator. All signals are represented as
real-valued numpy arrays sampled at fs Hz.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import hilbert, butter, filtfilt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 6) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 6) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


# ---------------------------------------------------------------------------
# AM with carrier
# ---------------------------------------------------------------------------

def am_modulate(message: np.ndarray, fc: float, fs: float,
                modulation_index: float = 0.7,
                carrier_amplitude: float = 1.0) -> np.ndarray:
    """
    Standard AM. modulation_index (mu) must be in (0, 1] for no over-modulation.
    Message should be normalized to [-1, 1] before calling.
    """
    t = np.arange(len(message)) / fs
    return carrier_amplitude * (1 + modulation_index * message) * np.cos(2 * np.pi * fc * t)


def am_demodulate_envelope(signal: np.ndarray, fs: float,
                           cutoff: float | None = None) -> np.ndarray:
    """Envelope detector via the analytic signal. Removes DC offset."""
    envelope = np.abs(hilbert(signal))
    if cutoff is not None:
        envelope = lowpass(envelope, fs, cutoff)
    return envelope - np.mean(envelope)


# ---------------------------------------------------------------------------
# DSB-SC
# ---------------------------------------------------------------------------

def dsb_sc_modulate(message: np.ndarray, fc: float, fs: float) -> np.ndarray:
    t = np.arange(len(message)) / fs
    return message * np.cos(2 * np.pi * fc * t)


def dsb_sc_demodulate(signal: np.ndarray, fc: float, fs: float,
                      cutoff: float) -> np.ndarray:
    """Coherent (synchronous) detection: multiply by carrier, low-pass, x2 gain."""
    t = np.arange(len(signal)) / fs
    return 2 * lowpass(signal * np.cos(2 * np.pi * fc * t), fs, cutoff)


# ---------------------------------------------------------------------------
# SSB (upper sideband) using Hilbert method
# ---------------------------------------------------------------------------

def ssb_modulate(message: np.ndarray, fc: float, fs: float,
                 sideband: str = "upper") -> np.ndarray:
    """
    Phasing (Hilbert) method:
        USB: s(t) = m(t)cos(wc*t) - m_hat(t)sin(wc*t)
        LSB: s(t) = m(t)cos(wc*t) + m_hat(t)sin(wc*t)
    """
    t = np.arange(len(message)) / fs
    m_hat = np.imag(hilbert(message))  # Hilbert transform of message
    sign = -1 if sideband.lower() == "upper" else +1
    return (message * np.cos(2 * np.pi * fc * t)
            + sign * m_hat * np.sin(2 * np.pi * fc * t))


def ssb_demodulate(signal: np.ndarray, fc: float, fs: float,
                   cutoff: float) -> np.ndarray:
    """Coherent detection identical to DSB-SC."""
    return dsb_sc_demodulate(signal, fc, fs, cutoff)


# ---------------------------------------------------------------------------
# FM
# ---------------------------------------------------------------------------

def fm_modulate(message: np.ndarray, fc: float, fs: float,
                kf: float, carrier_amplitude: float = 1.0) -> np.ndarray:
    """
    FM with frequency sensitivity kf (Hz/V).
    Instantaneous phase: phi(t) = 2*pi*fc*t + 2*pi*kf * integral(m(t) dt)
    """
    t = np.arange(len(message)) / fs
    integrated = np.cumsum(message) / fs
    return carrier_amplitude * np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integrated)


def fm_demodulate(signal: np.ndarray, fs: float, kf: float,
                  cutoff: float | None = None) -> np.ndarray:
    """
    Demodulate via the instantaneous phase derivative of the analytic signal.
    m_hat(t) = (1/(2*pi*kf)) * d/dt [unwrapped phase]
    """
    analytic = hilbert(signal)
    phase = np.unwrap(np.angle(analytic))
    # Subtract the carrier's linear phase first for numerical stability
    # but we can also just differentiate directly; the carrier component
    # becomes a DC bias that we remove.
    inst_freq = np.diff(phase) * fs / (2 * np.pi)
    inst_freq = np.concatenate([[inst_freq[0]], inst_freq])
    # Remove carrier DC component (mean)
    message_hat = (inst_freq - np.mean(inst_freq)) / kf
    if cutoff is not None:
        message_hat = lowpass(message_hat, fs, cutoff)
    return message_hat


# ---------------------------------------------------------------------------
# Convenience: Carson's rule for FM bandwidth
# ---------------------------------------------------------------------------

def carson_bandwidth(kf: float, peak_message: float, f_max: float) -> float:
    """B ≈ 2 * (delta_f + f_max), where delta_f = kf * |m|_peak."""
    delta_f = kf * peak_message
    return 2 * (delta_f + f_max)
