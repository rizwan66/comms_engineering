"""
filters/design.py
=================
Digital filter design & analysis.
Covers: FIR (window, Parks-McClellan), IIR (Butterworth, Chebyshev, Elliptic),
        Notch filter, Adaptive LMS, frequency response plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp


# ─────────────────────────────────────────────
# 1. IIR FILTER DESIGN
# ─────────────────────────────────────────────

def butterworth_lpf(cutoff_hz, fs, order=4):
    """Design a Butterworth low-pass filter. Returns (b, a) coefficients."""
    nyq = fs / 2
    Wn  = cutoff_hz / nyq
    b, a = sp.butter(order, Wn, btype='low', analog=False)
    return b, a


def chebyshev1_lpf(cutoff_hz, fs, order=4, ripple_db=1.0):
    """Chebyshev Type-I LPF — equiripple passband."""
    nyq = fs / 2
    b, a = sp.cheby1(order, ripple_db, cutoff_hz/nyq, btype='low')
    return b, a


def chebyshev2_lpf(cutoff_hz, fs, order=4, attenuation_db=40.0):
    """Chebyshev Type-II LPF — equiripple stopband."""
    nyq = fs / 2
    b, a = sp.cheby2(order, attenuation_db, cutoff_hz/nyq, btype='low')
    return b, a


def elliptic_lpf(cutoff_hz, fs, order=4, ripple_db=1.0, attenuation_db=60.0):
    """Elliptic (Cauer) LPF — equiripple in both bands. Most efficient."""
    nyq = fs / 2
    b, a = sp.ellip(order, ripple_db, attenuation_db, cutoff_hz/nyq, btype='low')
    return b, a


def bandpass_filter(low_hz, high_hz, fs, order=4, ftype='butter'):
    """Bandpass filter between low_hz and high_hz."""
    nyq = fs / 2
    Wn  = [low_hz/nyq, high_hz/nyq]
    if ftype == 'butter':
        b, a = sp.butter(order, Wn, btype='band')
    elif ftype == 'cheby1':
        b, a = sp.cheby1(order, 1.0, Wn, btype='band')
    else:
        b, a = sp.ellip(order, 1.0, 60.0, Wn, btype='band')
    return b, a


def notch_filter(notch_hz, fs, Q=30.0):
    """IIR notch filter at notch_hz with quality factor Q."""
    w0 = notch_hz / (fs / 2)
    b, a = sp.iirnotch(w0, Q)
    return b, a


# ─────────────────────────────────────────────
# 2. FIR FILTER DESIGN
# ─────────────────────────────────────────────

def fir_window_lpf(cutoff_hz, fs, num_taps=101, window='hamming'):
    """FIR LPF using window method. Always has linear phase."""
    nyq = fs / 2
    h = sp.firwin(num_taps, cutoff_hz/nyq, window=window)
    return h


def fir_bandpass(low_hz, high_hz, fs, num_taps=201, window='hamming'):
    """FIR bandpass filter via window method."""
    nyq  = fs / 2
    h = sp.firwin(num_taps, [low_hz/nyq, high_hz/nyq], pass_zero=False, window=window)
    return h


def fir_equiripple_lpf(cutoff_hz, stopband_hz, fs, passband_ripple=0.01, stopband_att=60):
    """
    FIR equiripple (Parks-McClellan / Remez) LPF.
    passband_ripple: max ripple in [0,1]
    stopband_att:    attenuation in dB
    """
    nyq = fs / 2
    # Convert dB attenuation to weight ratio
    w_stop = 10 ** (stopband_att / 20)
    # Estimate order
    N, beta = sp.kaiserord(stopband_att, (stopband_hz - cutoff_hz) / nyq)
    N = N | 1  # ensure odd (linear phase Type I)
    h = sp.firwin(N, cutoff_hz / nyq, window=('kaiser', beta))
    return h


# ─────────────────────────────────────────────
# 3. APPLY FILTER
# ─────────────────────────────────────────────

def apply_iir(b, a, signal):
    """Apply IIR filter with zero-phase (forward-backward) filtering."""
    return sp.filtfilt(b, a, signal)


def apply_fir(h, signal):
    """Apply FIR filter via convolution (causal)."""
    return np.convolve(signal, h, mode='same')


# ─────────────────────────────────────────────
# 4. ADAPTIVE LMS FILTER
# ─────────────────────────────────────────────

def lms_filter(x, d, mu=0.01, num_taps=32):
    """
    LMS Adaptive Filter.
    x   : reference/input signal
    d   : desired signal
    mu  : step size
    Returns: (output y, error e, weight history)
    """
    N = len(x)
    w = np.zeros(num_taps)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, num_taps))

    for n in range(num_taps, N):
        x_vec  = x[n:n-num_taps:-1]         # tap delay line
        y[n]   = np.dot(w, x_vec)
        e[n]   = d[n] - y[n]
        w      = w + 2 * mu * e[n] * x_vec  # LMS update
        w_history[n] = w

    return y, e, w_history


def rls_filter(x, d, lam=0.99, num_taps=32):
    """
    RLS Adaptive Filter.
    lam : forgetting factor (0 < λ ≤ 1)
    """
    N   = len(x)
    w   = np.zeros(num_taps)
    P   = np.eye(num_taps) / 0.01    # inverse correlation matrix
    y   = np.zeros(N)
    e   = np.zeros(N)

    for n in range(num_taps, N):
        x_vec = x[n:n-num_taps:-1]
        pi    = P @ x_vec
        K     = pi / (lam + x_vec @ pi)     # Kalman gain
        y[n]  = w @ x_vec
        e[n]  = d[n] - y[n]
        w     = w + K * e[n]
        P     = (P - np.outer(K, x_vec @ P)) / lam

    return y, e


# ─────────────────────────────────────────────
# 5. FREQUENCY RESPONSE PLOTTING
# ─────────────────────────────────────────────

def plot_frequency_response(filters_dict, fs, title="Filter Frequency Responses"):
    """
    Plot multiple filter frequency responses on one figure.
    filters_dict: {'Label': (b, a)} for IIR  or  {'Label': h} for FIR
    """
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(filters_dict)))

    for (label, filt), color in zip(filters_dict.items(), colors):
        if isinstance(filt, tuple):
            b, a = filt
            w, h = sp.freqz(b, a, worN=4096, fs=fs)
        else:
            w, h = sp.freqz(filt, worN=4096, fs=fs)

        mag_db = 20 * np.log10(np.abs(h) + 1e-12)
        phase  = np.unwrap(np.angle(h)) * 180 / np.pi

        ax_mag.plot(w, mag_db, label=label, color=color, lw=1.8)
        ax_phase.plot(w, phase, color=color, lw=1.8)

    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_xlabel("Frequency (Hz)")
    ax_mag.axhline(-3, color='red', ls='--', lw=0.8, label='-3 dB line')
    ax_mag.set_ylim([-80, 5])
    ax_mag.legend(fontsize=9)
    ax_mag.grid(True, alpha=0.3)
    ax_mag.set_title("Magnitude Response")

    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.grid(True, alpha=0.3)
    ax_phase.set_title("Phase Response")

    plt.tight_layout()
    return fig


def plot_pole_zero(b, a, title="Pole-Zero Plot"):
    """Plot poles and zeros on the Z-plane with unit circle."""
    zeros = np.roots(b)
    poles = np.roots(a)

    fig, ax = plt.subplots(figsize=(7, 7))
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, label='Unit Circle')

    ax.scatter(zeros.real, zeros.imag, marker='o', s=80, color='blue', zorder=5, label='Zeros')
    ax.scatter(poles.real, poles.imag, marker='x', s=120, color='red', linewidths=2, zorder=5, label='Poles')

    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from signals.generator import sine_wave, awgn

    fs = 4000
    fc = 300   # cutoff

    print("=" * 55)
    print("  DSP PROJECT — Filters Demo")
    print("=" * 55)

    # Build test signal: 100 Hz (signal) + 800 Hz (interference)
    t, s1 = sine_wave(100, 1.0, duration=0.5, fs=fs)
    t, s2 = sine_wave(800, 0.8, duration=0.5, fs=fs)
    noisy = s1 + s2

    # ── IIR filter comparison ──
    b_butt, a_butt = butterworth_lpf(fc, fs, order=4)
    b_cheb, a_cheb = chebyshev1_lpf(fc, fs, order=4, ripple_db=1)
    b_ell,  a_ell  = elliptic_lpf(fc, fs, order=4)

    fig1 = plot_frequency_response({
        'Butterworth N=4'  : (b_butt, a_butt),
        'Chebyshev-I N=4'  : (b_cheb, a_cheb),
        'Elliptic N=4'     : (b_ell,  a_ell),
    }, fs, title="IIR Filter Comparison (cutoff=300 Hz)")
    fig1.savefig("filters_iir_comparison.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: filters_iir_comparison.png")

    # ── FIR window comparison ──
    h_rect = fir_window_lpf(fc, fs, 101, 'boxcar')
    h_hamm = fir_window_lpf(fc, fs, 101, 'hamming')
    h_blac = fir_window_lpf(fc, fs, 101, 'blackman')

    fig2 = plot_frequency_response({
        'FIR Rectangular': h_rect,
        'FIR Hamming'    : h_hamm,
        'FIR Blackman'   : h_blac,
    }, fs, title="FIR Window Method Comparison (N=101, cutoff=300 Hz)")
    fig2.savefig("filters_fir_comparison.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: filters_fir_comparison.png")

    # ── Filtering demo ──
    filtered = apply_iir(b_butt, a_butt, noisy)
    fig3, axes = plt.subplots(3, 1, figsize=(13, 8))
    fig3.suptitle("Butterworth LPF Filtering Demo", fontsize=14, fontweight='bold')
    axes[0].plot(t, noisy,    color='tomato',      lw=1); axes[0].set_title("Noisy Input (100Hz + 800Hz)"); axes[0].grid(alpha=0.3)
    axes[1].plot(t, filtered, color='steelblue',   lw=1); axes[1].set_title("After Butterworth LPF (300Hz cutoff)"); axes[1].grid(alpha=0.3)
    axes[2].plot(t, s1,       color='mediumseagreen', lw=1); axes[2].set_title("Original 100Hz Signal (reference)"); axes[2].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    plt.tight_layout()
    fig3.savefig("filters_demo.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: filters_demo.png")

    # ── Pole-zero plot ──
    fig4 = plot_pole_zero(b_butt, a_butt, "Butterworth LPF Pole-Zero Plot")
    fig4.savefig("filters_pole_zero.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: filters_pole_zero.png")

    # ── Notch filter demo ──
    t_n, hum = sine_wave(50, 1.0, duration=0.5, fs=fs)   # 50 Hz power line hum
    _, sig_clean = sine_wave(200, 1.0, duration=0.5, fs=fs)
    combined = sig_clean + hum
    b_notch, a_notch = notch_filter(50, fs, Q=30)
    after_notch = apply_iir(b_notch, a_notch, combined)

    fig5, axes = plt.subplots(2, 1, figsize=(13, 6))
    fig5.suptitle("Notch Filter — Removing 50 Hz Power Hum", fontsize=14, fontweight='bold')
    axes[0].plot(t_n, combined,   color='tomato',    lw=1); axes[0].set_title("Signal + 50Hz Hum"); axes[0].grid(alpha=0.3)
    axes[1].plot(t_n, after_notch, color='steelblue', lw=1); axes[1].set_title("After Notch Filter"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig5.savefig("filters_notch.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: filters_notch.png")

    # ── LMS Adaptive Filter ──
    rng = np.random.default_rng(42)
    t_lms = np.arange(1000) / fs
    desired = np.sin(2 * np.pi * 100 * t_lms)
    reference = np.sin(2 * np.pi * 50 * t_lms)        # reference noise
    corrupted = desired + reference
    y_lms, e_lms, _ = lms_filter(reference, corrupted, mu=0.005, num_taps=64)

    fig6, axes = plt.subplots(3, 1, figsize=(13, 8))
    fig6.suptitle("LMS Adaptive Filter — Noise Cancellation", fontsize=14, fontweight='bold')
    axes[0].plot(t_lms, corrupted, color='tomato',    lw=1); axes[0].set_title("Corrupted Input (desired + reference noise)"); axes[0].grid(alpha=0.3)
    axes[1].plot(t_lms, e_lms,     color='steelblue', lw=1); axes[1].set_title("Error Signal (≈ cleaned desired)"); axes[1].grid(alpha=0.3)
    axes[2].plot(t_lms, desired,   color='seagreen',  lw=1); axes[2].set_title("Original Desired (reference)"); axes[2].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    plt.tight_layout()
    fig6.savefig("filters_lms_adaptive.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: filters_lms_adaptive.png")

    print("\n✅ Filters module demo complete — 6 figures saved.")
    plt.close('all')
