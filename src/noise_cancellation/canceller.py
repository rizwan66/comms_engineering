"""
noise_cancellation/canceller.py
================================
Noise cancellation algorithms:
  - Spectral Subtraction
  - Wiener Filter
  - LMS / RLS (adaptive)
  - Active Noise Cancellation (ANC) simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp


# ─────────────────────────────────────────────
# 1. SPECTRAL SUBTRACTION
# ─────────────────────────────────────────────

def spectral_subtraction(noisy, fs, noise_frames=10, alpha=1.0, beta=0.002,
                          frame_size=512, hop=256):
    """
    Spectral subtraction for speech/audio noise reduction.
    noisy      : noisy signal
    alpha      : over-subtraction factor (>1 → more aggressive)
    beta       : spectral floor (prevents musical noise)
    """
    window = np.hanning(frame_size)
    num_frames = (len(noisy) - frame_size) // hop
    output = np.zeros(len(noisy))
    norm   = np.zeros(len(noisy))

    # Estimate noise PSD from first noise_frames frames
    rfft_len  = frame_size // 2 + 1
    noise_psd = np.zeros(rfft_len)
    for i in range(noise_frames):
        frame     = noisy[i*hop : i*hop + frame_size] * window
        noise_psd += np.abs(np.fft.rfft(frame, n=frame_size))**2
    noise_psd /= noise_frames

    for i in range(num_frames):
        start = i * hop
        frame = noisy[start : start + frame_size] * window

        X     = np.fft.rfft(frame, n=frame_size)
        mag   = np.abs(X)
        phase = np.angle(X)

        # Subtract noise PSD
        mag_clean = np.sqrt(np.maximum(mag**2 - alpha * noise_psd,
                                        beta * mag**2))

        X_clean = mag_clean * np.exp(1j * phase)
        frame_clean = np.fft.irfft(X_clean, n=frame_size) * window

        output[start:start+frame_size] += frame_clean
        norm[start:start+frame_size]   += window**2

    norm = np.where(norm > 1e-8, norm, 1.0)
    return output / norm


# ─────────────────────────────────────────────
# 2. WIENER FILTER (freq domain)
# ─────────────────────────────────────────────

def wiener_filter_freq(noisy, signal_psd_estimate, noise_psd_estimate):
    """
    Frequency-domain Wiener filter.
    H(ω) = Sss(ω) / (Sss(ω) + Snn(ω))
    """
    X = np.fft.rfft(noisy)
    H = signal_psd_estimate / (signal_psd_estimate + noise_psd_estimate + 1e-10)
    return np.fft.irfft(H * X, n=len(noisy))


def estimate_psd(signal, frame_size=512, hop=256):
    """Estimate PSD via averaging magnitude squared FFT frames."""
    N      = len(signal)
    window = np.hanning(frame_size)
    n_frames = max(1, (N - frame_size) // hop)
    psd = np.zeros(frame_size // 2 + 1)
    for i in range(n_frames):
        frame = signal[i*hop : i*hop + frame_size] * window
        psd  += np.abs(np.fft.rfft(frame))**2
    return psd / n_frames


# ─────────────────────────────────────────────
# 3. LMS ADAPTIVE NOISE CANCELLER
# ─────────────────────────────────────────────

def lms_anc(reference, corrupted, mu=0.005, num_taps=64):
    """
    Active Noise Cancellation using LMS.
    reference : reference noise signal (from secondary microphone)
    corrupted : primary microphone (desired + noise)
    Returns cleaned signal e[n] = corrupted - y[n]
    """
    N = len(corrupted)
    w = np.zeros(num_taps)
    y = np.zeros(N)
    e = np.zeros(N)

    for n in range(num_taps, N):
        x_vec = reference[n:n-num_taps:-1]
        y[n]  = np.dot(w, x_vec)
        e[n]  = corrupted[n] - y[n]
        w     = w + 2 * mu * e[n] * x_vec

    return e, y


def rls_anc(reference, corrupted, lam=0.99, num_taps=64):
    """Active Noise Cancellation using RLS — faster convergence."""
    N  = len(corrupted)
    w  = np.zeros(num_taps)
    P  = np.eye(num_taps) / 0.01
    y  = np.zeros(N)
    e  = np.zeros(N)

    for n in range(num_taps, N):
        x_vec = reference[n:n-num_taps:-1]
        pi    = P @ x_vec
        K     = pi / (lam + x_vec @ pi)
        y[n]  = w @ x_vec
        e[n]  = corrupted[n] - y[n]
        w     = w + K * e[n]
        P     = (P - np.outer(K, x_vec @ P)) / lam

    return e, y


# ─────────────────────────────────────────────
# 4. SNR CALCULATION
# ─────────────────────────────────────────────

def compute_snr(clean, noisy_or_output):
    """SNR = 10·log10(P_signal / P_noise_residual)"""
    noise  = noisy_or_output - clean[:len(noisy_or_output)]
    p_sig  = np.mean(clean**2)
    p_noise = np.mean(noise**2) + 1e-12
    return 10 * np.log10(p_sig / p_noise)


# ─────────────────────────────────────────────
# 5. LEARNING CURVES
# ─────────────────────────────────────────────

def learning_curve(e, smooth=50):
    """Compute smoothed squared error learning curve."""
    mse = e**2
    # Moving average
    kernel = np.ones(smooth) / smooth
    return np.convolve(mse, kernel, mode='same')


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    fs  = 4000
    N   = 4000

    print("=" * 55)
    print("  DSP PROJECT — Noise Cancellation Demo")
    print("=" * 55)

    t = np.arange(N) / fs

    # Clean desired signal: speech-like sum of tones
    clean = (np.sin(2*np.pi*200*t) + 0.5*np.sin(2*np.pi*350*t)
           + 0.3*np.sin(2*np.pi*500*t))
    clean /= np.max(np.abs(clean))

    # Reference noise: 50 Hz hum + broadband
    hum       = np.sin(2*np.pi*50*t)
    broadband = 0.3 * rng.standard_normal(N)
    noise     = hum + broadband

    corrupted = clean + 0.8 * noise

    # ── Spectral Subtraction ──
    ss_out = spectral_subtraction(corrupted, fs, noise_frames=8, alpha=1.5)
    snr_in  = compute_snr(clean, corrupted)
    snr_ss  = compute_snr(clean[:len(ss_out)], ss_out)

    fig1, axes = plt.subplots(3, 1, figsize=(13, 9))
    fig1.suptitle("Spectral Subtraction Noise Reduction", fontsize=14, fontweight='bold')
    axes[0].plot(t, clean,     color='seagreen',  lw=1); axes[0].set_title("Clean Signal"); axes[0].grid(alpha=0.3)
    axes[1].plot(t, corrupted, color='tomato',     lw=1); axes[1].set_title(f"Noisy Input  (SNR={snr_in:.1f} dB)"); axes[1].grid(alpha=0.3)
    axes[2].plot(t[:len(ss_out)], ss_out, color='steelblue', lw=1)
    axes[2].set_title(f"After Spectral Subtraction (SNR≈{snr_ss:.1f} dB)"); axes[2].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel("Time (s)")
    plt.tight_layout()
    fig1.savefig("nc_spectral_subtraction.png", dpi=120, bbox_inches='tight')
    print(f"✓ Saved: nc_spectral_subtraction.png  (SNR: {snr_in:.1f} → {snr_ss:.1f} dB)")

    # ── LMS ANC ──
    reference = hum + 0.05 * rng.standard_normal(N)    # reference mic (mostly hum)
    e_lms, y_lms = lms_anc(reference, corrupted, mu=0.002, num_taps=64)
    snr_lms = compute_snr(clean, e_lms)
    lc_lms  = learning_curve(e_lms, smooth=100)

    # ── RLS ANC ──
    e_rls, y_rls = rls_anc(reference, corrupted, lam=0.995, num_taps=64)
    snr_rls = compute_snr(clean, e_rls)
    lc_rls  = learning_curve(e_rls, smooth=100)

    fig2, axes = plt.subplots(4, 1, figsize=(13, 12))
    fig2.suptitle("Adaptive ANC — LMS vs RLS", fontsize=14, fontweight='bold')
    axes[0].plot(t, corrupted, color='tomato',     lw=1); axes[0].set_title(f"Corrupted Input (SNR={snr_in:.1f} dB)"); axes[0].grid(alpha=0.3)
    axes[1].plot(t, e_lms,     color='steelblue',  lw=1); axes[1].set_title(f"LMS Output (SNR≈{snr_lms:.1f} dB, μ=0.002)"); axes[1].grid(alpha=0.3)
    axes[2].plot(t, e_rls,     color='darkorange',  lw=1); axes[2].set_title(f"RLS Output (SNR≈{snr_rls:.1f} dB, λ=0.995)"); axes[2].grid(alpha=0.3)
    axes[3].plot(t, lc_lms,    color='steelblue',  lw=1.5, label='LMS')
    axes[3].plot(t, lc_rls,    color='darkorange',  lw=1.5, label='RLS', ls='--')
    axes[3].set_title("Learning Curves (MSE)"); axes[3].set_yscale('log')
    axes[3].legend(); axes[3].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel("Time (s)")
    plt.tight_layout()
    fig2.savefig("nc_lms_rls.png", dpi=120, bbox_inches='tight')
    print(f"✓ Saved: nc_lms_rls.png  (LMS SNR:{snr_lms:.1f} dB | RLS SNR:{snr_rls:.1f} dB)")

    # ── SNR comparison bar chart ──
    fig3, ax = plt.subplots(figsize=(8, 5))
    methods = ['Noisy Input', 'Spectral Sub.', 'LMS ANC', 'RLS ANC']
    snrs    = [snr_in, snr_ss, snr_lms, snr_rls]
    colors  = ['tomato', 'mediumseagreen', 'steelblue', 'darkorange']
    bars = ax.bar(methods, snrs, color=colors, edgecolor='k', linewidth=0.5)
    ax.bar_label(bars, fmt='%.1f dB', padding=3)
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Noise Cancellation SNR Comparison")
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig3.savefig("nc_snr_comparison.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: nc_snr_comparison.png")

    print("\n✅ Noise Cancellation module demo complete — 3 figures saved.")
    plt.close('all')
