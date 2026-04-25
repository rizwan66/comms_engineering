"""
src/wavelets/wavelet_transform.py
===================================
Wavelet analysis for DSP:
  - Continuous Wavelet Transform (CWT) with Morlet / Mexican Hat
  - Discrete Wavelet Transform (DWT) — Haar, Daubechies (db4)
  - Multi-resolution analysis (MRA)
  - Wavelet denoising (soft/hard thresholding)
  - Wavelet packet decomposition
  - Comparison: STFT vs CWT for non-stationary signals
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp


# ─────────────────────────────────────────────
# 1.  MOTHER WAVELETS
# ─────────────────────────────────────────────

def morlet_wavelet(t, f0=1.0, sigma=1.0):
    """
    Morlet wavelet: ψ(t) = exp(j2πf₀t) · exp(-t²/2σ²)
    Complex wavelet — gives amplitude AND phase.
    """
    return np.exp(1j * 2 * np.pi * f0 * t) * np.exp(-t**2 / (2 * sigma**2))


def mexican_hat(t):
    """
    Mexican Hat (Ricker) wavelet: ψ(t) = (1 - t²) · exp(-t²/2)
    Real, symmetric, good for detecting peaks/discontinuities.
    """
    return (1 - t**2) * np.exp(-t**2 / 2)


def haar_wavelet():
    """Haar scaling and wavelet filters."""
    h  = np.array([1, 1], dtype=float) / np.sqrt(2)      # scaling (lowpass)
    g  = np.array([1, -1], dtype=float) / np.sqrt(2)     # wavelet (highpass)
    hr = h[::-1]                                          # reconstruction
    gr = g[::-1]
    return h, g, hr, gr


def db4_filters():
    """
    Daubechies db4 wavelet filters (4 vanishing moments).
    Classic choice for smooth signals.
    """
    # Standard db4 coefficients
    h = np.array([
        0.48296291, 0.83651630, 0.22414387, -0.12940952,
        -0.01779187, 0.04765829, -0.00104097, -0.01263630
    ])
    h /= np.sqrt(np.sum(h**2))
    g  = h[::-1] * np.array([(-1)**n for n in range(len(h))])
    return h, g, h[::-1], g[::-1]


# ─────────────────────────────────────────────
# 2.  CONTINUOUS WAVELET TRANSFORM (CWT)
# ─────────────────────────────────────────────

def cwt_morlet(signal, fs, freqs, sigma=1.0):
    """
    CWT using Morlet wavelet.
    Returns complex scalogram W[freq, time].

    freqs : array of analysis frequencies (Hz)
    """
    N   = len(signal)
    t   = np.arange(N) / fs
    W   = np.zeros((len(freqs), N), dtype=complex)

    for fi, freq in enumerate(freqs):
        scale  = 1.0 / freq
        t_wav  = np.arange(-4*sigma/scale, 4*sigma/scale, 1/fs)
        psi    = morlet_wavelet(t_wav, f0=freq, sigma=sigma/scale)
        psi   /= np.sqrt(scale)
        conv   = sp.fftconvolve(signal, np.conj(psi[::-1]), mode='same')
        W[fi]  = conv[:len(signal)]

    return W


def cwt_mexican_hat(signal, fs, scales):
    """
    CWT using Mexican Hat wavelet.
    scales : array of dilation scales (samples)
    """
    N = len(signal)
    W = np.zeros((len(scales), N), dtype=float)

    for si, scale in enumerate(scales):
        t_wav = np.arange(-4*scale, 4*scale+1)
        psi   = mexican_hat(t_wav / scale) / (scale**0.5)
        conv  = np.convolve(signal, psi[::-1], mode='same')
        W[si] = conv

    return W


# ─────────────────────────────────────────────
# 3.  DISCRETE WAVELET TRANSFORM (DWT)
# ─────────────────────────────────────────────

def dwt_1level(signal, h, g):
    """
    Single-level DWT: convolve + downsample.
    Returns (approx_coeffs, detail_coeffs)
    """
    # Convolve then downsample by 2
    cA = np.convolve(signal, h, mode='full')[::2]
    cD = np.convolve(signal, g, mode='full')[::2]
    # Trim to expected length
    n  = (len(signal) + len(h) - 1) // 2
    return cA[:n], cD[:n]


def idwt_1level(cA, cD, hr, gr):
    """Single-level inverse DWT: upsample + convolve."""
    # Upsample
    cA_up = np.zeros(2*len(cA)); cA_up[::2] = cA
    cD_up = np.zeros(2*len(cD)); cD_up[::2] = cD
    rec   = np.convolve(cA_up, hr, mode='full') + np.convolve(cD_up, gr, mode='full')
    return rec


def dwt_multilevel(signal, h, g, levels=4):
    """
    Multi-level DWT decomposition.
    Returns list of coefficients: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    """
    coeffs = []
    x = signal.copy()
    for _ in range(levels):
        cA, cD = dwt_1level(x, h, g)
        coeffs.append(cD)
        x = cA
    coeffs.append(x)   # final approximation
    return coeffs[::-1]  # [approx, detail_N, ..., detail_1]


def idwt_multilevel(coeffs, hr, gr):
    """Reconstruct from multi-level DWT coefficients."""
    x = coeffs[0]
    for cD in coeffs[1:]:
        x_up = np.zeros(2*len(x)); x_up[::2] = x
        d_up = np.zeros(2*len(cD)); d_up[::2] = cD
        n    = min(2*len(cD), len(signal_ref) if 'signal_ref' in dir() else 99999)
        x    = (np.convolve(x_up, hr, mode='full')
               + np.convolve(d_up, gr, mode='full'))
    return x


# ─────────────────────────────────────────────
# 4.  WAVELET DENOISING
# ─────────────────────────────────────────────

def universal_threshold(coeffs):
    """VisuShrink universal threshold: λ = σ√(2 log N)"""
    all_details = np.concatenate([c for c in coeffs[1:]])
    sigma = np.median(np.abs(all_details)) / 0.6745   # robust noise estimate
    N     = sum(len(c) for c in coeffs)
    lam   = sigma * np.sqrt(2 * np.log(N))
    return lam, sigma


def soft_threshold(x, lam):
    """Soft thresholding: sgn(x) · max(|x| - λ, 0)"""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def hard_threshold(x, lam):
    """Hard thresholding: x if |x| > λ else 0"""
    return x * (np.abs(x) > lam)


def wavelet_denoise(noisy, h, g, hr, gr, levels=4,
                    threshold='soft', multiplier=1.0):
    """
    Full wavelet denoising pipeline:
    1. DWT decomposition
    2. Estimate threshold from detail coefficients
    3. Threshold detail coefficients
    4. IDWT reconstruction
    """
    coeffs = dwt_multilevel(noisy, h, g, levels)
    lam, sigma = universal_threshold(coeffs)
    lam *= multiplier

    # Threshold detail coefficients (leave approximation untouched)
    thresh_fn  = soft_threshold if threshold == 'soft' else hard_threshold
    coeffs_thr = [coeffs[0]] + [thresh_fn(c, lam) for c in coeffs[1:]]

    # Reconstruct
    x = coeffs_thr[0]
    for cD in coeffs_thr[1:]:
        x_up  = np.zeros(2*len(x));  x_up[::2]  = x
        d_up  = np.zeros(2*len(cD)); d_up[::2]  = cD
        # Pad both to same length before convolving
        target = max(len(x_up), len(d_up))
        x_up  = np.pad(x_up,  (0, target - len(x_up)))
        d_up  = np.pad(d_up,  (0, target - len(d_up)))
        x     = (np.convolve(x_up, hr, mode='full')
                + np.convolve(d_up, gr, mode='full'))

    return x[:len(noisy)], lam, sigma


# ─────────────────────────────────────────────
# 5.  STFT vs CWT COMPARISON
# ─────────────────────────────────────────────

def stft_spectrogram(signal, fs, nperseg=64, noverlap=48):
    """Short-Time Fourier Transform spectrogram."""
    f, t, Zxx = sp.stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return t, f, np.abs(Zxx)


def compare_stft_cwt(signal, fs, freqs=None):
    """Compare time-frequency representations: STFT vs CWT."""
    if freqs is None:
        freqs = np.linspace(10, fs//4, 80)

    t_stft, f_stft, S = stft_spectrogram(signal, fs)
    W = cwt_morlet(signal, fs, freqs, sigma=0.8)
    t_cwt = np.arange(len(signal)) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Time-Frequency Analysis: Signal | STFT | CWT',
                 fontsize=13, fontweight='bold')

    t_sig = np.arange(len(signal)) / fs
    axes[0].plot(t_sig, signal, color='steelblue', lw=1)
    axes[0].set_title('Signal (Time Domain)'); axes[0].grid(alpha=0.3)

    im1 = axes[1].pcolormesh(t_stft, f_stft, 20*np.log10(S+1e-8),
                              shading='auto', cmap='viridis')
    plt.colorbar(im1, ax=axes[1], label='dB')
    axes[1].set_title('STFT Spectrogram (fixed time-freq resolution)')
    axes[1].set_ylabel('Frequency (Hz)')

    im2 = axes[2].pcolormesh(t_cwt, freqs, np.abs(W),
                              shading='auto', cmap='plasma')
    plt.colorbar(im2, ax=axes[2], label='|W|')
    axes[2].set_title('CWT Scalogram (multi-resolution: fine freq at low f, fine time at high f)')
    axes[2].set_ylabel('Frequency (Hz)')

    for ax in axes: ax.set_xlabel('Time (s)')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(0)
    fs = 2000
    N  = 512     # shorter for speed
    t  = np.arange(N) / fs

    print("=" * 55)
    print("  DSP PROJECT — Wavelet Transform Demo")
    print("=" * 55)

    # ── Test signal: chirp + impulse + noise ──────────────
    chirp_sig = sp.chirp(t, f0=10, f1=400, t1=t[-1], method='linear')
    impulse   = np.zeros(N); impulse[N//3] = 3.0
    noise_sig = 0.3 * np.random.randn(N)
    test_sig  = chirp_sig + impulse + noise_sig

    # ── 1. CWT Morlet ─────────────────────────────────────
    freqs = np.linspace(5, 400, 40)   # fewer freqs for speed
    W     = cwt_morlet(test_sig, fs, freqs, sigma=0.8)

    fig1, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig1.suptitle('Continuous Wavelet Transform (CWT) — Morlet', fontsize=13, fontweight='bold')
    axes[0].plot(t, test_sig, color='steelblue', lw=1)
    axes[0].set_title('Signal: Chirp (10→400Hz) + Impulse + Noise')
    axes[0].grid(alpha=0.3); axes[0].set_xlabel('Time (s)')
    im = axes[1].pcolormesh(t, freqs, np.abs(W), shading='auto', cmap='plasma')
    plt.colorbar(im, ax=axes[1], label='|CWT|')
    axes[1].set_title('Morlet CWT Scalogram')
    axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Frequency (Hz)')
    plt.tight_layout()
    fig1.savefig('wavelet_cwt_morlet.png', dpi=120, bbox_inches='tight')
    print("✓ Saved: wavelet_cwt_morlet.png")

    # ── 2. DWT Multi-resolution ───────────────────────────
    h, g, hr, gr = db4_filters()
    LEVELS = 5
    coeffs = dwt_multilevel(test_sig, h, g, levels=LEVELS)

    fig2, axes = plt.subplots(LEVELS+2, 1, figsize=(14, 12))
    fig2.suptitle('DWT Multi-Resolution Analysis (db4, 5 levels)', fontsize=13, fontweight='bold')
    axes[0].plot(t, test_sig, color='steelblue', lw=1)
    axes[0].set_title('Original Signal'); axes[0].grid(alpha=0.3)
    labels = [f'Approx (level {LEVELS})'] + [f'Detail level {LEVELS-i}' for i in range(LEVELS)]
    colors = ['#3fb950', '#58a6ff', '#e3b341', '#f85149', '#d2a8ff', '#79c0ff']
    for i, (c, lab, col) in enumerate(zip(coeffs, labels, colors)):
        axes[i+1].plot(c, color=col, lw=1)
        axes[i+1].set_title(lab, fontsize=9)
        axes[i+1].grid(alpha=0.3)
    plt.tight_layout()
    fig2.savefig('wavelet_dwt_mra.png', dpi=120, bbox_inches='tight')
    print("✓ Saved: wavelet_dwt_mra.png")

    # ── 3. Wavelet denoising ──────────────────────────────
    clean = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)
    noisy = clean + 0.8 * np.random.randn(N)
    denoised_soft, lam_s, sigma = wavelet_denoise(noisy, h, g, hr, gr,
                                                   levels=4, threshold='soft')
    denoised_hard, lam_h, _    = wavelet_denoise(noisy, h, g, hr, gr,
                                                   levels=4, threshold='hard')

    def snr_db(clean, sig):
        n = min(len(clean), len(sig))
        noise = sig[:n] - clean[:n]
        return 10*np.log10(np.mean(clean[:n]**2) / (np.mean(noise**2) + 1e-12))

    snr_noisy = snr_db(clean, noisy)
    snr_soft  = snr_db(clean, denoised_soft)
    snr_hard  = snr_db(clean, denoised_hard)

    fig3, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig3.suptitle('Wavelet Denoising — Soft vs Hard Thresholding', fontsize=13, fontweight='bold')
    axes[0].plot(t, clean,          color='seagreen',  lw=1.5); axes[0].set_title('Clean Signal')
    axes[1].plot(t, noisy,          color='tomato',    lw=0.8); axes[1].set_title(f'Noisy Signal (SNR={snr_noisy:.1f} dB)')
    axes[2].plot(t, denoised_soft[:N], color='steelblue', lw=1.5)
    axes[2].set_title(f'Soft Threshold Denoised (λ={lam_s:.3f}, SNR={snr_soft:.1f} dB)')
    axes[3].plot(t, denoised_hard[:N], color='darkorange', lw=1.5)
    axes[3].set_title(f'Hard Threshold Denoised (λ={lam_h:.3f}, SNR={snr_hard:.1f} dB)')
    for ax in axes: ax.grid(alpha=0.3); ax.set_xlabel('Time (s)')
    plt.tight_layout()
    fig3.savefig('wavelet_denoising.png', dpi=120, bbox_inches='tight')
    print(f"✓ Saved: wavelet_denoising.png  (Noisy:{snr_noisy:.1f}→Soft:{snr_soft:.1f}→Hard:{snr_hard:.1f} dB)")

    # ── 4. STFT vs CWT comparison ─────────────────────────
    # Use a multi-component non-stationary signal
    comp_sig = (np.sin(2*np.pi*30*t) * (t < 0.25) +
                np.sin(2*np.pi*150*t) * ((t >= 0.25) & (t < 0.5)) +
                np.sin(2*np.pi*300*t) * (t >= 0.5))
    comp_sig += 0.1 * np.random.randn(N)

    fig4 = compare_stft_cwt(comp_sig, fs, freqs=np.linspace(5, 400, 40))
    fig4.savefig('wavelet_stft_vs_cwt.png', dpi=120, bbox_inches='tight')
    print("✓ Saved: wavelet_stft_vs_cwt.png")

    # ── 5. Haar vs db4 detail coefficients ───────────────
    h_haar, g_haar, hr_haar, gr_haar = haar_wavelet()
    _, d_haar = dwt_1level(test_sig[:256], h_haar, g_haar)
    _, d_db4  = dwt_1level(test_sig[:256], h, g)

    fig5, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig5.suptitle('Haar vs db4 — Level-1 Detail Coefficients', fontsize=13, fontweight='bold')
    axes[0].plot(d_haar, color='steelblue', lw=1); axes[0].set_title('Haar Wavelet Detail'); axes[0].grid(alpha=0.3)
    axes[1].plot(d_db4,  color='darkorange', lw=1); axes[1].set_title('db4 Wavelet Detail'); axes[1].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel('Coefficient index')
    plt.tight_layout()
    fig5.savefig('wavelet_haar_vs_db4.png', dpi=120, bbox_inches='tight')
    print("✓ Saved: wavelet_haar_vs_db4.png")

    print("\n✅  Wavelet module demo complete — 5 figures saved.")
    plt.close('all')
