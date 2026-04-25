"""
signals/generator.py
====================
Signal generation and analysis for DSP learning.
Covers: sine, square, sawtooth, chirp, noise, FFT, DTFT, convolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# 1. SIGNAL GENERATORS
# ─────────────────────────────────────────────

def sine_wave(freq=1.0, amplitude=1.0, phase=0.0, duration=1.0, fs=1000):
    """Generate a sine wave: x(t) = A·sin(2πft + φ)"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    x = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return t, x


def square_wave(freq=1.0, amplitude=1.0, duty=0.5, duration=1.0, fs=1000):
    """Generate a square wave using Fourier series (first 50 harmonics)."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    x = np.zeros_like(t)
    for n in range(1, 51, 2):                        # odd harmonics only
        x += (4 / (n * np.pi)) * np.sin(2 * np.pi * n * freq * t)
    return t, amplitude * x / np.max(np.abs(x))


def sawtooth_wave(freq=1.0, amplitude=1.0, duration=1.0, fs=1000):
    """Generate a sawtooth wave."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    x = 2 * (t * freq - np.floor(0.5 + t * freq))
    return t, amplitude * x


def chirp(f_start=10, f_end=500, duration=1.0, fs=4000, amplitude=1.0):
    """Linear chirp: frequency sweeps from f_start to f_end."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    k = (f_end - f_start) / duration
    x = amplitude * np.sin(2 * np.pi * (f_start * t + 0.5 * k * t**2))
    return t, x


def awgn(signal, snr_db):
    """Add Additive White Gaussian Noise to achieve desired SNR (dB)."""
    sig_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise, noise


def unit_impulse(N=64, delay=0):
    """Discrete unit impulse δ[n - delay]."""
    x = np.zeros(N)
    x[delay] = 1.0
    return np.arange(N), x


def unit_step(N=64, start=0):
    """Discrete unit step u[n - start]."""
    x = np.zeros(N)
    x[start:] = 1.0
    return np.arange(N), x


# ─────────────────────────────────────────────
# 2. SPECTRAL ANALYSIS
# ─────────────────────────────────────────────

def compute_fft(signal, fs):
    """Compute single-sided FFT magnitude spectrum."""
    N = len(signal)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/fs)
    # Single-sided
    half = N // 2
    mag = (2.0 / N) * np.abs(X[:half])
    return freqs[:half], mag


def compute_psd(signal, fs):
    """Power Spectral Density via Welch's method (simple periodogram here)."""
    N = len(signal)
    X = np.fft.fft(signal * np.hanning(N))
    freqs = np.fft.fftfreq(N, d=1/fs)
    psd = (np.abs(X)**2) / (N * fs)
    half = N // 2
    return freqs[:half], psd[:half]


def spectrogram(signal, fs, window_size=256, hop=128):
    """Compute and return STFT spectrogram data."""
    window = np.hanning(window_size)
    num_frames = (len(signal) - window_size) // hop + 1
    S = np.zeros((window_size // 2, num_frames))
    for i in range(num_frames):
        frame = signal[i*hop : i*hop + window_size] * window
        S[:, i] = np.abs(np.fft.fft(frame))[:window_size // 2]
    times = np.arange(num_frames) * hop / fs
    freqs = np.fft.fftfreq(window_size, 1/fs)[:window_size // 2]
    return times, freqs, 20 * np.log10(S + 1e-10)


# ─────────────────────────────────────────────
# 3. CONVOLUTION DEMO
# ─────────────────────────────────────────────

def convolve_demo(x, h):
    """Discrete convolution y[n] = x[n] * h[n]."""
    return np.convolve(x, h, mode='full')


# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_signal_spectrum(t, x, fs, title="Signal & Spectrum"):
    freqs, mag = compute_fft(x, fs)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].plot(t, x, color='steelblue', lw=1.2)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Time Domain")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freqs, mag, color='darkorange', lw=1.2)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("|X(f)|")
    axes[1].set_title("Frequency Domain (FFT)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spectrogram(signal, fs, title="Spectrogram"):
    times, freqs, S = spectrogram(signal, fs)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.pcolormesh(times, freqs, S, shading='auto', cmap='inferno')
    plt.colorbar(im, ax=ax, label='dB')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_convolution(x, h, title="Convolution Demo"):
    y = convolve_demo(x, h)
    n_x = np.arange(len(x))
    n_h = np.arange(len(h))
    n_y = np.arange(len(y))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].stem(n_x, x, basefmt='k-', linefmt='C0-', markerfmt='C0o')
    axes[0].set_title("x[n] — Input Signal")
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(n_h, h, basefmt='k-', linefmt='C1-', markerfmt='C1o')
    axes[1].set_title("h[n] — Impulse Response")
    axes[1].grid(True, alpha=0.3)

    axes[2].stem(n_y, y, basefmt='k-', linefmt='C2-', markerfmt='C2o')
    axes[2].set_title("y[n] = x[n] * h[n] — Convolution Output")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# DEMO — run directly to see all plots
# ─────────────────────────────────────────────

if __name__ == "__main__":
    fs = 2000

    print("=" * 55)
    print("  DSP PROJECT — Signals & Systems Demo")
    print("=" * 55)

    # --- 1. Multi-tone signal + spectrum ---
    t1, s1 = sine_wave(50,  1.0, duration=0.1, fs=fs)
    t1, s2 = sine_wave(120, 0.5, duration=0.1, fs=fs)
    t1, s3 = sine_wave(300, 0.3, duration=0.1, fs=fs)
    composite = s1 + s2 + s3
    noisy, noise = awgn(composite, snr_db=15)

    fig1 = plot_signal_spectrum(t1, composite, fs, "Multi-tone Signal (50+120+300 Hz)")
    fig1.savefig("signals_clean.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: signals_clean.png")

    fig2 = plot_signal_spectrum(t1, noisy, fs, "Noisy Signal (SNR=15dB)")
    fig2.savefig("signals_noisy.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: signals_noisy.png")

    # --- 2. Chirp spectrogram ---
    t_c, chirp_sig = chirp(f_start=20, f_end=800, duration=1.0, fs=fs)
    fig3 = plot_spectrogram(chirp_sig, fs, "Chirp Signal Spectrogram (20→800 Hz)")
    fig3.savefig("chirp_spectrogram.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: chirp_spectrogram.png")

    # --- 3. Convolution demo ---
    _, x_imp = unit_impulse(N=20, delay=2)
    h_box = np.ones(5) / 5          # Moving average (box filter)
    fig4 = plot_convolution(x_imp, h_box, "Convolution: Impulse * Box Filter")
    fig4.savefig("convolution_demo.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: convolution_demo.png")

    # --- 4. Waveform comparison ---
    t_sq, sq = square_wave(5, duration=0.5, fs=fs)
    t_sa, sa = sawtooth_wave(5, duration=0.5, fs=fs)
    fig5, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig5.suptitle("Waveform Zoo", fontsize=14, fontweight='bold')
    _, sine = sine_wave(5, duration=0.5, fs=fs)
    axes[0].plot(t_sq, sine, color='steelblue'); axes[0].set_title("Sine"); axes[0].grid(alpha=0.3)
    axes[1].plot(t_sq, sq,   color='tomato');    axes[1].set_title("Square (Fourier approx)"); axes[1].grid(alpha=0.3)
    axes[2].plot(t_sa, sa,   color='mediumseagreen'); axes[2].set_title("Sawtooth"); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    fig5.savefig("waveform_zoo.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: waveform_zoo.png")

    print("\n✅ Signals module demo complete — 4 figures saved.")
    plt.close('all')
