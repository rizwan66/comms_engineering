"""
Example 1 — Analog modulation: AM, DSB-SC, SSB, FM.

Shows the time-domain waveforms, the spectra, and the recovered message
after demodulation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from src.analog import (am_modulate, am_demodulate_envelope,
                        dsb_sc_modulate, dsb_sc_demodulate,
                        ssb_modulate, ssb_demodulate,
                        fm_modulate, fm_demodulate)


def plot_spectrum(ax, signal, fs, title, color="#2563eb", xlim=None):
    N = len(signal)
    X = np.fft.fftshift(np.fft.fft(signal)) / N
    f = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))
    ax.plot(f / 1000, 20 * np.log10(np.abs(X) + 1e-12), color=color, lw=1)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-80, 0)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)


def main():
    # Setup
    fs = 100_000
    fc = 10_000
    duration = 0.05
    t = np.arange(0, duration, 1 / fs)
    # Two-tone message (speech-like)
    message = (0.8 * np.sin(2 * np.pi * 500 * t)
               + 0.3 * np.sin(2 * np.pi * 1200 * t))
    message /= np.max(np.abs(message))

    # Modulate
    s_am = am_modulate(message, fc, fs, modulation_index=0.7)
    s_dsb = dsb_sc_modulate(message, fc, fs)
    s_ssb = ssb_modulate(message, fc, fs, sideband="upper")
    s_fm = fm_modulate(message, fc, fs, kf=2000)

    # Demodulate
    m_am = am_demodulate_envelope(s_am, fs, cutoff=2000)
    m_dsb = dsb_sc_demodulate(s_dsb, fc, fs, cutoff=2000)
    m_ssb = ssb_demodulate(s_ssb, fc, fs, cutoff=2000)
    m_fm = fm_demodulate(s_fm, fs, kf=2000, cutoff=2000)

    for m in (m_am, m_dsb, m_ssb, m_fm):
        m *= 1.0 / max(np.max(np.abs(m)), 1e-9)

    # --- Time-domain figure ---
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    show_until = int(0.02 * fs)
    tt = t[:show_until] * 1000  # ms

    axes[0].plot(tt, message[:show_until], color="black", lw=1.2)
    axes[0].set_title("Message m(t)", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("Amplitude")

    for ax, sig, name, col in [
        (axes[1], s_am, "AM (with carrier, mu=0.7)", "#2563eb"),
        (axes[2], s_dsb, "DSB-SC", "#dc2626"),
        (axes[3], s_ssb, "SSB (upper)", "#059669"),
        (axes[4], s_fm, "FM (kf=2000 Hz/V)", "#7c3aed"),
    ]:
        ax.plot(tt, sig[:show_until], color=col, lw=0.8)
        # Envelope overlay for AM
        if "AM" in name and "SC" not in name:
            env = 1 + 0.7 * message[:show_until]
            ax.plot(tt, env, color="black", lw=1.2, ls="--", alpha=0.6,
                    label="Envelope = 1 + mu*m(t)")
            ax.plot(tt, -env, color="black", lw=1.2, ls="--", alpha=0.6)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    out = Path(__file__).parent.parent / "outputs" / "ex1_analog_waveforms.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")

    # --- Spectra figure ---
    fig2, axes = plt.subplots(2, 2, figsize=(13, 8))
    xlim = (-20, 20)
    plot_spectrum(axes[0, 0], s_am, fs, "AM (discrete carrier + two sidebands)",
                  color="#2563eb", xlim=xlim)
    plot_spectrum(axes[0, 1], s_dsb, fs, "DSB-SC (no carrier, two sidebands)",
                  color="#dc2626", xlim=xlim)
    plot_spectrum(axes[1, 0], s_ssb, fs, "SSB upper (only upper sideband)",
                  color="#059669", xlim=xlim)
    plot_spectrum(axes[1, 1], s_fm, fs, "FM (Bessel sidebands)",
                  color="#7c3aed", xlim=xlim)
    plt.tight_layout()
    out2 = Path(__file__).parent.parent / "outputs" / "ex1_analog_spectra.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    print(f"Saved: {out2}")

    # --- Recovered messages ---
    fig3, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, m_hat, name, col in [
        (axes[0], m_am, "AM recovered", "#2563eb"),
        (axes[1], m_dsb, "DSB-SC recovered", "#dc2626"),
        (axes[2], m_ssb, "SSB recovered", "#059669"),
        (axes[3], m_fm, "FM recovered", "#7c3aed"),
    ]:
        ax.plot(t[:show_until] * 1000, message[:show_until],
                color="black", lw=1.2, label="original", alpha=0.5)
        ax.plot(t[:show_until] * 1000, m_hat[:show_until], color=col, lw=1.2,
                label=name)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (ms)")
    fig3.suptitle("Demodulated messages vs. original", fontsize=12)
    plt.tight_layout()
    out3 = Path(__file__).parent.parent / "outputs" / "ex1_analog_recovered.png"
    plt.savefig(out3, dpi=120, bbox_inches="tight")
    print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
