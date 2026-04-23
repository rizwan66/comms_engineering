"""
Example 3 — Animated constellation diagram.

QPSK and 16-QAM constellations as SNR is swept from 0 dB to 25 dB.
Saved as an MP4 (or GIF fallback).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from src.digital import qpsk_modulate, qam_modulate, constellation_points
from src.channel import awgn


def main():
    np.random.seed(7)
    n_symbols = 800

    # Prepare QPSK symbols
    qpsk_bits = np.random.randint(0, 2, 2 * n_symbols).astype(np.int8)
    qpsk_sym = qpsk_modulate(qpsk_bits)

    # Prepare 16-QAM symbols
    qam_bits = np.random.randint(0, 2, 4 * n_symbols).astype(np.int8)
    qam_sym = qam_modulate(qam_bits, 16)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax in axes:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")

    # Ideal constellation markers
    for pt in constellation_points(4, "qpsk"):
        axes[0].plot(pt.real, pt.imag, "x", color="black", ms=14, mew=2)
    for pt in constellation_points(16, "qam"):
        axes[1].plot(pt.real, pt.imag, "x", color="black", ms=10, mew=2)

    scat_qpsk = axes[0].scatter([], [], s=8, c="#2563eb", alpha=0.6)
    scat_qam = axes[1].scatter([], [], s=8, c="#059669", alpha=0.6)

    title_qpsk = axes[0].set_title("QPSK — SNR = 0.0 dB")
    title_qam = axes[1].set_title("16-QAM — SNR = 0.0 dB")

    fig.suptitle("Constellation under AWGN — watch the clusters tighten as SNR rises",
                 fontsize=12)

    snr_values = np.concatenate([
        np.linspace(0, 25, 35),
        np.linspace(25, 0, 35),     # sweep back down for a loop
    ])

    def update(frame):
        snr = snr_values[frame]
        rx_qpsk = awgn(qpsk_sym, snr)
        rx_qam = awgn(qam_sym, snr)
        scat_qpsk.set_offsets(np.column_stack([rx_qpsk.real, rx_qpsk.imag]))
        scat_qam.set_offsets(np.column_stack([rx_qam.real, rx_qam.imag]))
        title_qpsk.set_text(f"QPSK — SNR = {snr:.1f} dB")
        title_qam.set_text(f"16-QAM — SNR = {snr:.1f} dB")
        return scat_qpsk, scat_qam, title_qpsk, title_qam

    anim = FuncAnimation(fig, update, frames=len(snr_values), interval=100, blit=False)
    out = Path(__file__).parent.parent / "animations" / "constellation_snr_sweep.gif"
    anim.save(out, writer=PillowWriter(fps=10), dpi=80)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
