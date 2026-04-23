"""
Example 4 — Multipath and fading channels.

Shows three effects:
    1. Multipath distortion (frequency-selective fading) on a QPSK signal
    2. Rayleigh flat-fading BER curve (plus AWGN for reference)
    3. Rician fading with varying K-factor
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from src.digital import qpsk_modulate, qpsk_demodulate, constellation_points
from src.channel import awgn, multipath_channel, rayleigh_fading, rician_fading
from src.coding import bit_error_rate, ber_bpsk_theory


def simulate_rayleigh_ber(ebn0_db, n_bits=200_000, seed=0):
    """Monte Carlo BER for QPSK over flat Rayleigh fading with AWGN."""
    rng = np.random.default_rng(seed)
    ber = np.zeros_like(ebn0_db, dtype=float)
    for i, e in enumerate(ebn0_db):
        bits = rng.integers(0, 2, n_bits).astype(np.int8)
        sym = qpsk_modulate(bits)
        # Per-symbol independent Rayleigh fades (fast fading)
        h = (rng.normal(size=len(sym)) + 1j * rng.normal(size=len(sym))) / np.sqrt(2)
        faded = sym * h
        snr_db = e + 10 * np.log10(2)          # QPSK: Es/N0 = 2 * Eb/N0
        rx = awgn(faded, snr_db)
        # Coherent demod assumes perfect channel knowledge -> divide by h
        equalized = rx / h
        decoded = qpsk_demodulate(equalized)
        ber[i] = bit_error_rate(bits, decoded)
    return ber


def ber_rayleigh_theory(ebn0_db):
    """Exact BER for BPSK/QPSK over flat Rayleigh fading."""
    ebn0 = 10 ** (ebn0_db / 10)
    return 0.5 * (1 - np.sqrt(ebn0 / (1 + ebn0)))


def main():
    np.random.seed(42)

    # --- (1) Multipath distortion on a QPSK constellation ---
    n_sym = 2000
    bits = np.random.randint(0, 2, 2 * n_sym).astype(np.int8)
    sym = qpsk_modulate(bits)
    # Channel with LOS + echo at symbol-rate delays (inter-symbol interference)
    taps = [(0, 1.0 + 0j), (1, 0.6 * np.exp(1j * 0.7)), (3, 0.3 * np.exp(1j * 1.9))]
    rx_mp = multipath_channel(sym, taps)
    rx_mp = awgn(rx_mp, 20)  # light noise on top

    # --- (2) Rayleigh fading constellation ---
    rx_rayleigh = rayleigh_fading(sym, coherence_samples=50)
    rx_rayleigh = awgn(rx_rayleigh, 20)

    # --- (3) Rician fading (K = 6 dB, LOS dominant) ---
    rx_rician = rician_fading(sym, K_db=6, coherence_samples=50)
    rx_rician = awgn(rx_rician, 20)

    # --- Figure: four constellations ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    plots = [
        (axes[0, 0], awgn(sym, 20), "AWGN only (SNR = 20 dB)", "#2563eb"),
        (axes[0, 1], rx_mp, "Multipath + AWGN (ISI visible)", "#dc2626"),
        (axes[1, 0], rx_rayleigh, "Rayleigh fading + AWGN", "#059669"),
        (axes[1, 1], rx_rician, "Rician (K = 6 dB) + AWGN", "#7c3aed"),
    ]
    for ax, data, name, col in plots:
        ax.scatter(data.real, data.imag, s=4, c=col, alpha=0.4)
        for pt in constellation_points(4, "qpsk"):
            ax.plot(pt.real, pt.imag, "x", color="black", ms=12, mew=2)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
    fig.suptitle("QPSK constellation under different channel conditions", fontsize=13)
    plt.tight_layout()
    out = Path(__file__).parent.parent / "outputs" / "ex4_channel_constellations.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    # --- Figure: Rayleigh BER vs AWGN BER ---
    ebn0 = np.arange(0, 31, 2)
    print("\nSimulating Rayleigh BER (slow — per-symbol fades)...")
    ber_rayl_sim = simulate_rayleigh_ber(ebn0)
    ebn0_fine = np.linspace(0, 30, 200)

    fig2, ax2 = plt.subplots(figsize=(10, 6.5))
    ax2.semilogy(ebn0_fine, ber_bpsk_theory(ebn0_fine), "--",
                 color="#2563eb", label="AWGN (theory)")
    ax2.semilogy(ebn0_fine, ber_rayleigh_theory(ebn0_fine), "--",
                 color="#dc2626", label="Rayleigh (theory)")
    ax2.semilogy(ebn0, np.maximum(ber_rayl_sim, 1e-7), "o",
                 color="#dc2626", ms=7, label="Rayleigh (sim)")
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("Bit Error Rate")
    ax2.set_title("AWGN vs. flat Rayleigh fading — the cost of multipath")
    ax2.set_ylim(1e-5, 1)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="lower left", fontsize=10)

    # Annotate: at BER=1e-3, Rayleigh needs ~20dB more
    ax2.axhline(1e-3, color="gray", ls=":", lw=1)
    ax2.text(28, 1.3e-3, "BER = 1e-3", fontsize=9, ha="right", color="gray")

    plt.tight_layout()
    out2 = Path(__file__).parent.parent / "outputs" / "ex4_rayleigh_ber.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
