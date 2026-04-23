"""
Example 2 — BER vs. Eb/N0 for BPSK, QPSK, 16-QAM, 64-QAM over AWGN.

Simulated curves compared against the theoretical expressions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from src.digital import (bpsk_modulate, bpsk_demodulate,
                         qpsk_modulate, qpsk_demodulate,
                         qam_modulate, qam_demodulate)
from src.channel import awgn
from src.coding import (ber_bpsk_theory, ber_qpsk_theory, ber_qam_theory,
                        bit_error_rate)


def simulate_ber(scheme: str, ebn0_db: np.ndarray,
                 n_bits: int = 200_000, M: int = 16,
                 seed: int = 0) -> np.ndarray:
    """Monte-Carlo BER simulation with AWGN at each Eb/N0."""
    rng = np.random.default_rng(seed)
    ber = np.zeros_like(ebn0_db, dtype=float)

    for i, e in enumerate(ebn0_db):
        bits = rng.integers(0, 2, n_bits).astype(np.int8)

        if scheme == "bpsk":
            symbols = bpsk_modulate(bits)
            k = 1
        elif scheme == "qpsk":
            symbols = qpsk_modulate(bits)
            k = 2
        elif scheme == "qam":
            k = int(np.log2(M))
            bits = bits[:(len(bits) // k) * k]
            symbols = qam_modulate(bits, M)
        else:
            raise ValueError(f"Unknown scheme {scheme}")

        # Eb/N0 -> SNR per symbol
        # Symbol energy is 1 (by our normalization), so Es/N0 = k * Eb/N0.
        snr_db = e + 10 * np.log10(k)
        rx = awgn(symbols, snr_db)

        if scheme == "bpsk":
            decoded = bpsk_demodulate(rx)
        elif scheme == "qpsk":
            decoded = qpsk_demodulate(rx)
        else:
            decoded = qam_demodulate(rx, M)

        ber[i] = bit_error_rate(bits, decoded)

    return ber


def main():
    ebn0_db = np.arange(0, 15, 1)

    print("Simulating BER curves...")
    ber_bpsk = simulate_ber("bpsk", ebn0_db, n_bits=200_000)
    print("  BPSK done")
    ber_qpsk = simulate_ber("qpsk", ebn0_db, n_bits=200_000)
    print("  QPSK done")
    ber_16qam = simulate_ber("qam", ebn0_db, n_bits=400_000, M=16)
    print("  16-QAM done")
    ber_64qam = simulate_ber("qam", ebn0_db, n_bits=600_000, M=64)
    print("  64-QAM done")

    fig, ax = plt.subplots(figsize=(11, 7))
    ebn0_fine = np.linspace(0, 15, 200)

    # Theory
    ax.semilogy(ebn0_fine, ber_bpsk_theory(ebn0_fine), "--",
                color="#2563eb", lw=1, label="BPSK (theory)")
    ax.semilogy(ebn0_fine, ber_qam_theory(ebn0_fine, 16), "--",
                color="#059669", lw=1, label="16-QAM (theory)")
    ax.semilogy(ebn0_fine, ber_qam_theory(ebn0_fine, 64), "--",
                color="#7c3aed", lw=1, label="64-QAM (theory)")

    # Simulation
    ax.semilogy(ebn0_db, np.maximum(ber_bpsk, 1e-7), "o",
                color="#2563eb", ms=7, label="BPSK (sim)")
    ax.semilogy(ebn0_db, np.maximum(ber_qpsk, 1e-7), "s",
                color="#dc2626", ms=6, label="QPSK (sim)")
    ax.semilogy(ebn0_db, np.maximum(ber_16qam, 1e-7), "^",
                color="#059669", ms=7, label="16-QAM (sim)")
    ax.semilogy(ebn0_db, np.maximum(ber_64qam, 1e-7), "D",
                color="#7c3aed", ms=6, label="64-QAM (sim)")

    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title("BER performance over AWGN: simulation vs. theory")
    ax.set_ylim(1e-6, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    out = Path(__file__).parent.parent / "outputs" / "ex2_ber_curves.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Summary table
    print(f"\n{'Eb/N0 (dB)':>10} {'BPSK':>10} {'QPSK':>10} {'16-QAM':>10} {'64-QAM':>10}")
    print("-" * 55)
    for i, e in enumerate(ebn0_db):
        print(f"{e:>10d} {ber_bpsk[i]:>10.1e} {ber_qpsk[i]:>10.1e} "
              f"{ber_16qam[i]:>10.1e} {ber_64qam[i]:>10.1e}")


if __name__ == "__main__":
    main()
