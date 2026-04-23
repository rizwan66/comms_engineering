"""
Example 5 — End-to-end text transmission.

Pipeline:
    text -> bits -> Hamming(7,4) -> QPSK -> RRC pulse -> AWGN -> matched filter
    -> sampler -> QPSK demod -> Hamming decode -> text

Sweeps SNR to show where the link "breaks".
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from src.digital import (qpsk_modulate, qpsk_demodulate,
                         pulse_shape, matched_filter, rrc_filter)
from src.channel import awgn
from src.coding import hamming74_encode, hamming74_decode, bit_error_rate
from src.link import text_to_bits, bits_to_text


def run_link(message: str, snr_db: float, use_coding: bool = True,
             seed: int = 0) -> dict:
    """Simulate one transmission and return diagnostics."""
    np.random.seed(seed)

    bits = text_to_bits(message)
    original_bits = bits.copy()

    if use_coding:
        tx_bits = hamming74_encode(bits)   # always multiple of 7
    else:
        tx_bits = bits.copy()
    # Pad to even length for QPSK
    qpsk_pad = len(tx_bits) % 2
    if qpsk_pad:
        tx_bits = np.concatenate([tx_bits, [0]])

    symbols = qpsk_modulate(tx_bits)
    shaped, h = pulse_shape(symbols, beta=0.35, span=10, sps=8)

    # Channel
    rx_signal = awgn(shaped, snr_db)

    # Receiver
    rx_symbols = matched_filter(rx_signal, h, sps=8, span=10)
    # Trim the filter-edge garbage (our matched_filter keeps same length as symbol stream)
    # Keep the first len(symbols) samples
    rx_symbols_clean = rx_symbols[:len(symbols)]
    rx_bits_all = qpsk_demodulate(rx_symbols_clean)
    # Drop QPSK padding
    if qpsk_pad:
        rx_bits_all = rx_bits_all[:-qpsk_pad]

    if use_coding:
        # rx_bits_all is a multiple of 7 by construction
        decoded_bits, n_corr = hamming74_decode(rx_bits_all)
        decoded_bits = decoded_bits[:len(original_bits)]
    else:
        decoded_bits = rx_bits_all[:len(original_bits)]
        n_corr = 0

    ber = bit_error_rate(original_bits, decoded_bits)
    recovered = bits_to_text(decoded_bits)

    return {
        "snr_db": snr_db,
        "ber": ber,
        "n_corrections": n_corr,
        "recovered": recovered,
        "rx_symbols": rx_symbols_clean,
    }


def main():
    message = ("Hello from Munich! Communications engineering is fun. "
               "Let us see how QPSK + Hamming(7,4) handles noise.")
    print(f"Original message ({len(message)} chars):\n  {message!r}\n")

    snrs_db = [-6, -4, -2, 0, 2, 4, 6]
    print(f"{'SNR (dB)':>8} | {'Coded BER':>10} | {'Uncoded BER':>12} | "
          f"{'Corrections':>11}")
    print("-" * 85)
    results_coded = []
    results_uncoded = []
    for snr in snrs_db:
        r_c = run_link(message, snr, use_coding=True, seed=1)
        r_u = run_link(message, snr, use_coding=False, seed=1)
        results_coded.append(r_c)
        results_uncoded.append(r_u)
        print(f"{snr:>8d} | {r_c['ber']:>10.1e} | {r_u['ber']:>12.1e} | "
              f"{r_c['n_corrections']:>11d}")

    print("\n--- Sample of recovered text at each SNR (coded) ---\n")
    for r in results_coded:
        preview = r["recovered"][:70].replace("\n", " ")
        # Replace non-printable with ?
        safe = "".join(c if 32 <= ord(c) < 127 else "?" for c in preview)
        print(f"SNR={r['snr_db']:>3d} dB: {safe}")

    # Constellation plot for 4 SNR values
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, r in zip(axes, [results_coded[0], results_coded[2],
                             results_coded[4], results_coded[6]]):
        ax.scatter(r["rx_symbols"].real, r["rx_symbols"].imag,
                   s=6, c="#2563eb", alpha=0.5)
        for pt in [(-1-1j)/np.sqrt(2), (-1+1j)/np.sqrt(2),
                   (1-1j)/np.sqrt(2), (1+1j)/np.sqrt(2)]:
            ax.plot(pt.real, pt.imag, "x", color="black", ms=12, mew=2)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(f"SNR = {r['snr_db']} dB\nBER = {r['ber']:.1e}", fontsize=10)
    fig.suptitle("End-to-end link: received constellations at increasing SNR",
                 fontsize=12)
    plt.tight_layout()
    out = Path(__file__).parent.parent / "outputs" / "ex5_link_constellations.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Coded-vs-uncoded BER
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(snrs_db, [max(r["ber"], 1e-6) for r in results_uncoded],
                "s-", color="#dc2626", ms=8, label="Uncoded QPSK")
    ax.semilogy(snrs_db, [max(r["ber"], 1e-6) for r in results_coded],
                "o-", color="#059669", ms=8, label="Hamming(7,4) + QPSK")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title("Coding gain: Hamming(7,4) vs. uncoded on the same pipeline")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out2 = Path(__file__).parent.parent / "outputs" / "ex5_coding_gain.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
