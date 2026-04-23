"""
Channel Coding & Link Metrics
=============================

    Hamming(7,4) : single-error-correcting, double-error-detecting
    Repetition   : simple baseline for comparison
    BER / SER    : measurement utilities
    Theoretical BER curves : for sanity checks
"""
from __future__ import annotations
import numpy as np
from scipy.special import erfc


# ---------------------------------------------------------------------------
# Hamming(7,4) codec
# ---------------------------------------------------------------------------

# Systematic generator: 4 data bits -> 7 code bits (p1 p2 d1 p3 d2 d3 d4)
_G = np.array([
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.int8)

_H = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.int8)


def hamming74_encode(bits: np.ndarray) -> np.ndarray:
    """Encode a stream of bits. Pads with zeros if len not divisible by 4."""
    pad = (-len(bits)) % 4
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.int8)])
    blocks = bits.reshape(-1, 4)
    encoded = (blocks @ _G.T) % 2
    return encoded.flatten().astype(np.int8)


def hamming74_decode(bits: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Decode. Returns (data_bits, n_corrections).
    Corrects one bit error per 7-bit block.
    """
    blocks = bits.reshape(-1, 7)
    syndrome = (blocks @ _H.T) % 2
    # Convert syndrome bits into error position (1..7, 0 = no error)
    error_pos = syndrome[:, 0] * 1 + syndrome[:, 1] * 2 + syndrome[:, 2] * 4
    corrections = 0
    for i, pos in enumerate(error_pos):
        if pos != 0:
            blocks[i, pos - 1] ^= 1
            corrections += 1
    # Data bits are at positions 3, 5, 6, 7 (1-indexed) -> indices 2, 4, 5, 6
    data = blocks[:, [2, 4, 5, 6]]
    return data.flatten().astype(np.int8), corrections


# ---------------------------------------------------------------------------
# Repetition code (for comparison)
# ---------------------------------------------------------------------------

def repetition_encode(bits: np.ndarray, n: int = 3) -> np.ndarray:
    return np.repeat(bits, n)


def repetition_decode(bits: np.ndarray, n: int = 3) -> np.ndarray:
    """Majority vote."""
    blocks = bits.reshape(-1, n)
    return (np.sum(blocks, axis=1) > n // 2).astype(np.int8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def bit_error_rate(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    n = min(len(tx_bits), len(rx_bits))
    return float(np.mean(tx_bits[:n] != rx_bits[:n]))


def symbol_error_rate(tx_symbols: np.ndarray, rx_symbols: np.ndarray,
                      tol: float = 1e-6) -> float:
    n = min(len(tx_symbols), len(rx_symbols))
    return float(np.mean(np.abs(tx_symbols[:n] - rx_symbols[:n]) > tol))


# ---------------------------------------------------------------------------
# Theoretical BER curves (AWGN, coherent detection)
# ---------------------------------------------------------------------------

def qfunc(x):
    """Q(x) = 0.5 * erfc(x / sqrt(2))"""
    return 0.5 * erfc(x / np.sqrt(2))


def ber_bpsk_theory(ebn0_db: np.ndarray) -> np.ndarray:
    """BPSK & QPSK share the same BER vs Eb/N0 curve."""
    ebn0 = 10 ** (ebn0_db / 10)
    return qfunc(np.sqrt(2 * ebn0))


def ber_qpsk_theory(ebn0_db: np.ndarray) -> np.ndarray:
    return ber_bpsk_theory(ebn0_db)


def ber_qam_theory(ebn0_db: np.ndarray, M: int) -> np.ndarray:
    """Approximate BER for square M-QAM with Gray coding."""
    k = np.log2(M)
    ebn0 = 10 ** (ebn0_db / 10)
    return (4 / k) * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt(3 * k * ebn0 / (M - 1)))
