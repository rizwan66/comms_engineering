"""
Digital Modulation
==================

Mappers and demappers for:

    BPSK     : 1 bit per symbol, constellation {-1, +1}
    QPSK     : 2 bits per symbol, constellation {+/-1 +/- j} / sqrt(2)
    M-QAM    : log2(M) bits per symbol, square Gray-coded constellation
               (M must be a power of 4: 4, 16, 64, 256)

The pipeline:
    bits -> symbols -> upsample -> pulse shape (RRC) -> channel -> matched filter
    -> sampler -> decisions -> bits

All baseband signals are complex numpy arrays. Passband conversion is handled
separately in channel.py.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Bit <-> integer helpers
# ---------------------------------------------------------------------------

def bits_to_ints(bits: np.ndarray, k: int) -> np.ndarray:
    """Pack bits in groups of k (MSB first) into integers."""
    assert len(bits) % k == 0, f"Bit length {len(bits)} not divisible by {k}"
    reshaped = bits.reshape(-1, k)
    weights = 2 ** np.arange(k - 1, -1, -1)
    return reshaped @ weights


def ints_to_bits(ints: np.ndarray, k: int) -> np.ndarray:
    """Unpack integers back into k-bit groups (MSB first)."""
    out = np.zeros((len(ints), k), dtype=np.int8)
    for i in range(k):
        out[:, k - 1 - i] = (ints >> i) & 1
    return out.flatten()


# ---------------------------------------------------------------------------
# BPSK
# ---------------------------------------------------------------------------

def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """0 -> -1, 1 -> +1. Returns complex symbols (imag = 0)."""
    return (2 * bits.astype(float) - 1) + 0j


def bpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    return (symbols.real >= 0).astype(np.int8)


# ---------------------------------------------------------------------------
# QPSK (Gray-coded)
# ---------------------------------------------------------------------------

# Gray mapping: bits b1 b0 -> (I, Q)
#   00 -> (+1, +1)
#   01 -> (+1, -1)
#   11 -> (-1, -1)
#   10 -> (-1, +1)
_QPSK_MAP = {
    0b00: (+1, +1),
    0b01: (+1, -1),
    0b11: (-1, -1),
    0b10: (-1, +1),
}


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    ints = bits_to_ints(bits, 2)
    iq = np.array([_QPSK_MAP[i] for i in ints], dtype=float)
    return (iq[:, 0] + 1j * iq[:, 1]) / np.sqrt(2)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(2 * len(symbols), dtype=np.int8)
    bits[0::2] = (symbols.real < 0).astype(np.int8)     # b1
    bits[1::2] = (symbols.imag < 0).astype(np.int8)     # b0
    return bits


# ---------------------------------------------------------------------------
# M-QAM (square, Gray-coded) — supports M = 4, 16, 64, 256
# ---------------------------------------------------------------------------

def _gray_code(n: int) -> np.ndarray:
    """Return Gray-coded sequence of length 2**n."""
    return np.array([i ^ (i >> 1) for i in range(2 ** n)])


def _qam_levels(M: int) -> tuple[np.ndarray, float]:
    """
    Return PAM levels and normalization factor for one axis of M-QAM.
    For M-QAM, each axis carries sqrt(M) levels: {-(L-1), ..., -1, 1, ..., L-1}.
    """
    L = int(np.sqrt(M))
    assert L * L == M, f"M = {M} is not a perfect square"
    levels = np.arange(-(L - 1), L, 2, dtype=float)    # e.g. [-3, -1, 1, 3] for L=4
    # Average symbol energy for uniform distribution over LxL grid:
    # E_s = (2/3)*(M-1)  (standard result)
    norm = np.sqrt((2 / 3) * (M - 1))
    return levels, norm


def qam_modulate(bits: np.ndarray, M: int) -> np.ndarray:
    k = int(np.log2(M))
    assert 2 ** k == M
    assert len(bits) % k == 0
    half = k // 2

    levels, norm = _qam_levels(M)
    gray = _gray_code(half)
    gray_to_level = {g: levels[i] for i, g in enumerate(gray)}

    ints = bits_to_ints(bits, k)
    i_bits = ints >> half                  # top half -> I axis
    q_bits = ints & ((1 << half) - 1)      # bottom half -> Q axis
    I = np.array([gray_to_level[b] for b in i_bits])
    Q = np.array([gray_to_level[b] for b in q_bits])
    return (I + 1j * Q) / norm


def qam_demodulate(symbols: np.ndarray, M: int) -> np.ndarray:
    k = int(np.log2(M))
    half = k // 2
    L = int(np.sqrt(M))
    levels, norm = _qam_levels(M)
    gray = _gray_code(half)
    level_to_gray = {levels[i]: gray[i] for i in range(L)}

    # Nearest-neighbor decision on each axis
    scaled = symbols * norm
    def decide(x):
        idx = np.clip(np.round((x + (L - 1)) / 2).astype(int), 0, L - 1)
        return levels[idx]

    I_hat = decide(scaled.real)
    Q_hat = decide(scaled.imag)

    i_bits = np.array([level_to_gray[x] for x in I_hat])
    q_bits = np.array([level_to_gray[x] for x in Q_hat])
    ints = (i_bits << half) | q_bits
    return ints_to_bits(ints, k)


def constellation_points(M: int, scheme: str = "qam") -> np.ndarray:
    """Return ideal constellation points for plotting."""
    if scheme.lower() == "bpsk":
        return np.array([-1+0j, 1+0j])
    if scheme.lower() == "qpsk":
        return np.array([(+1+1j), (+1-1j), (-1+1j), (-1-1j)]) / np.sqrt(2)
    # QAM
    levels, norm = _qam_levels(M)
    I, Q = np.meshgrid(levels, levels)
    return ((I + 1j * Q) / norm).flatten()


# ---------------------------------------------------------------------------
# Pulse shaping: Root-Raised-Cosine
# ---------------------------------------------------------------------------

def rrc_filter(beta: float, span: int, sps: int) -> np.ndarray:
    """
    Root-Raised-Cosine filter.
      beta  : roll-off factor (0..1)
      span  : filter length in symbols
      sps   : samples per symbol
    Unit-energy normalization.
    """
    N = span * sps
    t = (np.arange(N + 1) - N / 2) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = (np.sin(np.pi * ti * (1 - beta))
                   + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    """Insert sps-1 zeros between symbols."""
    out = np.zeros(len(symbols) * sps, dtype=complex)
    out[::sps] = symbols
    return out


def pulse_shape(symbols: np.ndarray, beta: float = 0.35,
                span: int = 10, sps: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Upsample and apply RRC filter. Returns (shaped_signal, filter_taps).
    """
    h = rrc_filter(beta, span, sps)
    upsampled = upsample(symbols, sps)
    shaped = np.convolve(upsampled, h, mode="same")
    return shaped, h


def matched_filter(rx: np.ndarray, h: np.ndarray, sps: int,
                   span: int = 10) -> np.ndarray:
    """
    Convolve with RRC (matched filter) and sample at symbol centers.
    """
    filtered = np.convolve(rx, h, mode="same")
    # The convolution delay of a "same" convolution with a symmetric filter is 0,
    # so we can sample at sps*k offsets.
    return filtered[::sps]
