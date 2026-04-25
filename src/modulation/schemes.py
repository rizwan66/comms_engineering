"""
modulation/schemes.py
=====================
Complete modulation & demodulation implementations.
Covers: AM, DSB-SC, FM, BPSK, QPSK, QAM, OFDM, BER curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp


# ─────────────────────────────────────────────
# 1. ANALOG MODULATION
# ─────────────────────────────────────────────

def am_modulate(message, fc, fs, ka=0.5):
    """AM: s(t) = Ac[1 + ka·m(t)]·cos(2πfct)"""
    t = np.arange(len(message)) / fs
    carrier = np.cos(2 * np.pi * fc * t)
    return (1 + ka * message) * carrier, t


def am_demodulate(s, fc, fs):
    """AM envelope demodulation."""
    t   = np.arange(len(s)) / fs
    env = np.abs(sp.hilbert(s)) - 1      # envelope - DC
    return env


def dsb_sc_modulate(message, fc, fs):
    """DSB-SC: s(t) = m(t)·cos(2πfct)"""
    t = np.arange(len(message)) / fs
    return message * np.cos(2 * np.pi * fc * t), t


def dsb_sc_demodulate(s, fc, fs, lpf_cutoff=500):
    """Coherent demodulation of DSB-SC."""
    t = np.arange(len(s)) / fs
    product = s * np.cos(2 * np.pi * fc * t)
    nyq  = fs / 2
    b, a = sp.butter(6, lpf_cutoff / nyq, btype='low')
    return sp.filtfilt(b, a, product) * 2


def fm_modulate(message, fc, fs, kf=50.0):
    """FM: s(t) = cos[2πfct + 2πkf∫m(τ)dτ]"""
    t   = np.arange(len(message)) / fs
    phi = 2 * np.pi * kf * np.cumsum(message) / fs
    return np.cos(2 * np.pi * fc * t + phi), t


def fm_demodulate(s, fs):
    """FM demodulation using instantaneous frequency (differentiator)."""
    analytic  = sp.hilbert(s)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq  = np.diff(inst_phase) / (2 * np.pi / fs)
    return np.append(inst_freq, inst_freq[-1])


# ─────────────────────────────────────────────
# 2. DIGITAL MODULATION
# ─────────────────────────────────────────────

def bpsk_modulate(bits, fc, fs, bit_rate=100):
    """BPSK: bit=0 → +cos, bit=1 → -cos"""
    samples_per_bit = int(fs / bit_rate)
    t_out = []
    s_out = []
    for i, b in enumerate(bits):
        t_bit = np.arange(samples_per_bit) / fs + i * samples_per_bit / fs
        phase = 0 if b == 0 else np.pi
        s_out.append(np.cos(2 * np.pi * fc * t_bit + phase))
        t_out.append(t_bit)
    t = np.concatenate(t_out)
    s = np.concatenate(s_out)
    return s, t


def bpsk_demodulate(s, fc, fs, bit_rate=100):
    """Coherent BPSK demodulation."""
    samples_per_bit = int(fs / bit_rate)
    t = np.arange(len(s)) / fs
    product = s * np.cos(2 * np.pi * fc * t)   # multiply by reference
    bits = []
    for i in range(len(s) // samples_per_bit):
        chunk = product[i*samples_per_bit:(i+1)*samples_per_bit]
        bits.append(0 if np.mean(chunk) > 0 else 1)
    return np.array(bits)


def qpsk_modulate(bits, fc, fs, bit_rate=100):
    """
    QPSK: pairs of bits mapped to 4 phases (45, 135, 225, 315 degrees).
    00→45°, 01→135°, 10→315°, 11→225°
    """
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)

    symbol_map = {(0,0): np.pi/4, (0,1): 3*np.pi/4,
                  (1,0): -np.pi/4, (1,1): -3*np.pi/4}

    samples_per_sym = int(fs / bit_rate)
    s_out, t_out = [], []

    for i in range(0, len(bits), 2):
        pair  = (bits[i], bits[i+1])
        phase = symbol_map[pair]
        idx   = i // 2
        t_sym = np.arange(samples_per_sym) / fs + idx * samples_per_sym / fs
        s_out.append(np.cos(2 * np.pi * fc * t_sym + phase))
        t_out.append(t_sym)

    return np.concatenate(s_out), np.concatenate(t_out)


def qam_modulate(bits, M=16, fc=1000, fs=8000, bit_rate=100):
    """
    M-QAM modulation (M must be perfect square: 4, 16, 64, 256).
    Returns modulated signal and symbol array.
    """
    bits_per_sym = int(np.log2(M))
    sqrtM = int(np.sqrt(M))
    if len(bits) % bits_per_sym != 0:
        bits = np.append(bits, np.zeros(bits_per_sym - len(bits) % bits_per_sym, dtype=int))

    # Gray-coded constellation points
    levels = np.arange(-sqrtM+1, sqrtM+1, 2)    # e.g. [-3,-1,1,3] for 16-QAM
    symbols_I, symbols_Q = [], []

    for i in range(0, len(bits), bits_per_sym):
        chunk = bits[i:i+bits_per_sym]
        I_bits = chunk[:bits_per_sym//2]
        Q_bits = chunk[bits_per_sym//2:]
        I_idx  = int(''.join(map(str, I_bits)), 2)
        Q_idx  = int(''.join(map(str, Q_bits)), 2)
        symbols_I.append(levels[I_idx % sqrtM])
        symbols_Q.append(levels[Q_idx % sqrtM])

    symbols_I = np.array(symbols_I, dtype=float)
    symbols_Q = np.array(symbols_Q, dtype=float)

    samples_per_sym = int(fs / bit_rate)
    s_out = []
    for k in range(len(symbols_I)):
        t_sym = np.arange(samples_per_sym) / fs + k * samples_per_sym / fs
        s_out.append(symbols_I[k] * np.cos(2*np.pi*fc*t_sym)
                   - symbols_Q[k] * np.sin(2*np.pi*fc*t_sym))
    return np.concatenate(s_out), symbols_I + 1j * symbols_Q


def plot_qam_constellation(symbols, M, title=None):
    """Plot IQ constellation diagram."""
    title = title or f"{M}-QAM Constellation"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(symbols.real, symbols.imag, alpha=0.6, s=50, color='steelblue')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 3. OFDM
# ─────────────────────────────────────────────

def ofdm_modulate(data_symbols, N_subcarriers=64, cp_len=16):
    """
    OFDM modulation using IFFT.
    data_symbols: complex array, length = N_subcarriers (one OFDM symbol)
    cp_len: cyclic prefix length
    """
    time_domain = np.fft.ifft(data_symbols, n=N_subcarriers)
    cp           = time_domain[-cp_len:]           # cyclic prefix
    return np.concatenate([cp, time_domain])


def ofdm_demodulate(rx, N_subcarriers=64, cp_len=16):
    """OFDM demodulation — remove CP then FFT."""
    rx_no_cp = rx[cp_len:cp_len + N_subcarriers]
    return np.fft.fft(rx_no_cp, n=N_subcarriers)


def ofdm_frame(num_symbols=10, N=64, cp=16, mod_order=4):
    """Generate a full OFDM frame with QPSK symbols on each subcarrier."""
    frames = []
    all_symbols = []
    rng = np.random.default_rng(0)
    for _ in range(num_symbols):
        phase_angles = rng.choice([np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4], size=N)
        qpsk_syms    = np.exp(1j * phase_angles)
        all_symbols.append(qpsk_syms)
        frames.append(ofdm_modulate(qpsk_syms, N, cp))
    return np.concatenate(frames), np.array(all_symbols)


# ─────────────────────────────────────────────
# 4. BER SIMULATION
# ─────────────────────────────────────────────

def ber_bpsk_theory(snr_db_range):
    """Theoretical BER for BPSK: Q(sqrt(2*Eb/N0))"""
    from scipy.special import erfc
    snr_lin = 10 ** (np.array(snr_db_range) / 10)
    return 0.5 * erfc(np.sqrt(snr_lin))


def ber_bpsk_simulation(snr_db_range, num_bits=10000):
    """Monte-Carlo BER simulation for BPSK over AWGN."""
    rng  = np.random.default_rng(42)
    bers = []
    for snr_db in snr_db_range:
        bits   = rng.integers(0, 2, num_bits)
        bpsk   = 2 * bits - 1                             # ±1
        snr    = 10 ** (snr_db / 10)
        noise  = rng.normal(0, 1/np.sqrt(2*snr), num_bits)
        rx     = bpsk + noise
        decoded = (rx < 0).astype(int)
        bers.append(np.sum(bits != decoded) / num_bits + 1e-9)
    return np.array(bers)


def plot_ber(snr_range=range(0, 15)):
    """Plot BER vs Eb/N0 for BPSK."""
    snr = list(snr_range)
    theory = ber_bpsk_theory(snr)
    sim    = ber_bpsk_simulation(snr, num_bits=50000)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(snr, theory, 'b-',  lw=2,   label='BPSK Theory')
    ax.semilogy(snr, sim,    'ro--', lw=1.5, label='BPSK Simulation', markersize=6)
    ax.set_xlabel("Eb/N₀ (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("BER vs Eb/N₀ — BPSK in AWGN")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-5, 1])
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    fs = 8000
    fc = 1000
    bit_rate = 100

    print("=" * 55)
    print("  DSP PROJECT — Modulation Demo")
    print("=" * 55)

    # ── FM modulation ──
    t_msg = np.linspace(0, 0.1, int(fs*0.1))
    message = np.sin(2 * np.pi * 200 * t_msg)
    s_fm, t_fm = fm_modulate(message, fc=fc, fs=fs, kf=50)
    demod_fm   = fm_demodulate(s_fm, fs)

    fig1, axes = plt.subplots(3, 1, figsize=(13, 8))
    fig1.suptitle("FM Modulation & Demodulation", fontsize=14, fontweight='bold')
    axes[0].plot(t_msg, message, color='steelblue');  axes[0].set_title("Message Signal (200 Hz)"); axes[0].grid(alpha=0.3)
    axes[1].plot(t_fm, s_fm,    color='darkorange');  axes[1].set_title("FM Modulated Signal (fc=1kHz)"); axes[1].grid(alpha=0.3)
    axes[2].plot(t_fm, demod_fm/max(abs(demod_fm)), color='mediumseagreen'); axes[2].set_title("Demodulated (normalized)"); axes[2].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel("Time (s)")
    plt.tight_layout()
    fig1.savefig("mod_fm.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: mod_fm.png")

    # ── BPSK ──
    bits = np.array([0,1,0,0,1,1,0,1,1,0])
    s_bpsk, t_bpsk = bpsk_modulate(bits, fc=fc, fs=fs, bit_rate=bit_rate)
    rx_bits = bpsk_demodulate(s_bpsk, fc=fc, fs=fs, bit_rate=bit_rate)

    fig2, axes = plt.subplots(2, 1, figsize=(13, 6))
    fig2.suptitle("BPSK Modulation", fontsize=14, fontweight='bold')
    axes[0].plot(t_bpsk, s_bpsk, color='steelblue', lw=1); axes[0].set_title("BPSK Signal"); axes[0].grid(alpha=0.3)
    axes[1].step(range(len(bits)), bits, where='mid', color='tomato', lw=2)
    axes[1].step(range(len(rx_bits)), rx_bits+0.05, where='mid', color='seagreen', lw=1.5, linestyle='--')
    axes[1].set_title(f"Original bits (red) vs Decoded (green)  — BER={np.mean(bits!=rx_bits):.2f}")
    axes[1].set_yticks([0, 1]); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig2.savefig("mod_bpsk.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: mod_bpsk.png")

    # ── 16-QAM constellation ──
    rng = np.random.default_rng(0)
    bits_qam = rng.integers(0, 2, 400)
    _, qam_symbols = qam_modulate(bits_qam, M=16, fc=fc, fs=fs, bit_rate=bit_rate)
    noise = rng.normal(0, 0.3, len(qam_symbols)) + 1j * rng.normal(0, 0.3, len(qam_symbols))
    fig3 = plot_qam_constellation(qam_symbols + noise, 16, "16-QAM Constellation (with noise)")
    fig3.savefig("mod_qam_constellation.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: mod_qam_constellation.png")

    # ── OFDM frame ──
    frame, syms = ofdm_frame(num_symbols=5, N=64, cp=16)
    fig4, axes = plt.subplots(2, 1, figsize=(13, 6))
    fig4.suptitle("OFDM Frame", fontsize=14, fontweight='bold')
    axes[0].plot(frame.real, color='steelblue', lw=0.8); axes[0].set_title("OFDM Time-Domain Signal (I component)"); axes[0].grid(alpha=0.3)
    axes[1].scatter(syms[0].real, syms[0].imag, color='darkorange', s=40); axes[1].set_title("QPSK Symbols on Subcarriers (one OFDM symbol)")
    axes[1].set_aspect('equal'); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig4.savefig("mod_ofdm.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: mod_ofdm.png")

    # ── BER curve ──
    fig5 = plot_ber(range(0, 13))
    fig5.savefig("mod_ber_bpsk.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: mod_ber_bpsk.png")

    print("\n✅ Modulation module demo complete — 5 figures saved.")
    plt.close('all')
