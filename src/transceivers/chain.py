"""
transceivers/chain.py
=====================
Full TX → Channel → RX communication chain simulation.
Covers: pulse shaping, up/down conversion, channel models,
        matched filter, timing recovery, equalization, BER.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp


# ─────────────────────────────────────────────
# 1. PULSE SHAPING
# ─────────────────────────────────────────────

def raised_cosine_filter(beta, span, sps):
    """
    Raised Cosine (RC) pulse shaping filter.
    beta : roll-off factor (0 to 1)
    span : filter span in symbols
    sps  : samples per symbol
    """
    N = span * sps + 1
    t = np.arange(-span/2, span/2 + 1/sps, 1/sps)
    t = np.where(np.abs(t) < 1e-10, 1e-10, t)   # avoid division by zero

    if beta == 0:
        h = np.sinc(t)
    else:
        h = (np.sinc(t) * np.cos(np.pi * beta * t)
             / (1 - (2 * beta * t)**2 + 1e-10))

    # Handle singularities at t = ±1/(2β)
    t_sing = 1 / (2 * beta)
    mask   = np.abs(np.abs(t) - t_sing) < 1e-4
    h[mask] = (beta / 2) * np.sin(np.pi / (2 * beta))

    return h / np.max(h)


def root_raised_cosine_filter(beta, span, sps):
    """
    Root Raised Cosine (RRC) — used at both TX and RX for matched filtering.
    TX: RRC → channel → RX: RRC → RC (optimal ISI-free detection)
    """
    N = span * sps + 1
    t = np.arange(-span/2, span/2 + 1/sps, 1/sps)
    h = np.zeros(len(t))

    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1 + beta * (4/np.pi - 1)
        elif abs(abs(ti) - 1/(4*beta)) < 1e-8:
            h[i] = (beta/np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*beta))
                                          + (1 - 2/np.pi) * np.cos(np.pi/(4*beta)))
        else:
            num = (np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta)))
            den = np.pi * ti * (1 - (4*beta*ti)**2)
            h[i] = num / den

    return h / np.max(h)


# ─────────────────────────────────────────────
# 2. TRANSMITTER
# ─────────────────────────────────────────────

def transmitter(bits, sps=8, beta=0.35, fc=1000, fs=8000, modulation='bpsk'):
    """
    Full transmitter chain:
    bits → symbol mapping → upsample → pulse shape → up-convert
    """
    # Symbol mapping
    if modulation == 'bpsk':
        symbols = 2 * bits.astype(float) - 1     # {0,1} → {-1,+1}
    elif modulation == 'qpsk':
        if len(bits) % 2:
            bits = np.append(bits, 0)
        I = 2 * bits[0::2] - 1
        Q = 2 * bits[1::2] - 1
        symbols = I + 1j * Q
    else:
        symbols = 2 * bits.astype(float) - 1

    # Upsample
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols

    # Pulse shape with RRC
    rrc = root_raised_cosine_filter(beta, span=6, sps=sps)
    shaped = sp.lfilter(rrc, [1.0], upsampled.real)
    if modulation == 'qpsk':
        shaped_q = sp.lfilter(rrc, [1.0], upsampled.imag)
        shaped = shaped + 1j * shaped_q

    # Up-convert (modulate to carrier)
    t = np.arange(len(shaped)) / fs
    tx = shaped.real * np.cos(2*np.pi*fc*t) - shaped.imag * np.sin(2*np.pi*fc*t)

    return tx, shaped, symbols, rrc


# ─────────────────────────────────────────────
# 3. CHANNEL MODELS
# ─────────────────────────────────────────────

def awgn_channel(signal, snr_db):
    """AWGN channel."""
    P_sig  = np.mean(signal**2)
    snr    = 10 ** (snr_db / 10)
    P_noise = P_sig / snr
    noise  = np.sqrt(P_noise) * np.random.randn(len(signal))
    return signal + noise, noise


def rayleigh_fading_channel(signal, snr_db, fd=10, fs=8000):
    """
    Rayleigh flat-fading channel + AWGN.
    fd: Doppler frequency (Hz)
    """
    N     = len(signal)
    t     = np.arange(N) / fs

    # Clarke's model: sum of phasors
    N_phasors = 16
    angles = np.random.uniform(0, 2*np.pi, N_phasors)
    h = np.zeros(N, dtype=complex)
    for alpha in angles:
        h += np.exp(1j * (2*np.pi*fd*t*np.cos(alpha) + alpha))
    h /= np.sqrt(N_phasors)

    faded = signal * np.abs(h)
    faded_noisy, _ = awgn_channel(faded, snr_db)
    return faded_noisy, np.abs(h)


def multipath_channel(signal, delays_samples, gains):
    """
    Multipath channel: sum of delayed copies.
    delays_samples: list of integer delays
    gains: corresponding gain for each path
    """
    out = np.zeros_like(signal, dtype=float)
    for d, g in zip(delays_samples, gains):
        if d < len(signal):
            out[d:] += g * signal[:-d] if d > 0 else g * signal
    return out


# ─────────────────────────────────────────────
# 4. RECEIVER
# ─────────────────────────────="────────────────
# ─────────────────────────────────────────────

def receiver(rx_signal, sps=8, beta=0.35, fc=1000, fs=8000,
             modulation='bpsk', delay=0):
    """
    Full receiver chain:
    down-convert → matched filter → downsample → decide
    """
    t = np.arange(len(rx_signal)) / fs

    # Down-convert
    I_raw = rx_signal * np.cos(2*np.pi*fc*t)   # in-phase
    Q_raw = rx_signal * (-np.sin(2*np.pi*fc*t)) # quadrature

    # Low-pass filter (remove 2fc component)
    nyq   = fs / 2
    b, a  = sp.butter(6, (fc*0.8)/nyq, btype='low')
    I_lpf = sp.filtfilt(b, a, I_raw) * 2
    Q_lpf = sp.filtfilt(b, a, Q_raw) * 2

    # Matched filter (RRC again)
    rrc   = root_raised_cosine_filter(beta, span=6, sps=sps)
    I_mf  = sp.lfilter(rrc, [1.0], I_lpf)
    Q_mf  = sp.lfilter(rrc, [1.0], Q_lpf)

    # Downsample at symbol rate
    rrc_delay = 3 * sps   # RRC filter group delay
    start = rrc_delay + delay
    I_sym = I_mf[start::sps]
    Q_sym = Q_mf[start::sps]

    # Decision
    if modulation == 'bpsk':
        decoded = (I_sym < 0).astype(int)
    elif modulation == 'qpsk':
        I_bits  = (I_sym < 0).astype(int)
        Q_bits  = (Q_sym < 0).astype(int)
        decoded = np.empty(len(I_bits)*2, dtype=int)
        decoded[0::2] = I_bits
        decoded[1::2] = Q_bits
    else:
        decoded = (I_sym < 0).astype(int)

    return decoded, I_sym, Q_sym


# ─────────────────────────────────────────────
# 5. EYE DIAGRAM
# ─────────────────────────────────────────────

def eye_diagram(signal, sps, num_periods=2, title="Eye Diagram"):
    """Plot eye diagram by overlaying consecutive symbol periods."""
    period = sps * num_periods
    n_traces = (len(signal) - period) // sps
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(min(n_traces, 200)):
        start = i * sps
        ax.plot(signal[start:start+period], color='steelblue', alpha=0.15, lw=0.8)
    ax.axvline(sps, color='red', ls='--', lw=1.0, label='Sampling instant')
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 6. FULL CHAIN BER vs SNR
# ─────────────────────────────────────────────

def ber_simulation_chain(snr_range, num_bits=1000, sps=4, fs=8000, fc=500):
    """Simulate full TX→channel→RX BER vs SNR."""
    bers = []
    rng  = np.random.default_rng(7)

    for snr_db in snr_range:
        bits = rng.integers(0, 2, num_bits)
        tx, _, tx_syms, _ = transmitter(bits, sps=sps, fc=fc, fs=fs)
        rx, _             = awgn_channel(tx, snr_db)
        rx_bits, _, _     = receiver(rx, sps=sps, fc=fc, fs=fs)

        # Align lengths
        n = min(len(bits), len(rx_bits))
        err  = np.sum(bits[:n] != rx_bits[:n])
        bers.append(max(err / n, 1e-6))

    return np.array(bers)


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fs, fc, sps = 8000, 800, 8

    print("=" * 55)
    print("  DSP PROJECT — Transceivers Demo")
    print("=" * 55)

    bits = rng.integers(0, 2, 50)

    # ── Pulse shaping ──
    rc  = raised_cosine_filter(0.35, span=6, sps=sps)
    rrc = root_raised_cosine_filter(0.35, span=6, sps=sps)
    t_f = np.arange(len(rc)) / (sps)

    fig1, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_f - len(rc)//(2*sps), rc,  color='steelblue',   lw=2, label='Raised Cosine (RC)')
    ax.plot(t_f - len(rrc)//(2*sps), rrc, color='darkorange',  lw=2, label='Root Raised Cosine (RRC)', ls='--')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("Symbol periods"); ax.set_ylabel("Amplitude")
    ax.set_title("Pulse Shaping Filters (β=0.35)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig("tx_pulse_shapes.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: tx_pulse_shapes.png")

    # ── Full TX chain ──
    tx, shaped, tx_syms, rrc_h = transmitter(bits, sps=sps, fc=fc, fs=fs)
    t_tx = np.arange(len(tx)) / fs

    fig2, axes = plt.subplots(3, 1, figsize=(13, 9))
    fig2.suptitle("Transmitter Chain — BPSK", fontsize=14, fontweight='bold')
    axes[0].stem(np.arange(len(tx_syms)), tx_syms, basefmt='k-', linefmt='C0-', markerfmt='C0o')
    axes[0].set_title("BPSK Symbols (±1)"); axes[0].grid(alpha=0.3)
    axes[1].plot(t_tx[:len(shaped.real)*1], shaped.real[:len(t_tx)], color='mediumseagreen', lw=1)
    axes[1].set_title("Pulse-Shaped Baseband"); axes[1].grid(alpha=0.3)
    axes[2].plot(t_tx, tx, color='steelblue', lw=0.8)
    axes[2].set_title(f"RF Signal (fc={fc} Hz)"); axes[2].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel("Time (s)")
    plt.tight_layout()
    fig2.savefig("tx_chain.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: tx_chain.png")

    # ── Full RX chain (AWGN) ──
    snr = 15
    rx, noise = awgn_channel(tx, snr)
    rx_bits, I_syms, Q_syms = receiver(rx, sps=sps, fc=fc, fs=fs)
    n = min(len(bits), len(rx_bits))
    ber = np.sum(bits[:n] != rx_bits[:n]) / n

    fig3, axes = plt.subplots(2, 1, figsize=(13, 6))
    fig3.suptitle(f"Receiver Chain — BPSK (SNR={snr}dB, BER={ber:.3f})", fontsize=14, fontweight='bold')
    axes[0].plot(t_tx, rx, color='tomato', lw=0.8); axes[0].set_title("Received Signal (noisy RF)"); axes[0].grid(alpha=0.3)
    axes[1].stem(np.arange(len(I_syms[:n])), I_syms[:n], basefmt='k-', linefmt='C2-', markerfmt='C2o')
    axes[1].set_title("Sampled I symbols after Matched Filter"); axes[1].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel("Time / Symbol index")
    plt.tight_layout()
    fig3.savefig("rx_chain.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: rx_chain.png")

    # ── Eye diagram ──
    shaped_real = shaped.real
    fig4 = eye_diagram(shaped_real, sps=sps, num_periods=2, title="Eye Diagram — RRC Pulse Shaping (β=0.35)")
    fig4.savefig("tx_eye_diagram.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: tx_eye_diagram.png")

    # ── Channel model comparison ──
    snr_range = range(0, 18, 2)
    print("Running BER simulation (this may take ~10 sec)...")
    bers_awgn = ber_simulation_chain(list(snr_range), num_bits=500, sps=sps, fs=fs, fc=fc)
    from scipy.special import erfc
    snr_arr   = np.array(list(snr_range))
    ber_theory = 0.5 * erfc(np.sqrt(10**(snr_arr/10)))

    fig5, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(snr_arr, ber_theory,  'b-',  lw=2,   label='BPSK Theory')
    ax.semilogy(snr_arr, bers_awgn,   'ro--', lw=1.5, label='Full Chain Simulation', markersize=6)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.set_title("Full TX→AWGN Channel→RX BER Performance")
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-4, 1])
    plt.tight_layout()
    fig5.savefig("rx_ber_chain.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: rx_ber_chain.png")

    print("\n✅ Transceivers module demo complete — 5 figures saved.")
    plt.close('all')
