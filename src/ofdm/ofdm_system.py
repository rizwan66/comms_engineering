"""
src/ofdm/ofdm_system.py
========================
Full OFDM system with:
  - QPSK/16-QAM symbol mapping on each subcarrier
  - IFFT (TX) and FFT (RX) pair
  - Cyclic prefix insertion and removal
  - Frequency-selective multipath channel
  - Per-subcarrier flat fading (the key OFDM insight)
  - Zero-Forcing (ZF) frequency-domain equalizer
  - BER vs SNR curves vs AWGN theory
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc


# ─────────────────────────────────────────────
# 1. SYMBOL MAPPING
# ─────────────────────────────────────────────

def qpsk_map(bits):
    """Map pairs of bits to QPSK symbols (Gray coded)."""
    bits = np.asarray(bits)
    if len(bits) % 2:
        bits = np.append(bits, 0)
    b = bits.reshape(-1, 2)
    # 00→+1+j, 01→-1+j, 11→-1-j, 10→+1-j  (Gray)
    lut = {(0,0): 1+1j, (0,1): -1+1j, (1,1): -1-1j, (1,0): 1-1j}
    return np.array([lut[(b[i,0], b[i,1])] for i in range(len(b))]) / np.sqrt(2)


def qpsk_demap(symbols):
    """Hard-decision QPSK demap."""
    bits = np.zeros(len(symbols)*2, dtype=int)
    bits[0::2] = (symbols.real < 0).astype(int)
    bits[1::2] = (symbols.imag < 0).astype(int)
    return bits


def qam16_map(bits):
    """Gray-coded 16-QAM mapper. Returns normalised symbols."""
    bits = np.asarray(bits)
    pad = (-len(bits)) % 4
    if pad:
        bits = np.append(bits, np.zeros(pad, int))
    b = bits.reshape(-1, 4)
    gray2bin = [0, 1, 3, 2]
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)   # normalise to unit avg power
    I_idx = b[:,0]*2 + b[:,1]
    Q_idx = b[:,2]*2 + b[:,3]
    I = levels[[gray2bin[i] for i in I_idx]]
    Q = levels[[gray2bin[i] for i in Q_idx]]
    return I + 1j*Q


def qam16_demap(symbols):
    """Hard-decision 16-QAM demap."""
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    bin2gray = [0, 1, 3, 2]
    def nearest(v):
        idx = np.argmin(np.abs(levels - v))
        g = bin2gray[idx]
        return [(g >> 1) & 1, g & 1]
    bits = []
    for s in symbols:
        bits += nearest(s.real) + nearest(s.imag)
    return np.array(bits, dtype=int)


# ─────────────────────────────────────────────
# 2. OFDM FRAME BUILDER
# ─────────────────────────────────────────────

class OFDMSystem:
    """
    OFDM system with configurable subcarriers, CP, and modulation.

    Parameters
    ----------
    N_fft      : FFT size (total subcarriers including guard)
    N_data     : number of data subcarriers (centred, DC excluded)
    cp_len     : cyclic prefix length (samples)
    modulation : 'qpsk' or '16qam'
    pilot_spacing : insert pilots every N subcarriers (0 = no pilots)
    """

    def __init__(self, N_fft=64, N_data=52, cp_len=16,
                 modulation='qpsk', pilot_spacing=8):
        self.N_fft       = N_fft
        self.N_data      = N_data
        self.cp_len      = cp_len
        self.modulation  = modulation
        self.pilot_spacing = pilot_spacing
        self.bits_per_sym = 2 if modulation == 'qpsk' else 4

        # Subcarrier index mapping (skip DC and guard bands)
        guard = (N_fft - N_data) // 2
        self.data_idx = np.concatenate([
            np.arange(1, N_data//2 + 1),                      # positive
            np.arange(N_fft - N_data//2, N_fft)               # negative (upper half)
        ])

        # Pilot positions within data_idx
        if pilot_spacing > 0:
            self.pilot_pos = self.data_idx[::pilot_spacing]
            self.data_pos  = np.setdiff1d(self.data_idx, self.pilot_pos)
        else:
            self.pilot_pos = np.array([], dtype=int)
            self.data_pos  = self.data_idx

        self.n_data_carriers = len(self.data_pos)

    @property
    def bits_per_ofdm_symbol(self):
        return self.n_data_carriers * self.bits_per_sym

    # ── TX ──────────────────────────────────────

    def modulate_bits(self, bits):
        """Map bits → complex symbols."""
        if self.modulation == 'qpsk':
            return qpsk_map(bits)
        else:
            return qam16_map(bits)

    def build_ofdm_symbol(self, data_symbols):
        """
        Place data + pilot symbols into frequency-domain frame,
        run IFFT, prepend CP.
        """
        freq = np.zeros(self.N_fft, dtype=complex)

        # Pilots: known BPSK pilots (value = 1+0j)
        freq[self.pilot_pos] = 1.0 + 0j

        # Data
        n = min(len(data_symbols), self.n_data_carriers)
        freq[self.data_pos[:n]] = data_symbols[:n]

        # IFFT → time domain
        time = np.fft.ifft(freq, n=self.N_fft) * np.sqrt(self.N_fft)

        # Cyclic prefix: copy last cp_len samples to front
        cp   = time[-self.cp_len:]
        return np.concatenate([cp, time])

    def transmit(self, bits):
        """Transmit a stream of bits → OFDM waveform."""
        # Pad bits to fill complete OFDM symbols
        bps = self.bits_per_ofdm_symbol
        pad = (-len(bits)) % bps
        bits = np.append(bits, np.zeros(pad, int))

        frames = []
        for i in range(0, len(bits), bps):
            chunk   = bits[i:i+bps]
            symbols = self.modulate_bits(chunk)
            frame   = self.build_ofdm_symbol(symbols)
            frames.append(frame)
        return np.concatenate(frames), len(bits) // bps

    # ── RX ──────────────────────────────────────

    def receive_ofdm_symbol(self, rx_frame, H_est=None):
        """
        Remove CP, FFT, equalize, extract data subcarriers.
        H_est : per-subcarrier channel estimate (complex). If None → no EQ.
        """
        # Remove CP
        time = rx_frame[self.cp_len : self.cp_len + self.N_fft]

        # FFT
        freq = np.fft.fft(time, n=self.N_fft) / np.sqrt(self.N_fft)

        # Channel estimation from pilots (ZF: divide by known pilot)
        if H_est is None and len(self.pilot_pos) > 0:
            H_pilots = freq[self.pilot_pos] / 1.0   # known pilot = 1
            # Interpolate to all data subcarriers (linear)
            all_idx  = self.data_idx
            H_est_all = np.interp(
                self.data_pos,
                self.pilot_pos,
                H_pilots.real
            ) + 1j * np.interp(
                self.data_pos,
                self.pilot_pos,
                H_pilots.imag
            )
        elif H_est is not None:
            H_est_all = H_est[self.data_pos]
        else:
            H_est_all = np.ones(self.n_data_carriers, dtype=complex)

        # ZF equalization: divide by channel
        data_rx = freq[self.data_pos] / (H_est_all + 1e-10)

        return data_rx, freq

    def demodulate_symbols(self, symbols):
        if self.modulation == 'qpsk':
            return qpsk_demap(symbols)
        else:
            return qam16_demap(symbols)

    def receive(self, rx_signal, n_symbols, true_channel=None):
        """Full RX: split into OFDM symbols, equalize, decode bits."""
        sym_len = self.N_fft + self.cp_len
        all_bits = []

        for i in range(n_symbols):
            frame   = rx_signal[i*sym_len : (i+1)*sym_len]
            if len(frame) < sym_len:
                break

            # Use true channel for perfect CSI (genie equalizer)
            H_est = true_channel if true_channel is not None else None
            data_rx, _ = self.receive_ofdm_symbol(frame, H_est)
            bits = self.demodulate_symbols(data_rx)
            all_bits.append(bits)

        return np.concatenate(all_bits) if all_bits else np.array([], int)


# ─────────────────────────────────────────────
# 3. CHANNEL MODELS
# ─────────────────────────────────────────────

def awgn(signal, snr_db):
    """AWGN channel."""
    P     = np.mean(np.abs(signal)**2)
    N0    = P / (10**(snr_db/10))
    noise = np.sqrt(N0/2) * (np.random.randn(len(signal))
                             + 1j*np.random.randn(len(signal)))
    return signal + noise


def multipath_channel(signal, delays, gains):
    """
    Frequency-selective multipath: y = sum_k g_k * x[n - d_k]
    delays : list of integer sample delays
    gains  : complex path gains
    """
    out = np.zeros(len(signal) + max(delays), dtype=complex)
    for d, g in zip(delays, gains):
        out[d:d+len(signal)] += g * signal
    return out[:len(signal)]


def channel_freq_response(delays, gains, N_fft):
    """True frequency-domain channel H[k] for perfect CSI equalizer."""
    H = np.zeros(N_fft, dtype=complex)
    for d, g in zip(delays, gains):
        H += g * np.exp(-1j * 2*np.pi * d * np.arange(N_fft) / N_fft)
    return H


# ─────────────────────────────────────────────
# 4. BER SIMULATION
# ─────────────────────────────────────────────

def simulate_ber(ofdm, snr_db_range, channel_type='awgn',
                 delays=None, gains=None, n_bits=8000):
    """Monte-Carlo BER for OFDM over AWGN or multipath."""
    rng  = np.random.default_rng(42)
    bers = []

    delays = delays or [0]
    gains  = gains  or [1.0]

    H_true = channel_freq_response(delays, gains, ofdm.N_fft) \
             if channel_type == 'multipath' else None

    for snr_db in snr_db_range:
        bits = rng.integers(0, 2, n_bits)
        tx, n_sym = ofdm.transmit(bits)

        if channel_type == 'multipath':
            rx = multipath_channel(tx, delays, gains)
        else:
            rx = tx.copy()

        rx = awgn(rx, snr_db)

        rx_bits = ofdm.receive(rx, n_sym, true_channel=H_true)
        n = min(len(bits), len(rx_bits))
        ber = np.sum(bits[:n] != rx_bits[:n]) / n
        bers.append(max(ber, 1e-6))

    return np.array(bers)


def qpsk_ber_theory(snr_db):
    """Theoretical QPSK BER in AWGN."""
    return 0.5 * erfc(np.sqrt(10**(np.array(snr_db)/10)))


# ─────────────────────────────────────────────
# 5. DEMO & PLOTS
# ─────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(0)

    print("=" * 58)
    print("  OFDM System Demo")
    print("  N_fft=64, CP=16, QPSK, multipath channel")
    print("=" * 58)

    # System config
    ofdm = OFDMSystem(N_fft=64, N_data=48, cp_len=16,
                      modulation='qpsk', pilot_spacing=6)

    print(f"  Data subcarriers  : {ofdm.n_data_carriers}")
    print(f"  Pilot subcarriers : {len(ofdm.pilot_pos)}")
    print(f"  Bits/OFDM symbol  : {ofdm.bits_per_ofdm_symbol}")
    print(f"  CP length         : {ofdm.cp_len} samples")

    # Multipath channel: 3-tap (LOS + 2 reflections)
    delays = [0, 4, 9]           # sample delays (must be < cp_len)
    gains  = [1.0, 0.5*np.exp(1j*0.8), 0.3*np.exp(-1j*1.2)]

    # Generate TX signal
    n_bits = ofdm.bits_per_ofdm_symbol * 20
    rng    = np.random.default_rng(7)
    bits_tx = rng.integers(0, 2, n_bits)
    tx, n_sym = ofdm.transmit(bits_tx)

    # Pass through multipath + AWGN (SNR=15dB)
    rx_mp   = multipath_channel(tx, delays, gains)
    rx_noisy = awgn(rx_mp, snr_db=15)

    # True channel frequency response
    H_true = channel_freq_response(delays, gains, ofdm.N_fft)

    # ── Figure 1: System overview ────────────────
    fig1 = plt.figure(figsize=(16, 12))
    gs   = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.45, wspace=0.35)
    fig1.suptitle("OFDM System — Cyclic Prefix · IFFT/FFT · Multipath · ZF Equalizer",
                  fontsize=13, fontweight='bold')

    # 1a. TX time-domain signal
    ax = fig1.add_subplot(gs[0, :2])
    t  = np.arange(len(tx))
    ax.plot(t[:300], tx.real[:300], color='steelblue', lw=0.9, label='I (real)')
    ax.plot(t[:300], tx.imag[:300], color='darkorange', lw=0.9, alpha=0.7, label='Q (imag)')
    ax.axvspan(0, ofdm.cp_len, alpha=0.15, color='red', label=f'CP ({ofdm.cp_len} samples)')
    ax.axvspan(ofdm.cp_len, ofdm.N_fft+ofdm.cp_len, alpha=0.08, color='green', label='OFDM symbol')
    ax.set_title("TX Signal (first 300 samples)")
    ax.set_xlabel("Sample"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 1b. Frequency-domain channel |H(k)|
    ax2 = fig1.add_subplot(gs[0, 2])
    k   = np.arange(ofdm.N_fft)
    ax2.plot(k, np.abs(H_true), color='tomato', lw=1.8)
    ax2.set_title("|H(k)| — Channel Freq. Response\n(frequency-selective)")
    ax2.set_xlabel("Subcarrier k"); ax2.set_ylabel("|H|"); ax2.grid(alpha=0.3)

    # 1c. Received spectrum (one symbol) before/after EQ
    frame_rx = rx_noisy[0:ofdm.N_fft+ofdm.cp_len]
    time_sym  = frame_rx[ofdm.cp_len:]
    freq_sym  = np.fft.fft(time_sym) / np.sqrt(ofdm.N_fft)
    freq_eq   = freq_sym / (H_true + 1e-10)

    ax3 = fig1.add_subplot(gs[1, 0])
    ax3.plot(np.abs(freq_sym), color='tomato', lw=1.2, label='Before EQ')
    ax3.plot(np.abs(freq_eq),  color='steelblue', lw=1.2, label='After ZF EQ', alpha=0.8)
    ax3.set_title("Subcarrier Magnitudes"); ax3.legend(fontsize=8)
    ax3.set_xlabel("Subcarrier"); ax3.grid(alpha=0.3)

    # 1d. IQ constellation before EQ
    ax4 = fig1.add_subplot(gs[1, 1])
    syms_raw, _ = ofdm.receive_ofdm_symbol(frame_rx, H_est=np.ones(ofdm.N_fft))
    ax4.scatter(syms_raw.real, syms_raw.imag, alpha=0.5, s=20, color='tomato')
    ax4.set_title("Constellation — No Equalization"); ax4.set_aspect('equal')
    ax4.axhline(0, color='k', lw=0.4); ax4.axvline(0, color='k', lw=0.4); ax4.grid(alpha=0.3)

    # 1e. IQ constellation after ZF EQ
    ax5 = fig1.add_subplot(gs[1, 2])
    syms_eq, _ = ofdm.receive_ofdm_symbol(frame_rx, H_est=H_true)
    ax5.scatter(syms_eq.real, syms_eq.imag, alpha=0.6, s=20, color='steelblue')
    ax5.set_title("Constellation — After ZF Equalizer"); ax5.set_aspect('equal')
    ax5.axhline(0, color='k', lw=0.4); ax5.axvline(0, color='k', lw=0.4); ax5.grid(alpha=0.3)

    # 1f. Spectrogram of TX
    ax6 = fig1.add_subplot(gs[2, :2])
    ax6.specgram(tx.real, NFFT=64, Fs=1, noverlap=32, cmap='plasma')
    ax6.set_title("TX Spectrogram — Flat Spectrum Across All Subcarriers")
    ax6.set_xlabel("Time"); ax6.set_ylabel("Frequency")

    # 1g. CP guard interval illustration
    ax7 = fig1.add_subplot(gs[2, 2])
    sym_len = ofdm.N_fft + ofdm.cp_len
    one_sym = tx[:sym_len].real
    ax7.plot(np.arange(ofdm.cp_len), one_sym[:ofdm.cp_len],
             color='red', lw=2, label='Cyclic Prefix')
    ax7.plot(np.arange(ofdm.cp_len, sym_len), one_sym[ofdm.cp_len:],
             color='steelblue', lw=1.5, label='OFDM Symbol')
    ax7.plot(np.arange(ofdm.N_fft), one_sym[ofdm.cp_len:],
             color='red', lw=0.8, ls='--', alpha=0.5)
    ax7.set_title("Cyclic Prefix = Copy of Last N_cp Samples")
    ax7.legend(fontsize=8); ax7.grid(alpha=0.3)

    fig1.savefig("ofdm_system.png", dpi=130, bbox_inches='tight')
    print("✓ Saved: ofdm_system.png")

    # ── Figure 2: BER curves ─────────────────────
    print("Running BER simulations...")
    snr_range = list(range(0, 20, 2))

    ber_awgn    = simulate_ber(ofdm, snr_range, 'awgn', n_bits=6000)
    ber_mp_eq   = simulate_ber(ofdm, snr_range, 'multipath',
                               delays=delays, gains=gains, n_bits=6000)
    ber_theory  = qpsk_ber_theory(snr_range)

    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(snr_range, ber_theory, 'k-',   lw=2,   label='QPSK Theory (AWGN)')
    ax.semilogy(snr_range, ber_awgn,   'b--o',  lw=1.5, ms=6, label='OFDM Simulation (AWGN)')
    ax.semilogy(snr_range, ber_mp_eq,  'r--s',  lw=1.5, ms=6, label='OFDM + Multipath + ZF EQ')
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Bit Error Rate", fontsize=12)
    ax.set_title("OFDM BER: AWGN vs Frequency-Selective Multipath + ZF Equalizer", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-4, 1]); ax.set_xlim([0, 18])
    fig2.savefig("ofdm_ber.png", dpi=130, bbox_inches='tight')
    print("✓ Saved: ofdm_ber.png")

    print("\n✅ OFDM module complete.")
    plt.close('all')
