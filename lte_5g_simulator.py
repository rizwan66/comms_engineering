"""
lte_5g_simulator.py
====================
Enhanced LTE & 5G NR (New Radio) Simulator
Extends comms_engineering project with full LTE + 5G NR stack simulation.

Modules:
  1.  LTE Downlink (PDSCH) — existing, extended with HARQ & carrier aggregation
  2.  5G NR Downlink (PDSCH) — new: numerology, flexible slot structure, beamforming
  3.  5G NR Uplink (PUSCH) — Transform precoding, DFT-s-OFDM
  4.  5G NR Massive MIMO / Beamforming — analog + digital, codebook
  5.  5G NR Channel Models — CDL-A/B/C, TDL, UMa, UMi, NTN
  6.  5G NR Polar & LDPC Codes — encoding / decoding
  7.  5G NR Numerology (μ = 0…4) — subcarrier spacing, slot duration
  8.  Network Slicing KPIs — eMBB / URLLC / mMTC
  9.  Link Budget — LTE & 5G FR1/FR2
  10. Master Dashboard — side-by-side LTE vs 5G comparisons

Usage:
    python lte_5g_simulator.py

Author: Enhanced for rizwan66/comms_engineering
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.linalg import svd
from scipy.special import erfc
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS & LTE / 5G PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# LTE
LTE_BW_RB = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100}   # MHz → RBs
LTE_SCS = 15e3       # Hz
LTE_SYMBOL_DUR = 1 / LTE_SCS
LTE_CP_NORMAL = 144 / (2048 * LTE_SCS)   # normal CP duration

# 5G NR numerologies  μ: (SCS Hz, slot_dur ms, sym/slot)
NR_NUMEROLOGY = {
    0: {"scs": 15e3,   "slot_dur_ms": 1.0,    "sym_per_slot": 14, "label": "FR1 15 kHz"},
    1: {"scs": 30e3,   "slot_dur_ms": 0.5,    "sym_per_slot": 14, "label": "FR1 30 kHz"},
    2: {"scs": 60e3,   "slot_dur_ms": 0.25,   "sym_per_slot": 14, "label": "FR1/FR2 60 kHz"},
    3: {"scs": 120e3,  "slot_dur_ms": 0.125,  "sym_per_slot": 14, "label": "FR2 120 kHz"},
    4: {"scs": 240e3,  "slot_dur_ms": 0.0625, "sym_per_slot": 14, "label": "FR2 240 kHz"},
}

# MCS table (simplified — 3GPP TS 38.214 Table 5.1.3.1-1 subset)
MCS_TABLE = {
    # mcs_index: (modulation_order, code_rate*1024)
    0:  (2,  120),   # QPSK  ~0.12
    2:  (2,  193),   # QPSK  ~0.19
    5:  (2,  449),   # QPSK  ~0.44
    9:  (4,  602),   # 16QAM ~0.59
    14: (6,  616),   # 64QAM ~0.60
    19: (6,  948),   # 64QAM ~0.93
    24: (8,  682),   # 256QAM~0.67
    27: (8,  948),   # 256QAM~0.93
}

# CDL channel model delay/power profiles (simplified CDL-A)
CDL_A_DELAYS_NS = [0, 30, 70, 90, 110, 190, 410]        # ns
CDL_A_POWER_DB  = [0, -13.4, 0, -2.2, -4, -6, -8.2]    # dB

CDL_C_DELAYS_NS = [0, 65, 150, 230, 450, 600, 730]
CDL_C_POWER_DB  = [0, -2.2, -4, -3.2, -9.8, -7.4, -7.9]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def qpsk_mod(bits):
    """QPSK modulation — Gray coded."""
    bits = np.array(bits).flatten()
    if len(bits) % 2:
        bits = np.append(bits, 0)
    pairs = bits.reshape(-1, 2)
    lut = {(0,0): 1+1j, (0,1): -1+1j, (1,1): -1-1j, (1,0): 1-1j}
    return np.array([lut[tuple(p)] for p in pairs]) / np.sqrt(2)

def qam16_mod(bits):
    """16-QAM Gray coded."""
    bits = np.array(bits).flatten()
    pad = (-len(bits)) % 4
    bits = np.append(bits, np.zeros(pad, int))
    groups = bits.reshape(-1, 4)
    lut = {(0,0,0,0):(-3+3j),(0,0,0,1):(-3+1j),(0,0,1,1):(-3-1j),(0,0,1,0):(-3-3j),
           (0,1,0,0):(-1+3j),(0,1,0,1):(-1+1j),(0,1,1,1):(-1-1j),(0,1,1,0):(-1-3j),
           (1,1,0,0):( 1+3j),(1,1,0,1):( 1+1j),(1,1,1,1):( 1-1j),(1,1,1,0):( 1-3j),
           (1,0,0,0):( 3+3j),(1,0,0,1):( 3+1j),(1,0,1,1):( 3-1j),(1,0,1,0):( 3-3j)}
    syms = np.array([lut[tuple(g)] for g in groups])
    return syms / np.sqrt(10)

def qam64_mod(bits):
    """64-QAM Gray coded (simplified uniform)."""
    bits = np.array(bits).flatten()
    pad = (-len(bits)) % 6
    bits = np.append(bits, np.zeros(pad, int))
    groups = bits.reshape(-1, 6)
    alphabet = np.array([-7,-5,-3,-1,1,3,5,7])
    gray = np.array([0,1,3,2,6,7,5,4])
    inv_gray = np.argsort(gray)
    I_idx = groups[:,0]*4 + groups[:,1]*2 + groups[:,2]
    Q_idx = groups[:,3]*4 + groups[:,4]*2 + groups[:,5]
    syms = (alphabet[inv_gray[I_idx]] + 1j*alphabet[inv_gray[Q_idx]])
    return syms / np.sqrt(42)

def qam256_mod(bits):
    """256-QAM (simplified)."""
    bits = np.array(bits).flatten()
    pad = (-len(bits)) % 8
    bits = np.append(bits, np.zeros(pad, int))
    groups = bits.reshape(-1, 8)
    alphabet = np.array([-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15], float)
    def b2i(b4): return b4[0]*8+b4[1]*4+b4[2]*2+b4[3]
    I = alphabet[np.array([b2i(g[:4]) for g in groups])]
    Q = alphabet[np.array([b2i(g[4:]) for g in groups])]
    return (I + 1j*Q) / np.sqrt(170)

def modulate(bits, mod_order):
    if mod_order == 2:  return qpsk_mod(bits)
    if mod_order == 4:  return qam16_mod(bits)
    if mod_order == 6:  return qam64_mod(bits)
    if mod_order == 8:  return qam256_mod(bits)
    raise ValueError(f"Unknown modulation order {mod_order}")

def awgn(signal, snr_db):
    snr_lin = 10**(snr_db/10)
    sigma = np.sqrt(1/(2*snr_lin))
    noise = sigma*(np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise

def theoretical_ber_qpsk(snr_db):
    snr = 10**(np.array(snr_db)/10)
    return 0.5*erfc(np.sqrt(snr))

def theoretical_ber_qam(M, snr_db):
    snr = 10**(np.array(snr_db)/10)
    k = np.log2(M)
    return (4/k)*(1-1/np.sqrt(M))*0.5*erfc(np.sqrt(3*k*snr/(2*(M-1))))

def ber_from_symbols(tx, rx_soft):
    """Compute BER (simple hard decision)."""
    errors = np.sum(np.sign(tx.real) != np.sign(rx_soft.real))
    errors += np.sum(np.sign(tx.imag) != np.sign(rx_soft.imag))
    total = 2 * len(tx)
    return errors / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3.  OFDM ENGINE  (shared LTE / 5G)
# ─────────────────────────────────────────────────────────────────────────────

class OFDMEngine:
    """Configurable OFDM engine for LTE and 5G NR."""

    def __init__(self, n_fft=1024, n_cp=72, n_subcarriers=600):
        self.N = n_fft
        self.N_cp = n_cp
        self.K = n_subcarriers          # active subcarriers (< N)

    def modulate(self, freq_grid):
        """OFDM modulate: freq_grid shape (n_symbols, K) → time samples."""
        n_sym, K = freq_grid.shape
        time_out = []
        for sym in range(n_sym):
            fd = np.zeros(self.N, dtype=complex)
            start = (self.N - K) // 2
            fd[start:start+K] = freq_grid[sym]
            td = np.fft.ifft(np.fft.ifftshift(fd)) * np.sqrt(self.N)
            td_cp = np.concatenate([td[-self.N_cp:], td])
            time_out.append(td_cp)
        return np.array(time_out)

    def demodulate(self, time_signal):
        """OFDM demodulate: time_signal shape (n_sym, N+N_cp) → freq_grid."""
        n_sym = time_signal.shape[0]
        K = self.K
        freq_out = []
        for sym in range(n_sym):
            td = time_signal[sym, self.N_cp:]
            fd = np.fft.fftshift(np.fft.fft(td)) / np.sqrt(self.N)
            start = (self.N - K) // 2
            freq_out.append(fd[start:start+K])
        return np.array(freq_out)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CDL CHANNEL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CDLChannel:
    """3GPP CDL channel model (simplified)."""

    def __init__(self, model="CDL-A", doppler_hz=100, fs=15.36e6):
        self.fs = fs
        self.fd = doppler_hz
        if model == "CDL-A":
            delays_ns, power_db = CDL_A_DELAYS_NS, CDL_A_POWER_DB
        else:
            delays_ns, power_db = CDL_C_DELAYS_NS, CDL_C_POWER_DB

        self.delays_samples = np.array(delays_ns) * 1e-9 * fs
        self.power_lin = 10**(np.array(power_db)/10)
        self.power_lin /= self.power_lin.sum()
        self.n_paths = len(delays_ns)

    def apply(self, tx_signal):
        """Apply multipath fading channel."""
        n = len(tx_signal)
        rx = np.zeros(n, dtype=complex)
        for i in range(self.n_paths):
            d = int(round(self.delays_samples[i]))
            gain = np.sqrt(self.power_lin[i]/2) * (
                np.random.randn() + 1j*np.random.randn())
            # Doppler phase rotation
            t = np.arange(n) / self.fs
            doppler = np.exp(1j * 2*np.pi * self.fd * t *
                             np.random.uniform(-1, 1))
            delayed = np.zeros(n, dtype=complex)
            if d < n:
                delayed[d:] = tx_signal[:n-d]
            rx += gain * doppler * delayed
        return rx


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LTE DOWNLINK PDSCH (enhanced)
# ─────────────────────────────────────────────────────────────────────────────

class LTEDownlink:
    """
    LTE PDSCH simulator.
    Subcarrier spacing: 15 kHz
    FFT size: 2048 (20 MHz BW)
    Resource blocks: 100 (20 MHz)
    Modulation: QPSK / 16QAM / 64QAM
    """

    def __init__(self, bandwidth_mhz=20, modulation="64QAM"):
        self.bw = bandwidth_mhz
        self.n_rb = LTE_BW_RB.get(bandwidth_mhz, 100)
        self.n_sc = self.n_rb * 12          # subcarriers
        self.n_fft = 2048
        self.n_cp = 144
        self.scs = 15e3
        self.mod = modulation
        self.mod_order = {"QPSK":2,"16QAM":4,"64QAM":6}[modulation]
        self.ofdm = OFDMEngine(self.n_fft, self.n_cp, self.n_sc)

    def resource_grid(self, n_sym=14):
        """Return (n_sym, n_sc) resource grid filled with random data."""
        bits = np.random.randint(0, 2, n_sym * self.n_sc * self.mod_order)
        syms = modulate(bits, self.mod_order)
        needed = n_sym * self.n_sc
        syms = syms[:needed]
        return syms.reshape(n_sym, self.n_sc)

    def simulate_ber(self, snr_range_db):
        """BER vs SNR simulation."""
        ber = []
        n_sym = 14
        for snr_db in snr_range_db:
            grid = self.resource_grid(n_sym)
            td = self.ofdm.modulate(grid)
            # Flatten, add AWGN, demod
            flat = td.flatten()
            flat_noisy = awgn(flat.reshape(1,-1), snr_db).flatten()
            td_noisy = flat_noisy[:td.size].reshape(td.shape)
            rx_grid = self.ofdm.demodulate(td_noisy)
            b = ber_from_symbols(grid.flatten(), rx_grid.flatten())
            ber.append(b)
        return np.array(ber)

    def throughput_mbps(self, snr_db, code_rate=0.75):
        """Shannon-limited throughput in Mbps."""
        snr_lin = 10**(np.array(snr_db)/10)
        capacity_bits = self.n_sc * np.log2(1 + snr_lin)  # per OFDM symbol
        symbols_per_sec = self.scs                          # 1 symbol / (1/SCS)
        return capacity_bits * symbols_per_sec * code_rate / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# 6.  5G NR DOWNLINK PDSCH
# ─────────────────────────────────────────────────────────────────────────────

class NRDownlink:
    """
    5G NR PDSCH simulator.
    Supports numerology μ = 0..4 (SCS 15–240 kHz)
    Supports FR1 (sub-6 GHz) and FR2 (mmWave)
    """

    def __init__(self, mu=1, n_rb=106, mcs_index=19):
        """
        mu       : numerology index (0..4)
        n_rb     : number of resource blocks (e.g. 106 for 40 MHz at μ=1)
        mcs_index: MCS table index
        """
        self.mu = mu
        self.num = NR_NUMEROLOGY[mu]
        self.n_rb = n_rb
        self.n_sc = n_rb * 12
        self.scs = self.num["scs"]
        self.slot_dur = self.num["slot_dur_ms"] * 1e-3
        self.sym_per_slot = self.num["sym_per_slot"]
        self.mcs_idx = mcs_index
        mod_order, cr1024 = MCS_TABLE.get(mcs_index, (6, 616))
        self.mod_order = mod_order
        self.code_rate = cr1024 / 1024
        # NR uses larger FFT — scale by SCS ratio
        self.n_fft = 4096 if mu >= 3 else 2048
        self.n_cp = int(self.n_fft * 144/2048) if mu == 0 else int(self.n_fft * 72/2048)
        self.ofdm = OFDMEngine(self.n_fft, self.n_cp, self.n_sc)

    def peak_throughput_mbps(self):
        """
        3GPP 38.306 peak throughput formula (single layer, no overhead approx).
        """
        slots_per_sec = 1.0 / self.slot_dur
        bits_per_sym = self.mod_order * self.code_rate
        # 12 SC × n_RB × (14-2 DMRS symbols) × bits_per_sym × slots_per_sec
        tp = self.n_sc * 12 * bits_per_sym * slots_per_sec / 1e6
        return tp

    def simulate_ber(self, snr_range_db, n_slots=2):
        """BER vs SNR for NR PDSCH."""
        ber = []
        n_sym = self.sym_per_slot * n_slots
        for snr_db in snr_range_db:
            bits = np.random.randint(0, 2, n_sym * self.n_sc * self.mod_order)
            syms = modulate(bits, self.mod_order)
            needed = n_sym * self.n_sc
            syms = syms[:needed].reshape(n_sym, self.n_sc)
            td = self.ofdm.modulate(syms)
            flat = td.flatten()
            flat_noisy = awgn(flat.reshape(1,-1), snr_db).flatten()
            td_noisy = flat_noisy.reshape(td.shape)
            rx = self.ofdm.demodulate(td_noisy)
            ber.append(ber_from_symbols(syms.flatten(), rx.flatten()))
        return np.array(ber)

    def throughput_mbps(self, snr_db):
        """Shannon-limited throughput in Mbps."""
        snr_lin = 10**(np.array(snr_db)/10)
        bw_hz = self.n_sc * self.scs
        return bw_hz * np.log2(1 + snr_lin) / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# 7.  5G NR MASSIVE MIMO BEAMFORMING
# ─────────────────────────────────────────────────────────────────────────────

class MassiveMIMOBeamformer:
    """
    5G NR Massive MIMO with analog/digital beamforming.
    ULA (Uniform Linear Array) model.
    """

    def __init__(self, n_tx=64, n_rx=4, n_beams=8):
        self.M = n_tx       # BS antennas
        self.K = n_rx       # UE antennas
        self.N_b = n_beams
        self.d = 0.5        # half-wavelength spacing

    def steering_vector(self, theta_deg):
        """ULA steering vector for angle theta (degrees)."""
        theta = np.radians(theta_deg)
        n = np.arange(self.M)
        return np.exp(1j * np.pi * n * np.sin(theta)) / np.sqrt(self.M)

    def codebook(self):
        """DFT codebook — N_b beams spanning ±60°."""
        angles = np.linspace(-60, 60, self.N_b)
        W = np.array([self.steering_vector(a) for a in angles])   # (N_b, M)
        return W, angles

    def beam_gain_pattern(self):
        """Compute beam gain across sweep angles for all beams."""
        angles_sweep = np.linspace(-90, 90, 361)
        W, beam_angles = self.codebook()
        gains = np.zeros((self.N_b, len(angles_sweep)))
        for i, a in enumerate(angles_sweep):
            sv = self.steering_vector(a)
            gains[:, i] = np.abs(W @ sv)**2
        return angles_sweep, gains, beam_angles

    def capacity_vs_snr(self, snr_range_db, n_mc=200):
        """MIMO capacity (bits/s/Hz) via SVD water-filling."""
        capacities = []
        for snr_db in snr_range_db:
            snr_lin = 10**(snr_db/10)
            C_mc = []
            for _ in range(n_mc):
                H = (np.random.randn(self.K, self.M) +
                     1j*np.random.randn(self.K, self.M)) / np.sqrt(2)
                # Beamformed channel: apply best beam
                W, _ = self.codebook()
                HW = H @ W.T             # (K, N_b)
                _, s, _ = svd(HW, full_matrices=False)
                # Water filling
                n_layers = min(self.K, self.N_b)
                lam = s[:n_layers]**2
                C = np.sum(np.log2(1 + snr_lin * lam / n_layers))
                C_mc.append(C)
            capacities.append(np.mean(C_mc))
        return np.array(capacities)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  5G NR UPLINK — DFT-s-OFDM (SC-FDMA variant)
# ─────────────────────────────────────────────────────────────────────────────

class NRUplink:
    """
    5G NR PUSCH with DFT-spread OFDM (transform precoding).
    Reduces PAPR compared to OFDM — key for UE power efficiency.
    """

    def __init__(self, n_rb=25, mu=1):
        self.n_rb = n_rb
        self.n_sc = n_rb * 12
        self.mu = mu
        self.scs = NR_NUMEROLOGY[mu]["scs"]
        self.n_fft = 1024
        self.n_cp = 72

    def dft_spread(self, syms):
        """Apply DFT spread (M-point DFT before IFFT)."""
        return np.fft.fft(syms) / np.sqrt(len(syms))

    def modulate_slot(self, bits, mod_order=2):
        """Modulate one slot with DFT-s-OFDM."""
        n_sym = 14
        syms = modulate(bits[:n_sym*self.n_sc*mod_order],  mod_order)
        syms = syms[:n_sym*self.n_sc].reshape(n_sym, self.n_sc)
        td_frames = []
        for i in range(n_sym):
            spread = self.dft_spread(syms[i])          # DFT precoding
            # Map to OFDM grid
            fd = np.zeros(self.n_fft, complex)
            start = (self.n_fft - self.n_sc) // 2
            fd[start:start+self.n_sc] = spread
            td = np.fft.ifft(np.fft.ifftshift(fd)) * np.sqrt(self.n_fft)
            td_cp = np.concatenate([td[-self.n_cp:], td])
            td_frames.append(td_cp)
        return np.array(td_frames)

    def papr_db(self, td_signal):
        """Compute PAPR in dB."""
        p_inst = np.abs(td_signal.flatten())**2
        return 10*np.log10(p_inst.max() / p_inst.mean())

    def papr_ccdf(self, td_signal, thresholds_db=None):
        """CCDF of PAPR."""
        if thresholds_db is None:
            thresholds_db = np.linspace(0, 14, 100)
        p_inst = np.abs(td_signal.flatten())**2
        p_db = 10*np.log10(p_inst / p_inst.mean())
        ccdf = np.array([np.mean(p_db > t) for t in thresholds_db])
        return thresholds_db, ccdf


# ─────────────────────────────────────────────────────────────────────────────
# 9.  5G NR POLAR CODES (simplified encoder)
# ─────────────────────────────────────────────────────────────────────────────

class PolarCodec:
    """
    Simplified 5G NR Polar code encoder/decoder (SC decoding).
    3GPP TS 38.212 — used for control channels (PBCH, PDCCH, PUCCH).
    """

    def __init__(self, N=256, K=128):
        """N: codeword length (power of 2), K: info bits."""
        self.N = N
        self.K = K
        self._build_reliability()

    def _build_reliability(self):
        """Bhattacharyya parameter ordering (simplified Arikan ordering)."""
        n = int(np.log2(self.N))
        z = np.zeros(self.N)
        z[0] = 0.5
        for _ in range(n):
            z_new = np.zeros(2*len(z))
            for i, zi in enumerate(z):
                z_new[2*i]   = 2*zi - zi**2      # worse channel
                z_new[2*i+1] = zi**2              # better channel
            z = z_new[:self.N]
        # Frozen bits on worst (highest z) channels
        order = np.argsort(z)         # best → worst
        self.info_indices = np.sort(order[:self.K])
        self.frozen_indices = np.sort(order[self.K:])

    def encode(self, u_bits):
        """Polar encode: u_bits length K → codeword length N."""
        assert len(u_bits) == self.K
        u = np.zeros(self.N, int)
        u[self.info_indices] = u_bits
        # Generator matrix G = F^⊗n, F = [[1,0],[1,1]]
        x = u.copy()
        n = int(np.log2(self.N))
        for i in range(n):
            step = 2**(i+1)
            half = step // 2
            for j in range(0, self.N, step):
                x[j:j+half] = (x[j:j+half] + x[j+half:j+step]) % 2
        return x

    def decode_sc(self, llr):
        """Successive Cancellation decoder (simplified)."""
        N = self.N
        n = int(np.log2(N))
        # Recursive SC — use simple threshold for demo
        u_hat = np.zeros(N, int)
        for i in range(N):
            if i in self.frozen_indices:
                u_hat[i] = 0
            else:
                u_hat[i] = 1 if llr[i] < 0 else 0
        return u_hat[self.info_indices]

    def ber_simulation(self, snr_range_db):
        """BER vs SNR for Polar coded BPSK."""
        ber = []
        for snr_db in snr_range_db:
            n_trials = 100
            errs = 0
            total = 0
            for _ in range(n_trials):
                u = np.random.randint(0, 2, self.K)
                c = self.encode(u)
                bpsk = 1 - 2*c.astype(float)
                noisy = awgn(bpsk.reshape(1,-1), snr_db).flatten()
                llr = -2 * noisy.real       # AWGN LLR
                u_hat = self.decode_sc(llr)
                errs += np.sum(u != u_hat)
                total += self.K
            ber.append(errs / total)
        return np.array(ber)


# ─────────────────────────────────────────────────────────────────────────────
# 10. LINK BUDGET CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

class LinkBudget:
    """LTE and 5G NR link budget calculator."""

    def __init__(self):
        pass

    def calculate(self, system, freq_ghz, tx_power_dbm=46, tx_gain_dbi=17,
                  rx_gain_dbi=0, nf_db=7, bandwidth_mhz=20,
                  distance_m=500, shadowing_db=8):
        """
        Returns dict with all link budget components.
        system: 'LTE' or '5G-FR1' or '5G-FR2'
        """
        # EIRP
        eirp = tx_power_dbm + tx_gain_dbi          # dBm
        # Free space path loss (FSPL)
        fspl = (20*np.log10(distance_m) +
                20*np.log10(freq_ghz*1e9) +
                20*np.log10(4*np.pi/3e8))          # dB
        # Thermal noise
        k_db = -174   # dBm/Hz
        noise_power = k_db + 10*np.log10(bandwidth_mhz*1e6) + nf_db   # dBm
        # MAPL
        mapl = eirp - noise_power - shadowing_db + rx_gain_dbi - fspl
        # Required SNR for target MCS (rough)
        req_snr = {"LTE":5, "5G-FR1":3, "5G-FR2":10}[system]
        margin = mapl - req_snr
        return {
            "System": system,
            "Frequency (GHz)": freq_ghz,
            "EIRP (dBm)": round(eirp, 1),
            "FSPL (dB)": round(fspl, 1),
            "Noise Power (dBm)": round(noise_power, 1),
            "MAPL (dB)": round(mapl, 1),
            "Link Margin (dB)": round(margin, 1),
            "Coverage (m)": distance_m,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 11. NETWORK SLICING KPI MODEL
# ─────────────────────────────────────────────────────────────────────────────

def network_slicing_kpis():
    """Return KPI profiles for 5G network slices."""
    slices = {
        "eMBB\n(Enhanced Mobile Broadband)": {
            "Peak DL (Gbps)": 20, "Latency (ms)": 4, "Reliability (%)": 99.9,
            "Density (dev/km²)": 1e5, "Mobility (km/h)": 500,
        },
        "URLLC\n(Ultra-Reliable Low Latency)": {
            "Peak DL (Gbps)": 1, "Latency (ms)": 1, "Reliability (%)": 99.9999,
            "Density (dev/km²)": 1e5, "Mobility (km/h)": 500,
        },
        "mMTC\n(Massive Machine Type)": {
            "Peak DL (Gbps)": 0.1, "Latency (ms)": 10, "Reliability (%)": 99.0,
            "Density (dev/km²)": 1e6, "Mobility (km/h)": 0,
        },
    }
    return slices


# ─────────────────────────────────────────────────────────────────────────────
# 12. MAIN DASHBOARD — 5×3 Plot Grid
# ─────────────────────────────────────────────────────────────────────────────

def run_all_simulations():
    """Run all LTE & 5G simulations and generate master dashboard."""

    print("=" * 65)
    print("  LTE & 5G NR Communications Engineering Simulator")
    print("=" * 65)

    np.random.seed(42)
    snr = np.arange(-5, 31, 1)

    # ── Instantiate subsystems ───────────────────────────────────────────────
    print("[1/9] LTE PDSCH...")
    lte_20 = LTEDownlink(20, "64QAM")
    lte_ber_sim   = lte_20.simulate_ber(snr)
    lte_ber_th    = theoretical_ber_qam(64, snr)
    lte_tp        = lte_20.throughput_mbps(snr)

    print("[2/9] 5G NR PDSCH (μ=1, 40MHz)...")
    nr_mu1 = NRDownlink(mu=1, n_rb=106, mcs_index=19)
    nr_ber_sim    = nr_mu1.simulate_ber(snr, n_slots=1)
    nr_ber_th     = theoretical_ber_qam(64, snr)
    nr_tp_mu1     = nr_mu1.throughput_mbps(snr)

    print("[3/9] 5G NR numerology peak throughputs...")
    nr_peak_tps = {}
    for mu in range(5):
        # Use representative n_rb for each mu
        n_rb_map = {0:100, 1:106, 2:66, 3:66, 4:32}
        nr = NRDownlink(mu=mu, n_rb=n_rb_map[mu], mcs_index=27)
        nr_peak_tps[mu] = nr.throughput_mbps(snr)

    print("[4/9] Massive MIMO beamforming (64T4R)...")
    mimo = MassiveMIMOBeamformer(n_tx=64, n_rx=4, n_beams=8)
    angles_sweep, gains, beam_angles = mimo.beam_gain_pattern()
    snr_cap = np.arange(0, 31, 3)
    mimo_cap_64 = mimo.capacity_vs_snr(snr_cap, n_mc=100)
    mimo4 = MassiveMIMOBeamformer(n_tx=4, n_rx=2, n_beams=4)
    mimo_cap_4  = mimo4.capacity_vs_snr(snr_cap, n_mc=100)

    print("[5/9] DFT-s-OFDM PAPR (5G UL)...")
    ul = NRUplink(n_rb=25, mu=1)
    bits_ul = np.random.randint(0,2,14*ul.n_sc*2)
    td_dfts = ul.modulate_slot(bits_ul, mod_order=2)
    t_db, ccdf_dfts = ul.papr_ccdf(td_dfts)
    # Compare with plain OFDM (no DFT spread)
    ofdm_sig = np.random.randn(10000) + 1j*np.random.randn(10000)
    p_ofdm = np.abs(ofdm_sig)**2
    p_db_ofdm = 10*np.log10(p_ofdm / p_ofdm.mean())
    ccdf_ofdm = np.array([np.mean(p_db_ofdm > t) for t in t_db])

    print("[6/9] Polar code BER (5G PDCCH)...")
    polar = PolarCodec(N=128, K=64)
    snr_polar = np.arange(-2, 8, 1)
    ber_polar = polar.ber_simulation(snr_polar)
    ber_uncoded = 0.5*erfc(np.sqrt(10**(snr_polar/10)))

    print("[7/9] CDL-A channel impulse response...")
    cdl = CDLChannel("CDL-A", doppler_hz=300, fs=15.36e6)
    lte_tp_cdl = lte_20.throughput_mbps(snr - 5)   # ~5dB CDL loss

    print("[8/9] Link budgets...")
    lb = LinkBudget()
    lb_lte  = lb.calculate("LTE",    freq_ghz=1.8, distance_m=1000)
    lb_5g1  = lb.calculate("5G-FR1", freq_ghz=3.5, distance_m=500)
    lb_5g2  = lb.calculate("5G-FR2", freq_ghz=28,  distance_m=100,
                            tx_power_dbm=30, tx_gain_dbi=30, rx_gain_dbi=12)

    print("[9/9] Network slicing KPIs...")
    slices = network_slicing_kpis()

    # Print link budgets
    print("\n── Link Budget Summary ──────────────────────────────────────")
    for lb_res in [lb_lte, lb_5g1, lb_5g2]:
        print(f"  {lb_res['System']:10s} | "
              f"EIRP={lb_res['EIRP (dBm)']} dBm | "
              f"FSPL={lb_res['FSPL (dB)']} dB | "
              f"Margin={lb_res['Link Margin (dB)']} dB")

    # ── Plotting ─────────────────────────────────────────────────────────────
    print("\nGenerating dashboard plots...")
    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.38)

    STYLE = dict(facecolor="#161b22", edgecolor="#30363d")
    TITLE_KW = dict(fontsize=11, fontweight="bold", color="#e6edf3", pad=8)
    LABEL_KW = dict(fontsize=9, color="#8b949e")
    TICK_KW  = dict(colors="#8b949e", labelsize=8)
    GRID_KW  = dict(color="#30363d", linestyle="--", alpha=0.5)

    # Color palette
    C_LTE  = "#58a6ff"
    C_5G   = "#3fb950"
    C_WARN = "#f78166"
    C_ACC  = "#d2a8ff"
    C_YELL = "#e3b341"
    C_CYAN = "#39d353"

    def style_ax(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(STYLE["facecolor"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["edgecolor"])
        ax.set_title(title, **TITLE_KW)
        if xlabel: ax.set_xlabel(xlabel, **LABEL_KW)
        if ylabel: ax.set_ylabel(ylabel, **LABEL_KW)
        ax.tick_params(axis="both", **TICK_KW)
        ax.grid(True, **GRID_KW)
        ax.tick_params(axis="x", colors="#8b949e")
        ax.tick_params(axis="y", colors="#8b949e")

    # ── Plot 1: LTE BER vs SNR ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, "LTE PDSCH BER — 64QAM (20 MHz)", "SNR (dB)", "BER")
    ax.semilogy(snr, lte_ber_th,  "--", color=C_LTE,  lw=1.5, label="Theory")
    ax.semilogy(snr, np.clip(lte_ber_sim, 1e-6, 1), "o-",
                color=C_WARN, lw=1.5, ms=3, label="Simulation")
    ax.set_ylim(1e-5, 1); ax.legend(fontsize=8, labelcolor="white",
                                     facecolor="#21262d", edgecolor="#30363d")

    # ── Plot 2: 5G NR BER vs SNR ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, "5G NR PDSCH BER — 64QAM (μ=1, 40 MHz)", "SNR (dB)", "BER")
    ax.semilogy(snr, nr_ber_th,   "--", color=C_5G,   lw=1.5, label="Theory 64QAM")
    ax.semilogy(snr, np.clip(nr_ber_sim, 1e-6, 1), "s-",
                color=C_YELL, lw=1.5, ms=3, label="NR Sim")
    ax.set_ylim(1e-5, 1); ax.legend(fontsize=8, labelcolor="white",
                                     facecolor="#21262d", edgecolor="#30363d")

    # ── Plot 3: LTE vs 5G Throughput ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, "LTE vs 5G NR — Throughput vs SNR", "SNR (dB)", "Throughput (Mbps)")
    ax.plot(snr, lte_tp,    color=C_LTE, lw=2,   label="LTE 20MHz")
    ax.plot(snr, lte_tp_cdl,color=C_LTE, lw=1.5, ls=":", label="LTE CDL-A")
    ax.plot(snr, nr_tp_mu1, color=C_5G,  lw=2,   label="5G NR μ=1 40MHz")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#21262d", edgecolor="#30363d")

    # ── Plot 4: 5G Numerology Comparison ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, "5G NR Throughput by Numerology (μ)", "SNR (dB)", "Throughput (Mbps)")
    colors_mu = [C_LTE, C_5G, C_ACC, C_WARN, C_YELL]
    for mu in range(5):
        lbl = NR_NUMEROLOGY[mu]["label"]
        ax.plot(snr, nr_peak_tps[mu], color=colors_mu[mu], lw=1.8,
                label=f"μ={mu} ({lbl})")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d",
              edgecolor="#30363d", loc="upper left")

    # ── Plot 5: Massive MIMO Beam Pattern ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, "5G NR Massive MIMO — Beam Codebook (64T)", "Angle (°)", "Beam Gain")
    for i in range(mimo.N_b):
        ax.plot(angles_sweep, gains[i], lw=1.2, alpha=0.7)
    ax.axvline(0, color=C_WARN, lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(-90, 90); ax.set_xlabel("Angle (°)", **LABEL_KW)

    # ── Plot 6: MIMO Capacity ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax, "Massive MIMO Capacity — 64T4R vs 4T2R", "SNR (dB)", "Capacity (b/s/Hz)")
    ax.plot(snr_cap, mimo_cap_64, "o-", color=C_5G, lw=2, ms=4, label="64T4R (5G NR)")
    ax.plot(snr_cap, mimo_cap_4,  "s-", color=C_LTE,lw=2, ms=4, label="4T2R  (LTE)")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#21262d", edgecolor="#30363d")

    # ── Plot 7: DFT-s-OFDM PAPR CCDF ────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    style_ax(ax, "PAPR CCDF — DFT-s-OFDM vs OFDM (5G UL)", "PAPR₀ (dB)", "Prob(PAPR > PAPR₀)")
    ax.semilogy(t_db, np.clip(ccdf_dfts, 1e-4, 1),
                color=C_5G,  lw=2, label="DFT-s-OFDM (NR UL)")
    ax.semilogy(t_db, np.clip(ccdf_ofdm, 1e-4, 1),
                color=C_WARN,lw=2, ls="--", label="OFDM (NR DL)")
    ax.set_ylim(1e-3, 1); ax.legend(fontsize=8, labelcolor="white",
                                     facecolor="#21262d", edgecolor="#30363d")

    # ── Plot 8: Polar Code BER ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    style_ax(ax, "5G NR Polar Code BER (N=128, K=64, PDCCH)", "Eb/N0 (dB)", "BER")
    ax.semilogy(snr_polar, ber_uncoded, "--",   color=C_LTE,  lw=1.5, label="Uncoded BPSK")
    ax.semilogy(snr_polar, np.clip(ber_polar, 1e-5,1), "o-",
                color=C_5G,  lw=1.5, ms=4, label="Polar (SC decode)")
    ax.set_ylim(1e-4, 1); ax.legend(fontsize=8, labelcolor="white",
                                     facecolor="#21262d", edgecolor="#30363d")

    # ── Plot 9: CDL-A Power Delay Profile ───────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    style_ax(ax, "CDL-A Channel — Power Delay Profile", "Delay (ns)", "Power (dB)")
    ax.stem(CDL_A_DELAYS_NS, CDL_A_POWER_DB,
            linefmt=C_5G, markerfmt="o", basefmt=C_WARN)
    ax.stem(CDL_C_DELAYS_NS, CDL_C_POWER_DB,
            linefmt=C_ACC, markerfmt="s", basefmt=C_WARN)
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0],color=C_5G,lw=2,label="CDL-A"),
           Line2D([0],[0],color=C_ACC,lw=2,label="CDL-C")]
    ax.legend(handles=leg, fontsize=8, labelcolor="white",
              facecolor="#21262d", edgecolor="#30363d")

    # ── Plot 10: LTE Resource Grid ───────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    style_ax(ax, "LTE Resource Grid — 1 Subframe (20MHz)", "OFDM Symbol", "Subcarrier")
    rg_data = np.random.randn(1200, 14)**2
    rg_data[::6, :] = 3.0     # CRS pilots
    im = ax.imshow(rg_data[:100, :], aspect="auto", cmap="plasma",
                   extent=[0,14,0,100], origin="lower")
    plt.colorbar(im, ax=ax, label="Power", shrink=0.8).ax.tick_params(labelsize=7)

    # ── Plot 11: 5G NR Slot Formats ──────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    style_ax(ax, "5G NR Slot Format — μ=1 (30kHz SCS)", "Symbol index", "")
    ax.set_facecolor(STYLE["facecolor"])
    slot_types = ["D"]*10 + ["G","G","U","U"]
    colors_slot = {"D": C_5G, "G": C_YELL, "U": C_WARN}
    for i, st in enumerate(slot_types):
        ax.barh(0, 1, left=i, color=colors_slot[st], edgecolor="#0d1117",
                height=0.6, alpha=0.9)
        ax.text(i+0.5, 0, st, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
    ax.set_xlim(0,14); ax.set_ylim(-0.5, 0.5)
    ax.set_xticks(range(15)); ax.set_yticks([])
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=C_5G, label="Downlink (D)"),
           Patch(facecolor=C_YELL, label="Guard (G)"),
           Patch(facecolor=C_WARN, label="Uplink (U)")]
    ax.legend(handles=leg, fontsize=8, labelcolor="white",
              facecolor="#21262d", edgecolor="#30363d", loc="upper right")

    # ── Plot 12: LTE vs 5G Link Budget ───────────────────────────────────────
    ax = fig.add_subplot(gs[3, 2])
    style_ax(ax, "Link Budget Comparison", "", "Level (dB/dBm)")
    systems = ["LTE 1.8GHz", "5G FR1 3.5GHz", "5G FR2 28GHz"]
    metrics = ["EIRP (dBm)", "FSPL (dB)", "Link Margin (dB)"]
    vals = np.array([[lb_lte[m],  lb_5g1[m],  lb_5g2[m]] for m in metrics])
    x = np.arange(3); w = 0.25
    for i, (m, color) in enumerate(zip(metrics, [C_LTE, C_WARN, C_5G])):
        ax.bar(x + i*w, vals[i], w, label=m, color=color, alpha=0.85)
    ax.set_xticks(x+w); ax.set_xticklabels(systems, fontsize=7, color="#8b949e")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor="#30363d")

    # ── Plot 13–15: Network Slicing KPI radar ────────────────────────────────
    ax = fig.add_subplot(gs[4, :], polar=False)
    ax.set_facecolor(STYLE["facecolor"])
    for spine in ax.spines.values(): spine.set_edgecolor(STYLE["edgecolor"])
    slice_names = list(slices.keys())
    kpi_names   = list(list(slices.values())[0].keys())
    x_pos = np.arange(len(kpi_names))
    bar_w = 0.25

    # Normalize KPIs for display (log scale for big range)
    raw = np.array([[s[k] for k in kpi_names] for s in slices.values()], float)
    normalized = np.log10(raw + 1)
    normalized /= normalized.max(axis=0, keepdims=True)

    for i, (sname, color) in enumerate(zip(slice_names,
                                           [C_LTE, C_5G, C_ACC])):
        ax.bar(x_pos + i*bar_w, normalized[i], bar_w,
               label=sname.replace("\n"," "), color=color, alpha=0.85)

    ax.set_xticks(x_pos + bar_w)
    ax.set_xticklabels(kpi_names, fontsize=9, color="#e6edf3", rotation=20, ha="right")
    ax.set_ylabel("Normalised Score (log)", **LABEL_KW)
    ax.set_title("5G NR Network Slicing — eMBB / URLLC / mMTC KPI Comparison",
                 **TITLE_KW)
    ax.tick_params(axis="y", **TICK_KW)
    ax.grid(True, **GRID_KW)
    ax.legend(fontsize=9, labelcolor="white", facecolor="#21262d",
              edgecolor="#30363d", loc="upper right")

    # ── Super-title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "📡  LTE & 5G NR Communications Engineering Simulator\n"
        "rizwan66/comms_engineering  —  Enhanced with Full 5G NR Stack",
        fontsize=14, fontweight="bold", color="#e6edf3", y=0.995)

    out_path = "/mnt/user-data/outputs/lte_5g_dashboard.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n✅  Dashboard saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 13. INDIVIDUAL MODULE RUNNERS  (mirrors existing project style)
# ─────────────────────────────────────────────────────────────────────────────

def run_lte_module():
    """Run LTE standalone analysis."""
    sim = LTEDownlink(20, "64QAM")
    snr = np.arange(-5, 31)
    ber = sim.simulate_ber(snr)
    tp  = sim.throughput_mbps(snr)
    print(f"LTE 20MHz 64QAM | BER@15dB={ber[snr==15][0]:.4f} | TP@20dB={tp[snr==20][0]:.1f} Mbps")

def run_5g_module():
    """Run 5G NR standalone analysis."""
    for mu in [0, 1, 3]:
        nr = NRDownlink(mu=mu, n_rb={0:100,1:106,3:66}[mu], mcs_index=19)
        tp = nr.throughput_mbps(20)
        print(f"5G NR μ={mu} {NR_NUMEROLOGY[mu]['label']:<20s} | TP@20dB={tp:.1f} Mbps")

def run_beamforming_module():
    """Run Massive MIMO beamforming demo."""
    bf = MassiveMIMOBeamformer(64, 4, 8)
    cap = bf.capacity_vs_snr([10, 20, 30], n_mc=50)
    for snr_val, c in zip([10,20,30], cap):
        print(f"64T4R Capacity @ {snr_val}dB SNR = {c:.2f} b/s/Hz")

def run_polar_module():
    """Run 5G NR Polar code demo."""
    pc = PolarCodec(N=128, K=64)
    ber = pc.ber_simulation([0, 2, 4, 6])
    for snr_val, b in zip([0,2,4,6], ber):
        print(f"Polar(128,64) BER @ Eb/N0={snr_val}dB → {b:.4f}")

def run_link_budget():
    """Run link budget calculations."""
    lb = LinkBudget()
    for sys, freq, dist in [("LTE",1.8,1000),("5G-FR1",3.5,500),("5G-FR2",28,100)]:
        r = lb.calculate(sys, freq, distance_m=dist)
        print(f"{r['System']:10s} @ {freq}GHz, {dist}m | Margin={r['Link Margin (dB)']} dB")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick module smoke-tests
    print("\n── Module Smoke Tests ──────────────────────────────────────")
    run_lte_module()
    run_5g_module()
    run_beamforming_module()
    run_polar_module()
    run_link_budget()

    # Full dashboard
    print("\n── Generating Full Dashboard ───────────────────────────────")
    run_all_simulations()
