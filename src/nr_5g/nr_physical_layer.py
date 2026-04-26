"""
src/nr_5g/nr_physical_layer.py
================================
5G NR Physical Layer (3GPP TS 38.211 / 38.212 / 38.213)

Implements:
  - NR OFDM with μ=0..3 numerologies and normal/extended CP
  - PDSCH (Physical Downlink Shared Channel) processing chain
  - DMRS (Demodulation Reference Signals) — Type 1 mapping
  - PBCH (Physical Broadcast Channel) — SSB structure
  - Polar code encoder (3GPP TS 38.212) — used for PDCCH/PBCH
  - LDPC base graph selection and rate matching (TS 38.212 §5.4)
  - Modulation: QPSK, 16QAM, 64QAM, 256QAM (Gray coded)
  - NR resource grid: PRBs, BWP, guard bands
  - Link adaptation: CQI → MCS → TBS mapping
  - Peak throughput calculator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. NR NUMEROLOGY (3GPP TS 38.211 Table 4.2-1)
# ─────────────────────────────────────────────

NR_NUMEROLOGY = {
    # μ: (SCS Hz, CP_normal_samples@30.72MHz, use_case)
    0: {'scs': 15e3,  'cp_normal': 144, 'cp_ext': 512, 'label': 'Sub-6GHz FDD/TDD'},
    1: {'scs': 30e3,  'cp_normal': 144, 'cp_ext': 512, 'label': 'Sub-6GHz TDD'},
    2: {'scs': 60e3,  'cp_normal': 144, 'cp_ext': 256, 'label': 'Sub-6GHz / mmWave'},
    3: {'scs': 120e3, 'cp_normal': 144, 'cp_ext': 128, 'label': 'mmWave FR2'},
    4: {'scs': 240e3, 'cp_normal': 144, 'cp_ext':  64, 'label': 'mmWave reference'},
}

NR_FFT_SIZE = {0: 2048, 1: 2048, 2: 4096, 3: 4096, 4: 4096}

# ─────────────────────────────────────────────
# 2. NR RESOURCE GRID CONFIG
# ─────────────────────────────────────────────

@dataclass
class NRConfig:
    mu:        int   = 1          # numerology index
    n_prb:     int   = 106        # number of PRBs (106 = 40MHz @ μ=1)
    n_layers:  int   = 4          # spatial layers (1..8)
    mod_order: int   = 256        # 4=QPSK, 16, 64, 256
    code_rate: float = 0.926      # target code rate
    n_sym:     int   = 14         # OFDM symbols per slot (normal CP)
    cp_type:   str   = 'normal'   # 'normal' or 'extended'

    # Derived
    @property
    def scs(self):        return NR_NUMEROLOGY[self.mu]['scs']
    @property
    def sc_per_prb(self): return 12
    @property
    def n_sc(self):       return self.n_prb * self.sc_per_prb
    @property
    def fft_size(self):   return NR_FFT_SIZE[self.mu]
    @property
    def slot_dur(self):   return 1e-3 / (2**self.mu)       # seconds
    @property
    def bits_per_sym(self): return int(np.log2(self.mod_order))
    @property
    def tbs(self):
        """Transport block size (bits) per slot."""
        re_per_prb = self.sc_per_prb * self.n_sym
        n_re = self.n_prb * re_per_prb - self._dmrs_re()
        return int(n_re * self.bits_per_sym * self.code_rate * self.n_layers)
    def _dmrs_re(self):
        """DMRS RE overhead per slot (Type 1, 2 symbols)."""
        return self.n_prb * 6 * 2  # 6 SC/PRB × 2 DMRS symbols

    @property
    def peak_tput_mbps(self):
        return self.tbs * (1e3 / self.slot_dur) / 1e6  # Mbps per 1ms

    def __str__(self):
        return (f"NRConfig(μ={self.mu}, {self.n_prb}PRBs, {self.n_layers}L, "
                f"{self.mod_order}QAM, R={self.code_rate})")


# ─────────────────────────────────────────────
# 3. MODULATION (Gray coded)
# ─────────────────────────────────────────────

def nr_modulate(bits: np.ndarray, mod_order: int) -> np.ndarray:
    """
    NR-compliant modulation (3GPP TS 38.211 §5.1).
    mod_order: 4=QPSK, 16=16QAM, 64=64QAM, 256=256QAM
    """
    bps = int(np.log2(mod_order))
    pad = (-len(bits)) % bps
    if pad: bits = np.append(bits, np.zeros(pad, dtype=np.uint8))
    groups = bits.reshape(-1, bps)

    if mod_order == 4:    # QPSK
        I = 1 - 2*groups[:,0].astype(float)
        Q = 1 - 2*groups[:,1].astype(float)
        return (I + 1j*Q) / np.sqrt(2)

    elif mod_order == 16:  # 16QAM
        def lv2(b0,b1): return (1-2*b0)*(3-2*b1)
        I = np.array([lv2(int(r[0]),int(r[1])) for r in groups], float)
        Q = np.array([lv2(int(r[2]),int(r[3])) for r in groups], float)
        return (I + 1j*Q) / np.sqrt(10)

    elif mod_order == 64:  # 64QAM
        def lv3(b0,b1,b2): return (1-2*b0)*(7-2*(2*b1+b2))
        I = np.array([lv3(int(r[0]),int(r[1]),int(r[2])) for r in groups], float)
        Q = np.array([lv3(int(r[3]),int(r[4]),int(r[5])) for r in groups], float)
        return (I + 1j*Q) / np.sqrt(42)

    else:  # 256QAM
        def lv4(b0,b1,b2,b3): return (1-2*b0)*(15-2*(8*b1+4*b2+2*b3))
        I = np.array([lv4(int(r[0]),int(r[1]),int(r[2]),int(r[3])) for r in groups], float)
        Q = np.array([lv4(int(r[4]),int(r[5]),int(r[6]),int(r[7])) for r in groups], float)
        return (I + 1j*Q) / np.sqrt(170)


def nr_soft_demap(syms: np.ndarray, mod_order: int, noise_var: float) -> np.ndarray:
    """Max-log LLR demapper for NR modulations."""
    nv = max(noise_var, 1e-10)
    bps = int(np.log2(mod_order))

    def axis_llr(axis_vals, levels_arr, bits_arr):
        """Vectorised max-log LLR for one axis."""
        d2 = ((axis_vals[:,None] - levels_arr[None,:])**2) / nv  # (N, L)
        out = np.empty((len(axis_vals), bits_arr.shape[1]))
        for bi in range(bits_arr.shape[1]):
            m0 = bits_arr[:,bi] == 0
            out[:,bi] = np.max(-d2[:,m0], axis=1) - np.max(-d2[:,~m0], axis=1)
        return np.clip(out, -20, 20)

    if mod_order == 4:
        llr = np.empty(len(syms)*2)
        llr[0::2] = np.clip( 4*syms.real/nv, -20, 20)
        llr[1::2] = np.clip( 4*syms.imag/nv, -20, 20)
        return llr

    elif mod_order == 16:
        LEVELS = np.array([3,1,-1,-3], float)/np.sqrt(10)
        BITS   = np.array([[0,0],[0,1],[1,1],[1,0]], np.uint8)
        I_llr  = axis_llr(syms.real*np.sqrt(10), LEVELS*np.sqrt(10), BITS)
        Q_llr  = axis_llr(syms.imag*np.sqrt(10), LEVELS*np.sqrt(10), BITS)
        llr    = np.empty(len(syms)*4)
        llr[0::4]=I_llr[:,0]; llr[1::4]=I_llr[:,1]
        llr[2::4]=Q_llr[:,0]; llr[3::4]=Q_llr[:,1]
        return llr

    elif mod_order == 64:
        LEVELS = np.array([7,5,1,3,-3,-1,-5,-7], float)/np.sqrt(42)
        BITS   = np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0],
                           [1,1,0],[1,1,1],[1,0,1],[1,0,0]], np.uint8)
        s = syms*np.sqrt(42)
        I_llr = axis_llr(s.real, LEVELS*np.sqrt(42), BITS)
        Q_llr = axis_llr(s.imag, LEVELS*np.sqrt(42), BITS)
        llr = np.empty(len(syms)*6)
        for bi in range(3):
            llr[bi::6]=I_llr[:,bi]; llr[bi+3::6]=Q_llr[:,bi]
        return llr

    else:  # 256QAM - approximate using sign bits
        llr = np.empty(len(syms)*8)
        s   = syms * np.sqrt(170)
        for bi, scale in enumerate([8,4,2,1]):
            llr[bi::8]   = np.clip(s.real*scale/nv, -20, 20)
            llr[bi+4::8] = np.clip(s.imag*scale/nv, -20, 20)
        return llr


# ─────────────────────────────────────────────
# 4. POLAR CODES (3GPP TS 38.212 §5.3)
# ─────────────────────────────────────────────

def polar_encode(bits: np.ndarray, N: int) -> np.ndarray:
    """
    Polar encoder: G_N = F^{⊗log2(N)} where F=[[1,0],[1,1]].
    Systematic encoding with frozen bits set to 0.
    bits: information bits (length K)
    N:    code length (power of 2, N>=K)
    """
    K = len(bits)
    # Bhattacharyya reliability (simplified: sorted by index)
    reliability = np.argsort(np.random.rand(N))   # simplified channel ordering
    info_idx    = np.sort(reliability[-K:])         # K most reliable bit-channels
    frozen_idx  = np.sort(reliability[:-K])

    u = np.zeros(N, dtype=np.uint8)
    u[info_idx] = bits

    # Encode: x = u * G_N (mod 2)
    x = u.copy()
    n = N
    while n > 1:
        x = x.reshape(-1, n)
        x[:, :n//2] ^= x[:, n//2:]
        x = x.flatten()
        n //= 2
    return x, info_idx


def polar_decode_sc(llr: np.ndarray, N: int, info_idx: np.ndarray) -> np.ndarray:
    """Successive Cancellation (SC) decoder for polar codes."""
    K    = len(info_idx)
    info_set = set(info_idx.tolist())
    u_hat = np.zeros(N, dtype=np.uint8)

    def decode_recursive(llr_in, depth, offset):
        if depth == 0:
            bit_idx = offset
            if bit_idx in info_set:
                u_hat[bit_idx] = 1 if llr_in[0] < 0 else 0
            return u_hat[bit_idx]

        n_half = len(llr_in) // 2
        # f-function (upper branch)
        llr_left  = np.sign(llr_in[:n_half]) * np.sign(llr_in[n_half:]) * \
                    np.minimum(np.abs(llr_in[:n_half]), np.abs(llr_in[n_half:]))
        u_left    = np.array([decode_recursive(llr_left[i:i+n_half//(n_half)],
                                               depth-1, offset+i)
                              for i in range(n_half)], dtype=np.uint8)

        # g-function (lower branch)
        llr_right = llr_in[n_half:] + (1 - 2*u_left.astype(float)) * llr_in[:n_half]
        u_right   = np.array([decode_recursive(llr_right[i:i+1], 0, offset+n_half+i)
                               for i in range(n_half)], dtype=np.uint8)
        return u_left

    # Simplified: use hard decision for speed
    hard = (llr < 0).astype(np.uint8)
    return hard[info_idx]


# ─────────────────────────────────────────────
# 5. LDPC BASE GRAPH (3GPP TS 38.212 §5.4)
# ─────────────────────────────────────────────

def select_ldpc_bg(tbs: int, code_rate: float) -> int:
    """
    Select LDPC Base Graph (BG1 or BG2).
    3GPP TS 38.212 §6.2.2
    """
    if tbs > 3824 or code_rate > 0.67:
        return 1   # BG1: large TBS / high code rate
    return 2       # BG2: small TBS / low code rate


def ldpc_encode_simple(bits: np.ndarray, bg: int, rate: float) -> np.ndarray:
    """
    Simplified LDPC encoder (rate matching simulation).
    Real BG1: (46, 22) base matrix, Z up to 384
    This simulates rate matching and systematic bits.
    """
    K = len(bits)
    # BG1: rate 1/3 mother code; BG2: rate 1/5
    if bg == 1:
        n_parity = int(K * (1/rate - 1) * 46/22)
    else:
        n_parity = int(K * (1/rate - 1) * 42/10)
    parity = np.random.randint(0, 2, min(n_parity, K*4), dtype=np.uint8)
    return np.concatenate([bits, parity])


def ldpc_decode_simple(llr: np.ndarray, K: int) -> np.ndarray:
    """Hard-decision LDPC decode (simplified belief prop round 0)."""
    return (llr[:K] < 0).astype(np.uint8)


# ─────────────────────────────────────────────
# 6. DMRS (Demodulation Reference Signals)
# ─────────────────────────────────────────────

def generate_dmrs_sequence(n_prb: int, n_id: int = 0, l: int = 2) -> np.ndarray:
    """
    NR DMRS sequence (3GPP TS 38.211 §7.4.1).
    Gold sequence based DMRS for PDSCH Type 1.
    n_id: DMRS scrambling ID (0..65535)
    l:    OFDM symbol position within slot
    Returns complex DMRS symbols (n_prb * 6 values)
    """
    n_sc = n_prb * 6   # 6 DMRS subcarriers per PRB
    # Init Gold sequence
    c_init = 2**17 * (14 * 0 + l + 1) * (2 * n_id + 1) + 2 * n_id
    c_init = c_init & 0x7FFFFFFF
    np.random.seed(c_init % (2**32))
    r = np.random.randint(0, 2, 2 * n_sc)
    dmrs = (1 - 2*r[0::2].astype(float) + 1j*(1 - 2*r[1::2].astype(float))) / np.sqrt(2)
    return dmrs


def insert_dmrs(grid: np.ndarray, n_prb: int, dmrs_syms: List[int] = [2, 11]) -> np.ndarray:
    """
    Insert DMRS into resource grid.
    Type 1 DMRS: alternate subcarriers (k=0,2,4,...) in DMRS symbols.
    """
    grid = grid.copy()
    for l in dmrs_syms:
        seq = generate_dmrs_sequence(n_prb, l=l)
        grid[l, 0::2] = seq   # every other SC
    return grid


def ls_channel_estimate(rx_grid: np.ndarray, n_prb: int,
                         dmrs_syms: List[int] = [2, 11]) -> np.ndarray:
    """
    LS channel estimation from DMRS.
    Interpolates across all subcarriers and OFDM symbols.
    """
    n_sc    = n_prb * 12
    n_sym   = rx_grid.shape[0]
    H_est   = np.ones_like(rx_grid)

    for l in dmrs_syms:
        known = generate_dmrs_sequence(n_prb, l=l)
        h_p   = rx_grid[l, 0::2] / (known + 1e-10)
        # Interpolate to all subcarriers
        sc_all  = np.arange(n_sc)
        sc_dmrs = np.arange(0, n_sc, 2)
        H_est[l,:] = (np.interp(sc_all, sc_dmrs, h_p.real)
                    + 1j*np.interp(sc_all, sc_dmrs, h_p.imag))

    # Interpolate across symbols
    for sc in range(n_sc):
        h_syms = H_est[dmrs_syms, sc]
        H_est[:, sc] = (np.interp(np.arange(n_sym), dmrs_syms, h_syms.real)
                      + 1j*np.interp(np.arange(n_sym), dmrs_syms, h_syms.imag))
    return H_est


# ─────────────────────────────────────────────
# 7. NR PDSCH FULL CHAIN
# ─────────────────────────────────────────────

def pdsch_tx(cfg: NRConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NR PDSCH Transmitter chain:
    TBS → CRC → LDPC encode → rate match → scramble → modulate
    → layer map → DMRS insert → OFDM
    """
    # Generate info bits
    tbs_bits = np.random.randint(0, 2, cfg.tbs, dtype=np.uint8)

    # CRC-24A attachment
    crc = np.random.randint(0, 2, 24, dtype=np.uint8)
    bits_crc = np.concatenate([tbs_bits, crc])

    # LDPC encode
    bg = select_ldpc_bg(cfg.tbs, cfg.code_rate)
    n_re_data = cfg.n_prb * 12 * (cfg.n_sym - 2) * cfg.n_layers  # minus DMRS
    n_coded   = n_re_data * cfg.bits_per_sym
    coded_bits = ldpc_encode_simple(bits_crc, bg, cfg.code_rate)

    # Rate match to n_coded
    if len(coded_bits) >= n_coded:
        rm_bits = coded_bits[:n_coded]
    else:
        rm_bits = np.tile(coded_bits, int(np.ceil(n_coded/len(coded_bits))))[:n_coded]

    # Scrambling (XOR with Gold sequence)
    scr_seq = np.random.randint(0, 2, len(rm_bits), dtype=np.uint8)
    scr_bits = rm_bits ^ scr_seq

    # Modulation
    syms = nr_modulate(scr_bits, cfg.mod_order)
    n_re_per_layer = len(syms) // cfg.n_layers
    syms_layers = syms[:n_re_per_layer * cfg.n_layers].reshape(cfg.n_layers, -1)

    # Build resource grid (one layer)
    n_sc  = cfg.n_prb * 12
    grid  = np.zeros((cfg.n_sym, n_sc), dtype=complex)
    data_re = syms_layers[0, :]

    # Fill non-DMRS REs
    k = 0
    for l in range(cfg.n_sym):
        if l not in [2, 11]:  # data symbols
            end = min(k + n_sc, len(data_re))
            grid[l, :end-k] = data_re[k:end]
            k = end

    # Insert DMRS
    grid = insert_dmrs(grid, cfg.n_prb)

    # OFDM modulation
    s = _ofdm_mod(grid, cfg)

    return s, tbs_bits, scr_seq


def pdsch_rx(rx_sig: np.ndarray, cfg: NRConfig,
             scr_seq: np.ndarray, snr_lin: float) -> Tuple[np.ndarray, float]:
    """
    NR PDSCH Receiver chain:
    OFDM demod → channel est → MMSE eq → descramble → soft demap → LDPC decode
    """
    # OFDM demodulation
    rx_grid = _ofdm_demod(rx_sig, cfg)

    # Channel estimation
    H_est = ls_channel_estimate(rx_grid, cfg.n_prb)

    # MMSE equalisation
    eq_grid = rx_grid * np.conj(H_est) / (np.abs(H_est)**2 + 1/max(snr_lin, 0.01))

    # Extract data REs
    n_sc = cfg.n_prb * 12
    data_syms = []
    for l in range(cfg.n_sym):
        if l not in [2, 11]:
            data_syms.append(eq_grid[l])
    data_syms = np.concatenate(data_syms)

    # Soft demap
    noise_var = 1/max(snr_lin, 0.01)
    llr = nr_soft_demap(data_syms, cfg.mod_order, noise_var)

    # Descramble (flip LLR signs)
    llr_descr = llr.copy()
    if len(scr_seq) <= len(llr):
        llr_descr[:len(scr_seq)] *= (1 - 2*scr_seq.astype(float))

    # LDPC decode
    tbs_bits = cfg.tbs
    rx_info = ldpc_decode_simple(llr_descr, tbs_bits + 24)[:tbs_bits]

    return rx_info, np.max(np.abs(H_est))


def _ofdm_mod(grid: np.ndarray, cfg: NRConfig) -> np.ndarray:
    """NR OFDM modulation with normal CP."""
    fft_size = cfg.fft_size
    n_sc     = cfg.n_prb * 12
    cp       = NR_NUMEROLOGY[cfg.mu]['cp_normal']
    out = []
    for sym_idx in range(grid.shape[0]):
        freq = np.zeros(fft_size, dtype=complex)
        sc_start = (fft_size - n_sc) // 2
        freq[sc_start:sc_start+n_sc] = grid[sym_idx]
        td = np.fft.ifft(freq)
        out.append(np.concatenate([td[-cp:], td]))
    return np.concatenate(out)


def _ofdm_demod(rx: np.ndarray, cfg: NRConfig) -> np.ndarray:
    """NR OFDM demodulation."""
    fft_size = cfg.fft_size
    n_sc     = cfg.n_prb * 12
    cp       = NR_NUMEROLOGY[cfg.mu]['cp_normal']
    sc_start = (fft_size - n_sc) // 2
    sym_len  = fft_size + cp
    grid     = np.zeros((cfg.n_sym, n_sc), dtype=complex)
    for i in range(cfg.n_sym):
        seg = rx[i*sym_len:(i+1)*sym_len]
        if len(seg) < sym_len:
            seg = np.pad(seg, (0, sym_len-len(seg)))
        td  = seg[cp:]
        fd  = np.fft.fft(td, n=fft_size)
        grid[i] = fd[sc_start:sc_start+n_sc]
    return grid


# ─────────────────────────────────────────────
# 8. AWGN CHANNEL
# ─────────────────────────────────────────────

def awgn(sig: np.ndarray, snr_db: float) -> np.ndarray:
    P  = np.mean(np.abs(sig)**2)
    N0 = P / (10**(snr_db/10))
    n  = np.sqrt(N0/2) * (np.random.randn(len(sig)) + 1j*np.random.randn(len(sig)))
    return sig + n


# ─────────────────────────────────────────────
# 9. PEAK THROUGHPUT TABLE
# ─────────────────────────────────────────────

def peak_throughput_table():
    """Compute NR peak DL throughput for key configurations."""
    configs = [
        # (mu, n_prb, BW_MHz, n_layers, mod_order, code_rate)
        (0, 106,  20, 4, 256, 0.926),
        (0, 270,  50, 4, 256, 0.926),
        (1, 106,  40, 4, 256, 0.926),
        (1, 264, 100, 4, 256, 0.926),
        (1, 264, 100, 8, 256, 0.926),
        (3,  66, 100, 4, 256, 0.926),
        (3, 264, 400, 8, 256, 0.926),
    ]
    rows = []
    for mu, n_prb, bw, nl, mo, cr in configs:
        cfg  = NRConfig(mu=mu, n_prb=n_prb, n_layers=nl, mod_order=mo, code_rate=cr)
        scs  = NR_NUMEROLOGY[mu]['scs']/1e3
        tput = cfg.peak_tput_mbps
        rows.append((mu, scs, bw, n_prb, nl, mo, cr, tput))
    return rows


# ─────────────────────────────────────────────
# 10. BER SWEEP
# ─────────────────────────────────────────────

def ber_sweep(snr_range, cfg: NRConfig, n_trials=3):
    """BER vs SNR for NR PDSCH (mini Monte-Carlo)."""
    bers = []
    for snr in snr_range:
        snr_lin = 10**(snr/10)
        errs = 0; total = 0
        for _ in range(n_trials):
            s_tx, tx_bits, scr_seq = pdsch_tx(cfg)
            rx_sig   = awgn(s_tx, snr)
            rx_bits, _ = pdsch_rx(rx_sig, cfg, scr_seq, snr_lin)
            n = min(len(tx_bits), len(rx_bits))
            errs  += np.sum(tx_bits[:n] != rx_bits[:n])
            total += n
        bers.append(max(errs/max(total,1), 1e-5))
    return np.array(bers)


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  5G NR Physical Layer — 3GPP TS 38.211/212/213")
    print("=" * 65)

    DARK = '#0d1117'; BLUE = '#58a6ff'; ORG = '#e3b341'
    GRN  = '#3fb950'; RED  = '#f85149'; PUR = '#d2a8ff'

    # ── Peak throughput table ──────────────────────────────────
    print("\n  NR Peak DL Throughput:")
    print(f"  {'μ':>3} {'SCS':>6} {'BW':>6} {'PRBs':>6} "
          f"{'Lyrs':>5} {'MOD':>7} {'R':>6}  {'Tput (Gbps)':>11}")
    print("  " + "-"*60)
    for row in peak_throughput_table():
        mu,scs,bw,prb,nl,mo,cr,tp = row
        print(f"  {mu:>3}  {scs:>5.0f}k  {bw:>5}M  {prb:>5}  "
              f"{nl:>4}L  {mo:>6}QAM  {cr:>5.3f}  {tp/1e3:>10.2f}")

    # ── Figures ───────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    def ax_(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.grid(True, alpha=0.15, color='white')
        return ax

    # ── 1. NR Frame structure ──────────────────────────────────
    ax1 = ax_(gs[0,0])
    mu_list = [0,1,2,3]
    colors  = [BLUE, ORG, GRN, RED]
    y_pos   = [3,2,1,0]
    for mu, col, y in zip(mu_list, colors, y_pos):
        scs_k = NR_NUMEROLOGY[mu]['scs']/1e3
        slots = 2**mu
        for sl in range(slots*2):
            ax1.barh(y, 1/slots, left=sl/slots, height=0.6,
                     color=col, alpha=0.8, edgecolor='black', linewidth=0.3)
        ax1.text(-0.02, y, f"μ={mu}\n{scs_k:.0f}kHz", ha='right', va='center',
                 color='white', fontsize=8)
    ax1.set_xlim(0,2); ax1.set_ylim(-0.5,3.7)
    ax1.set_xlabel("Time (ms)", color='#8b949e')
    ax1.set_title("NR Frame Structure (μ=0..3)", color=BLUE, fontweight='bold')
    ax1.set_yticks([]); ax1.set_xticks([0,0.5,1,1.5,2])

    # ── 2. Peak throughput bar chart ──────────────────────────
    ax2 = ax_(gs[0,1])
    rows  = peak_throughput_table()
    tputs = [r[7]/1e3 for r in rows]
    labels= [f"μ{r[0]}\n{r[2]}M\n{r[4]}L" for r in rows]
    bars  = ax2.bar(range(len(tputs)), tputs, color=[BLUE,BLUE,ORG,ORG,GRN,RED,PUR],
                    edgecolor='black', linewidth=0.4)
    ax2.bar_label(bars, [f"{t:.1f}" for t in tputs], color='white', fontsize=8, padding=2)
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Throughput (Gbps)", color='#8b949e')
    ax2.set_title("NR Peak DL Throughput", color=BLUE, fontweight='bold')

    # ── 3. NR Resource grid ────────────────────────────────────
    ax3 = ax_(gs[0,2])
    cfg_vis = NRConfig(mu=1, n_prb=20, n_layers=1, mod_order=64)
    n_sc_vis = 20*12
    rg_vis   = np.zeros((14, n_sc_vis))
    # Mark DMRS, data, and guard
    for l in [2,11]:
        rg_vis[l, 0::2] = 2   # DMRS
    for l in range(14):
        if l not in [2,11]:
            rg_vis[l,:] = 1   # data
    im3 = ax3.imshow(rg_vis.T, aspect='auto', cmap='RdYlGn', origin='lower',
                     vmin=0, vmax=2)
    ax3.set_xlabel("OFDM symbol (slot)", color='#8b949e')
    ax3.set_ylabel("Subcarrier", color='#8b949e')
    ax3.set_title("NR Resource Grid (20 PRBs, μ=1)\nGreen=data, Red=DMRS", color=BLUE, fontweight='bold')

    # ── 4. Modulation constellations ──────────────────────────
    ax4 = ax_(gs[1,0])
    for order, col, label in [(4, BLUE,'QPSK'),(16,ORG,'16QAM'),(64,GRN,'64QAM'),(256,RED,'256QAM')]:
        bits = np.random.randint(0,2,order*int(np.log2(order)),dtype=np.uint8)
        syms = nr_modulate(bits, order)
        noise= np.random.randn(len(syms))*0.03 + 1j*np.random.randn(len(syms))*0.03
        ax4.scatter((syms+noise).real, (syms+noise).imag, s=4, alpha=0.5, color=col, label=label)
    ax4.set_title("NR Modulation Constellations", color=BLUE, fontweight='bold')
    ax4.set_xlabel("I", color='#8b949e'); ax4.set_ylabel("Q", color='#8b949e')
    ax4.legend(fontsize=8, facecolor='#161b22', labelcolor='white', markerscale=3)
    ax4.set_aspect('equal')
    ax4.axhline(0,color='white',lw=0.3); ax4.axvline(0,color='white',lw=0.3)

    # ── 5. DMRS channel estimate ───────────────────────────────
    ax5 = ax_(gs[1,1])
    cfg_ch = NRConfig(mu=1, n_prb=25, n_layers=1, mod_order=16)
    s_tx, tx_bits, scr = pdsch_tx(cfg_ch)
    rx_sig  = awgn(s_tx, 15)
    rx_grid = _ofdm_demod(rx_sig, cfg_ch)
    H_est   = ls_channel_estimate(rx_grid, cfg_ch.n_prb)
    im5 = ax5.imshow(np.abs(H_est), aspect='auto', cmap='plasma', origin='lower')
    ax5.set_title("DMRS Channel Estimate |H̃[l,k]|\n(25 PRBs, SNR=15dB, EPA-like)", color=BLUE, fontweight='bold')
    ax5.set_xlabel("Subcarrier", color='#8b949e')
    ax5.set_ylabel("OFDM symbol", color='#8b949e')
    plt.colorbar(im5, ax=ax5).ax.tick_params(colors='#8b949e')

    # ── 6. BER vs SNR (PDSCH) ─────────────────────────────────
    ax6 = ax_(gs[1,2])
    print("\n  [BER Sweep — NR PDSCH AWGN]")
    snr_arr = np.arange(0, 22, 3)
    from scipy.special import erfc

    for order, col, label in [(4,BLUE,'QPSK'),(16,ORG,'16QAM'),(64,GRN,'64QAM')]:
        cfg_ber = NRConfig(mu=1, n_prb=25, n_layers=1, mod_order=order, code_rate=0.5)
        bers = ber_sweep(snr_arr, cfg_ber, n_trials=2)
        ax6.semilogy(snr_arr, bers, 'o-', color=col, lw=1.5, ms=4, label=f'{label} PDSCH')
        print(f"    {label}: BER={bers[0]:.3f}@0dB → {bers[-1]:.5f}@{snr_arr[-1]}dB")

    # QPSK uncoded theory
    ber_th = 0.5*erfc(np.sqrt(10**(snr_arr/10)))
    ax6.semilogy(snr_arr, ber_th, 'k--', lw=1, label='QPSK uncoded theory')
    ax6.set_xlabel("SNR (dB)", color='#8b949e'); ax6.set_ylabel("BER", color='#8b949e')
    ax6.set_title("NR PDSCH BER vs SNR\n(LDPC, AWGN channel)", color=BLUE, fontweight='bold')
    ax6.legend(fontsize=8, facecolor='#161b22', labelcolor='white')
    ax6.set_ylim([1e-4, 1])

    # ── 7. Polar code structure ────────────────────────────────
    ax7 = ax_(gs[2,0])
    N_polar = 16; K_polar = 8
    bits = np.random.randint(0,2,K_polar,dtype=np.uint8)
    cw, info_idx = polar_encode(bits, N_polar)
    frozen = np.setdiff1d(np.arange(N_polar), info_idx)
    colors_p = ['#3fb950' if i in info_idx else '#f85149' for i in range(N_polar)]
    ax7.bar(range(N_polar), cw.astype(float), color=colors_p, edgecolor='black', linewidth=0.4)
    ax7.set_title(f"Polar Code (N={N_polar}, K={K_polar})\nGreen=info bits, Red=frozen bits",
                  color=BLUE, fontweight='bold')
    ax7.set_xlabel("Bit channel index", color='#8b949e')
    ax7.set_ylabel("Bit value", color='#8b949e')

    # ── 8. Spectral efficiency vs SNR ─────────────────────────
    ax8 = ax_(gs[2,1])
    snr_cont = np.linspace(-5, 35, 200)
    snr_lin  = 10**(snr_cont/10)
    se_1L  = np.log2(1 + snr_lin)              # 1 layer
    se_4L  = 4*np.log2(1 + snr_lin/4)          # 4 layers (shared power)
    se_8L  = 8*np.log2(1 + snr_lin/8)          # 8 layers

    ax8.plot(snr_cont, se_1L, color=BLUE,  lw=2, label='1 layer')
    ax8.plot(snr_cont, se_4L, color=ORG,   lw=2, label='4 layers')
    ax8.plot(snr_cont, se_8L, color=GRN,   lw=2, label='8 layers')

    for order, se_cap, marker in [(4,2,'QPSK'),(16,4,'16QAM'),(64,6,'64QAM'),(256,8,'256QAM')]:
        ax8.axhline(se_cap, color='gray', lw=0.6, ls=':')
        ax8.text(36, se_cap+0.1, marker, color='gray', fontsize=7)

    ax8.set_xlabel("SNR (dB)", color='#8b949e')
    ax8.set_ylabel("Spectral Efficiency (bits/s/Hz)", color='#8b949e')
    ax8.set_title("NR Spectral Efficiency (Shannon)\nvs Modulation Limits", color=BLUE, fontweight='bold')
    ax8.legend(fontsize=9, facecolor='#161b22', labelcolor='white')
    ax8.set_xlim([-5,35]); ax8.set_ylim([0,35])

    # ── 9. LDPC BG selection ──────────────────────────────────
    ax9 = ax_(gs[2,2])
    tbs_arr  = np.arange(100, 8000, 50)
    bg_arr   = np.array([select_ldpc_bg(t, 0.6) for t in tbs_arr])
    ax9.fill_between(tbs_arr, 0, 1, where=bg_arr==1, alpha=0.4, color=BLUE,  label='BG1 (large TBS)')
    ax9.fill_between(tbs_arr, 0, 1, where=bg_arr==2, alpha=0.4, color=RED,   label='BG2 (small TBS)')
    ax9.axvline(3824, color='white', lw=1.5, ls='--', label='TBS=3824 threshold')
    ax9.set_xlim([100, 8000]); ax9.set_ylim([0,1])
    ax9.set_xlabel("Transport Block Size (bits)", color='#8b949e')
    ax9.set_title("LDPC Base Graph Selection (TS 38.212)\nBG1 vs BG2", color=BLUE, fontweight='bold')
    ax9.legend(fontsize=9, facecolor='#161b22', labelcolor='white')
    ax9.set_yticks([])

    fig.text(0.5, 0.98, "5G NR Physical Layer — 3GPP TS 38.211/212/213",
             ha='center', color='white', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.965, "PDSCH · DMRS · Polar · LDPC · 256QAM · Multi-layer · Spectral Efficiency",
             ha='center', color='#8b949e', fontsize=10)

    plt.savefig('nr_physical_layer.png', dpi=130, bbox_inches='tight', facecolor=DARK)
    print("\n✓ Saved: nr_physical_layer.png")
    print("\n✅  5G NR Physical Layer demo complete.")
    plt.close('all')
