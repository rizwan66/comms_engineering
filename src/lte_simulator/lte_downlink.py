"""
src/lte_simulator/lte_downlink.py
===================================
LTE-inspired Downlink Physical Layer Simulator.

Architecture (3GPP TS 36.211 simplified):
  TX: Info bits → Conv. Code (rate 1/2) → Rate Match → QAM Map
      → OFDM Resource Grid (pilots + data) → CP-OFDM TX
  CH: AWGN / EPA multipath channel
  RX: CP-OFDM RX → LS Channel Estimation → MMSE Equalise
      → Soft LLR Demap → Rate De-match → Viterbi Decode
      → CRC check

Numerology (50 RB, 10 MHz BW):
  FFT size:          1024
  Active subcarriers: 600  (50 RBs × 12 SC/RB)
  Subcarrier spacing:  15 kHz
  CP length (normal):  72 samples
  OFDM symbols/subframe: 14
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp
import itertools

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. LTE CONFIGURATION
# ─────────────────────────────────────────────

class LTEConfig:
    FFT_SIZE   = 1024
    N_SC       = 600          # active subcarriers (50 RB × 12)
    CP_NORMAL  = 72
    N_SYM      = 14           # OFDM symbols per subframe
    FS         = 15.36e6      # sample rate (Hz)
    PILOT_SYMS = [0, 4, 7, 11]  # CRS symbol positions

    MCS_TABLE = {           # (modulation, code_rate)
        4:  ('QPSK',  0.50),
        12: ('16QAM', 0.67),
        20: ('64QAM', 0.75),
    }

# ─────────────────────────────────────────────
# 2. CRC
# ─────────────────────────────────────────────

def crc16(bits):
    """CRC-16 (16-bit checksum)."""
    poly = 0x1021; reg = 0xFFFF
    for b in bits:
        reg ^= int(b) << 15
        for _ in range(8):
            reg = ((reg << 1) ^ poly) & 0xFFFF if reg & 0x8000 else (reg << 1) & 0xFFFF
    return np.array([(reg >> (15-i)) & 1 for i in range(16)], dtype=np.uint8)

def attach_crc(bits):
    return np.concatenate([bits, crc16(bits)])

def check_crc(bits):
    if len(bits) < 16: return False, bits
    data, chk = bits[:-16], bits[-16:]
    return np.array_equal(crc16(data), chk), data

# ─────────────────────────────────────────────
# 3. CONVOLUTIONAL ENCODER (rate 1/2, K=7)
# ─────────────────────────────────────────────

G1 = 0o171   # 121 octal = generator 1
G2 = 0o133   # 91  octal = generator 2
K  = 7        # constraint length
MEM = K - 1   # memory (register) length = 6 bits

def conv_encode(bits):
    """Rate 1/2 convolutional encoder, K=7, generators (171,133) octal."""
    reg = 0
    out = []
    for b in bits:
        reg = ((reg >> 1) | (int(b) << (K-2))) & ((1 << MEM) - 1)
        # bit & generator parity
        c1 = bin(((int(b) << MEM) | reg) & G1).count('1') % 2
        c2 = bin(((int(b) << MEM) | reg) & G2).count('1') % 2
        out.extend([c1, c2])
    # Tail-biting: flush 6 zero bits
    for _ in range(MEM):
        reg = (reg >> 1) & ((1 << MEM) - 1)
        c1  = bin(reg & (G1 & ((1<<MEM)-1))).count('1') % 2
        c2  = bin(reg & (G2 & ((1<<MEM)-1))).count('1') % 2
        out.extend([c1, c2])
    return np.array(out, dtype=np.uint8)

# ─────────────────────────────────────────────
# 4. VITERBI DECODER (soft input)
# ─────────────────────────────────────────────

def _branch_metrics(llr_pair, state, next_bit):
    """Compute branch metric for Viterbi (max-log)."""
    reg = state
    c1 = bin(((next_bit << MEM) | ((reg >> 1) | (next_bit << (MEM-1)) & ((1<<MEM)-1))) & G1).count('1') % 2
    c2 = bin(((next_bit << MEM) | ((reg >> 1) | (next_bit << (MEM-1)) & ((1<<MEM)-1))) & G2).count('1') % 2
    # LLR convention: positive → bit 0
    m1 = llr_pair[0] if c1 == 0 else -llr_pair[0]
    m2 = llr_pair[1] if c2 == 0 else -llr_pair[1]
    return m1 + m2

def viterbi_decode(llr, n_states=64):
    """
    Soft Viterbi decoder for rate 1/2, K=7 convolutional code.
    llr: soft LLR values, length = 2 * (n_info + 6)
    """
    N = len(llr) // 2  # number of trellis steps
    INF = 1e9

    # Path metrics [state]
    pm = np.full(n_states, -INF)
    pm[0] = 0.0
    # Traceback
    tb = np.zeros((N, n_states), dtype=np.int8)

    for t in range(N):
        new_pm = np.full(n_states, -INF)
        pair   = llr[2*t : 2*t+2]
        for state in range(n_states):
            if pm[state] == -INF:
                continue
            for bit in range(2):
                reg_new = ((state >> 1) | (bit << (MEM-1))) & (n_states - 1)
                # compute parity
                inp = (bit << MEM) | ((state >> 1) | (bit << (MEM-1)))
                c1  = bin(inp & G1).count('1') % 2
                c2  = bin(inp & G2).count('1') % 2
                m   = (pair[0] if c1==0 else -pair[0]) + (pair[1] if c2==0 else -pair[1])
                metric = pm[state] + m
                if metric > new_pm[reg_new]:
                    new_pm[reg_new] = metric
                    tb[t, reg_new] = state
        pm = new_pm

    # Traceback from best final state
    best = int(np.argmax(pm))
    bits = np.zeros(N, dtype=np.uint8)
    state = best
    for t in range(N-1, -1, -1):
        prev  = tb[t, state]
        # Determine input bit from state transition
        bits[t] = (state >> (MEM-1)) & 1
        state   = prev

    return bits[:N-MEM]  # remove tail

# ─────────────────────────────────────────────
# 5. RATE MATCHING
# ─────────────────────────────────────────────

def rate_match(bits, target):
    """Circular buffer rate matching."""
    if len(bits) == 0: return bits
    if len(bits) >= target:
        return bits[:target]
    reps = int(np.ceil(target / len(bits)))
    return np.tile(bits, reps)[:target]

def rate_dematch_llr(llr, original_len):
    """Accumulate LLRs from circular buffer."""
    out = np.zeros(original_len, dtype=float)
    for i, v in enumerate(llr):
        out[i % original_len] += v
    return out

# ─────────────────────────────────────────────
# 6. QAM MAPPER / SOFT DEMAPPER
# ─────────────────────────────────────────────

def qpsk_map(bits):
    if len(bits) % 2: bits = np.append(bits, 0)
    # bit0 controls I: 0→+1, 1→-1  |  bit1 controls Q: 0→+1, 1→-1
    pairs = bits.reshape(-1,2)
    I = 1 - 2*pairs[:,0].astype(float)   # 0→+1, 1→-1
    Q = 1 - 2*pairs[:,1].astype(float)
    return (I + 1j*Q) / np.sqrt(2)

def qam16_map(bits):
    if len(bits) % 4: bits = np.append(bits, np.zeros(4-len(bits)%4,dtype=np.uint8))
    g = bits.reshape(-1,4)
    # bit0=0 → I>0, bit1=0 → |I|=3, bit1=1 → |I|=1
    # Gray: (0,0)→+3, (0,1)→+1, (1,1)→-1, (1,0)→-3
    def gray_level(b0, b1):
        return (1-2*b0) * (3 - 2*b1)    # b0=0→+, b1=0→3, b1=1→1
    I = np.array([gray_level(int(r[0]),int(r[1])) for r in g], dtype=float)
    Q = np.array([gray_level(int(r[2]),int(r[3])) for r in g], dtype=float)
    return (I + 1j*Q) / np.sqrt(10)

def qam64_map(bits):
    if len(bits) % 6: bits = np.append(bits, np.zeros(6-len(bits)%6,dtype=np.uint8))
    g = bits.reshape(-1,6)
    def gray6(b0,b1,b2):
        # b0=sign, b1=magnitude MSB, b2=magnitude LSB
        # 000→+7, 001→+5, 011→+3, 010→+1, 110→-1, 111→-3, 101→-5, 100→-7
        mag = 7 - 2*(b1*2 + b2*(1-2*b1) + (1-b1)*b2*2)
        return (1-2*b0) * (7 - 2*(2*b1 + b2))
    def lv(b0,b1,b2): return (1-2*int(b0))*(7 - 2*(2*int(b1)+int(b2)))
    I = np.array([lv(r[0],r[1],r[2]) for r in g],dtype=float)
    Q = np.array([lv(r[3],r[4],r[5]) for r in g],dtype=float)
    return (I + 1j*Q) / np.sqrt(42)

def modulate(bits, mod):
    if mod == 'QPSK':  return qpsk_map(bits)
    if mod == '16QAM': return qam16_map(bits)
    return qam64_map(bits)

def soft_demap(syms, mod, noise_var=0.1):
    """Soft LLR demapper. LLR>0 → bit=0 more likely."""
    nv = noise_var + 1e-10
    if mod == 'QPSK':
        llr = np.empty(len(syms)*2)
        llr[0::2] =  4*syms.real / nv   # I → bit 0
        llr[1::2] =  4*syms.imag / nv   # Q → bit 1
    elif mod == '16QAM':
        s = syms * np.sqrt(10)
        llr = np.empty(len(syms)*4)
        # bit0: I>0 → bit0=0 → LLR>0 when I>0
        llr[0::4] =  4*s.real / nv
        # bit1: |I|>2 → bit1=0 (|I|=3) → LLR>0 when |I|>2
        llr[1::4] =  4*(np.abs(s.real) - 2) / nv
        llr[2::4] =  4*s.imag / nv
        llr[3::4] =  4*(np.abs(s.imag) - 2) / nv
    else:  # 64QAM — vectorised max-log LLR
        # LEVELS ordered to match qam64_map: 000→+7,001→+5,011→+1,010→+3,110→-3,111→-1,101→-5,100→-7
        LEVELS = np.array([7,5,1,3,-3,-1,-5,-7], dtype=float) / np.sqrt(42)
        BITS   = np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0],
                           [1,1,0],[1,1,1],[1,0,1],[1,0,0]], dtype=np.uint8)
        llr = np.empty(len(syms)*6)
        for axis_idx, axis_vals in enumerate([syms.real, syms.imag]):
            # axis_vals: (N,), LEVELS: (8,)  → d2: (N,8)
            d2 = ((axis_vals[:,None] - LEVELS[None,:])**2) / nv
            neg_d2 = -d2   # (N,8)
            for bi in range(3):
                mask0 = BITS[:,bi] == 0        # (8,) bool
                s0 = np.max(neg_d2[:,mask0], axis=1)   # (N,)
                s1 = np.max(neg_d2[:,~mask0], axis=1)
                llr[axis_idx*3+bi::6] = np.clip(s0-s1, -20, 20)
    return np.clip(llr, -20, 20)

# ─────────────────────────────────────────────
# 7. RESOURCE GRID
# ─────────────────────────────────────────────

class ResourceGrid:
    def __init__(self, cfg: LTEConfig):
        self.cfg        = cfg
        self.pilot_syms = cfg.PILOT_SYMS
        self.pilot_sc   = np.arange(0, cfg.N_SC, 6)
        self.pilot_pos  = set((s,c) for s in self.pilot_syms for c in self.pilot_sc)

    def n_data(self):
        return cfg.N_SC * cfg.N_SYM - len(self.pilot_pos)

    def fill(self, data_syms, pilot=1+0j):
        grid = np.zeros((self.cfg.N_SYM, self.cfg.N_SC), dtype=complex)
        di = 0
        for s in range(self.cfg.N_SYM):
            for c in range(self.cfg.N_SC):
                if (s,c) in self.pilot_pos:
                    grid[s,c] = pilot
                elif di < len(data_syms):
                    grid[s,c] = data_syms[di]; di += 1
        return grid

    def extract_data(self, grid):
        return np.array([grid[s,c] for s in range(self.cfg.N_SYM)
                         for c in range(self.cfg.N_SC) if (s,c) not in self.pilot_pos])

    def n_data_re(self):
        return self.cfg.N_SC * self.cfg.N_SYM - len(self.pilot_pos)

# ─────────────────────────────────────────────
# 8. OFDM MOD / DEMOD
# ─────────────────────────────────────────────

def ofdm_mod(grid, fft_size, cp):
    out = []
    for sym in grid:
        freq = np.zeros(fft_size, dtype=complex)
        sc_start = (fft_size - len(sym)) // 2
        freq[sc_start:sc_start+len(sym)] = sym
        td = np.fft.ifft(freq)
        out.append(np.concatenate([td[-cp:], td]))
    return np.concatenate(out)

def ofdm_demod(rx, fft_size, cp, n_sym, n_sc):
    sym_len = fft_size + cp
    sc_start = (fft_size - n_sc) // 2
    grid = np.zeros((n_sym, n_sc), dtype=complex)
    for i in range(n_sym):
        seg  = rx[i*sym_len : i*sym_len+sym_len]
        if len(seg) < sym_len: break
        td   = seg[cp:]
        fd   = np.fft.fft(td, n=fft_size)
        grid[i] = fd[sc_start:sc_start+n_sc]
    return grid

# ─────────────────────────────────────────────
# 9. CHANNEL
# ─────────────────────────────────────────────

def awgn(sig, snr_db):
    P  = np.mean(np.abs(sig)**2)
    n0 = P / (10**(snr_db/10))
    return sig + np.sqrt(n0/2)*(np.random.randn(len(sig))+1j*np.random.randn(len(sig)))

def epa_channel(sig, fs):
    """3GPP EPA multipath (5 taps)."""
    delays  = np.array([0, 30, 70, 90, 110, 190, 410]) * 1e-9
    powers  = np.array([0, -1, -2, -3, -8, -17.2, -20.8])
    gains   = 10**(powers/20) * (np.random.randn(len(delays))+1j*np.random.randn(len(delays)))/np.sqrt(2)
    samp_d  = np.round(delays * fs).astype(int)
    out     = np.zeros(len(sig)+samp_d.max(), dtype=complex)
    for d, g in zip(samp_d, gains):
        out[d:d+len(sig)] += g * sig
    return out[:len(sig)]

# ─────────────────────────────────────────────
# 10. CHANNEL ESTIMATION
# ─────────────────────────────────────────────

def ls_estimate(rx_grid, rg: ResourceGrid, pilot=1+0j):
    H = np.ones_like(rx_grid)
    for s in rg.pilot_syms:
        h_p = rx_grid[s, rg.pilot_sc] / pilot
        H[s,:] = np.interp(np.arange(rg.cfg.N_SC), rg.pilot_sc, h_p.real) + \
              1j*np.interp(np.arange(rg.cfg.N_SC), rg.pilot_sc, h_p.imag)
    for c in range(rg.cfg.N_SC):
        h_c = H[rg.pilot_syms, c]
        H[:,c] = np.interp(np.arange(rg.cfg.N_SYM), rg.pilot_syms, h_c.real) + \
              1j*np.interp(np.arange(rg.cfg.N_SYM), rg.pilot_syms, h_c.imag)
    return H

def mmse_eq(rx, H, snr_lin):
    return rx * np.conj(H) / (np.abs(H)**2 + 1/max(snr_lin,0.01))

# ─────────────────────────────────────────────
# 11. FULL SIMULATION
# ─────────────────────────────────────────────

cfg = LTEConfig()

def run_one(mcs, snr_db, channel='awgn'):
    mod_str, rate = cfg.MCS_TABLE[mcs]
    bps = {'QPSK':2,'16QAM':4,'64QAM':6}[mod_str]

    rg     = ResourceGrid(cfg)
    n_data = rg.n_data_re()   # data REs per subframe

    # How many info bits fit in 1 subframe after coding
    n_coded_bits = n_data * bps
    n_info       = max(8, int(n_coded_bits * rate / 2) - 16)  # /2 for rate-1/2 conv code

    # TX
    info  = np.random.randint(0, 2, n_info, dtype=np.uint8)
    bcrc  = attach_crc(info)
    coded = conv_encode(bcrc)                    # rate 1/2 output
    rm    = rate_match(coded, n_coded_bits)      # rate match to fill grid
    syms  = modulate(rm, mod_str)

    # OFDM
    grid    = rg.fill(syms)
    tx_sig  = ofdm_mod(grid, cfg.FFT_SIZE, cfg.CP_NORMAL)

    # Channel
    if channel == 'epa':
        rx_sig = epa_channel(tx_sig, cfg.FS)
        rx_sig = awgn(rx_sig, snr_db)
    else:
        rx_sig = awgn(tx_sig, snr_db)

    # RX
    snr_lin  = 10**(snr_db/10)
    rx_grid  = ofdm_demod(rx_sig, cfg.FFT_SIZE, cfg.CP_NORMAL, cfg.N_SYM, cfg.N_SC)
    H        = ls_estimate(rx_grid, rg)
    eq_grid  = mmse_eq(rx_grid, H, snr_lin)
    rx_syms  = rg.extract_data(eq_grid)

    noise_var = 1 / max(snr_lin, 0.01)
    llr       = soft_demap(rx_syms[:len(syms)], mod_str, noise_var=noise_var)
    llr_dm    = rate_dematch_llr(llr, len(coded))
    rx_dec    = viterbi_decode(llr_dm)

    # CRC
    crc_ok, rx_info = check_crc(rx_dec[:len(bcrc)])
    n     = min(len(info), len(rx_info))
    ber   = np.sum(info[:n] != rx_info[:n]) / max(n, 1)
    return ber, crc_ok

# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  LTE DOWNLINK LINK SIMULATOR")
    print("  3GPP TS 36.211 — PDSCH Physical Layer")
    print("=" * 60)

    # Numerology summary
    rg_demo = ResourceGrid(cfg)
    print(f"\n  Numerology:")
    print(f"  FFT size          : {cfg.FFT_SIZE}")
    print(f"  Active subcarriers: {cfg.N_SC}  (50 RBs x 12)")
    print(f"  OFDM symbols/sf   : {cfg.N_SYM}")
    print(f"  Data REs/sf       : {rg_demo.n_data_re()}")
    print(f"  Pilot REs/sf      : {len(rg_demo.pilot_pos)}")

    # ── Resource grid visualisation ──────────────────────
    dummy_syms = np.ones(rg_demo.n_data_re(), dtype=complex) * (0.5+0.5j)
    grid_vis   = rg_demo.fill(dummy_syms)

    # Create mask: 0=data, 1=pilot
    mask = np.zeros_like(grid_vis, dtype=float)
    for (s,c) in rg_demo.pilot_pos:
        mask[s,c] = 1.0

    fig1, ax = plt.subplots(figsize=(14,4))
    ax.imshow(mask.T[:60,:], aspect='auto', cmap='RdYlGn_r', origin='lower')
    ax.set_xlabel("OFDM symbol index"); ax.set_ylabel("Subcarrier index")
    ax.set_title("LTE Resource Grid — Green=Data, Red=CRS Pilot (first 60 subcarriers)")
    plt.tight_layout()
    fig1.savefig("lte_resource_grid.png", dpi=120, bbox_inches='tight')
    print("\n✓ Saved: lte_resource_grid.png")

    # ── BER sweep ────────────────────────────────────────
    snr_range = list(range(0, 22, 3))
    results   = {}
    mcs_list  = [4, 12, 20]

    print("\n[BER Sweep — AWGN channel]")
    for mcs in mcs_list:
        mod_str, rate = cfg.MCS_TABLE[mcs]
        print(f"\n  MCS={mcs} ({mod_str}, rate={rate:.2f}):")
        bers, crcs = [], []
        for snr in snr_range:
            ber, crc = run_one(mcs, snr, 'awgn')
            bers.append(max(ber, 1e-5)); crcs.append(crc)
            crc_str = "✓" if crc else "✗"
            print(f"  SNR={snr:3d}dB  BER={ber:.4f}  CRC={crc_str}")
        results[mcs] = (bers, crcs)

    # BER / throughput figure
    from scipy.special import erfc
    snr_arr = np.array(snr_range)

    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("LTE PDSCH — BER & Throughput vs SNR (AWGN)", fontsize=13, fontweight='bold')

    colors = ['steelblue','darkorange','mediumseagreen']
    styles = ['-o','-s','-^']
    for (mcs, (bers, crcs)), col, sty in zip(results.items(), colors, styles):
        mod_str, rate = cfg.MCS_TABLE[mcs]
        axes[0].semilogy(snr_arr, bers, sty, color=col, lw=1.5,
                         label=f"MCS{mcs} {mod_str} r={rate}", markersize=5)

    # QPSK theory reference
    ber_th = 0.5 * erfc(np.sqrt(10**(snr_arr/10)))
    axes[0].semilogy(snr_arr, ber_th, 'k--', lw=1, label='Uncoded QPSK theory')
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("BER")
    axes[0].set_title("Bit Error Rate"); axes[0].legend(fontsize=8)
    axes[0].set_ylim([1e-4, 1]); axes[0].grid(True, which='both', alpha=0.3)

    # Normalised throughput (1 - BER) * spectral efficiency
    for (mcs,(bers,crcs)), col, sty in zip(results.items(), colors, styles):
        mod_str, rate = cfg.MCS_TABLE[mcs]
        bps = {'QPSK':2,'16QAM':4,'64QAM':6}[mod_str]
        eff = [bps * rate * (1-b) for b in bers]
        axes[1].plot(snr_arr, eff, sty, color=col, lw=1.5,
                     label=f"MCS{mcs} {mod_str}", markersize=5)
    axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("bits/s/Hz")
    axes[1].set_title("Normalised Throughput"); axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig("lte_ber_throughput.png", dpi=120, bbox_inches='tight')
    print("\n✓ Saved: lte_ber_throughput.png")

    # ── Channel estimation visualisation ─────────────────
    print("\n[Channel Estimation Demo — EPA channel]")
    # TX one subframe
    info = np.random.randint(0, 2, 500, dtype=np.uint8)
    bcrc = attach_crc(info)
    cod  = conv_encode(bcrc)
    rg   = ResourceGrid(cfg)
    rm   = rate_match(cod, rg.n_data_re()*2)
    syms = modulate(rm, 'QPSK')
    grid = rg.fill(syms)
    tx   = ofdm_mod(grid, cfg.FFT_SIZE, cfg.CP_NORMAL)

    # EPA channel
    rx_epa = epa_channel(tx, cfg.FS)
    rx_epa = awgn(rx_epa, 20)
    rx_g   = ofdm_demod(rx_epa, cfg.FFT_SIZE, cfg.CP_NORMAL, cfg.N_SYM, cfg.N_SC)
    H_est  = ls_estimate(rx_g, rg)

    fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig3.suptitle("LTE Channel Estimation — EPA Multipath (SNR=20dB)", fontsize=13, fontweight='bold')
    axes[0].plot(np.abs(H_est[0,:]), color='steelblue', lw=1.2)
    axes[0].set_title("Estimated |H| — Symbol 0"); axes[0].set_xlabel("Subcarrier"); axes[0].grid(alpha=0.3)
    axes[1].imshow(np.abs(H_est), aspect='auto', cmap='viridis', origin='lower')
    axes[1].set_title("Channel Magnitude (time × freq)"); axes[1].set_xlabel("Subcarrier"); axes[1].set_ylabel("OFDM symbol")
    eq_g = mmse_eq(rx_g, H_est, 100)
    rx_s = rg.extract_data(eq_g)[:1000]
    axes[2].scatter(rx_s.real, rx_s.imag, alpha=0.4, s=8, color='darkorange')
    axes[2].set_title("QPSK Constellation after MMSE EQ"); axes[2].set_aspect('equal')
    axes[2].axhline(0,color='k',lw=0.4); axes[2].axvline(0,color='k',lw=0.4); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    fig3.savefig("lte_channel_estimation.png", dpi=120, bbox_inches='tight')
    print("✓ Saved: lte_channel_estimation.png")

    print("\n✅  LTE simulator demo complete — 3 figures saved.")
    plt.close('all')
