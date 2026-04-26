"""
src/nr_numerology/nr_frame.py
==============================
5G NR (New Radio) Numerology and Frame Structure
3GPP TS 38.211

Covers:
  - Numerology (μ = 0..4): subcarrier spacing, slot duration
  - Frame / subframe / slot / symbol structure
  - Resource grid: PRB, RE, CORESET
  - Reference signals: DMRS, CSI-RS, SSB (Synchronisation Signal Block)
  - Physical channels: PDSCH, PDCCH, PBCH, PUSCH, PUCCH
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 1. NR NUMEROLOGY TABLE (3GPP TS 38.211 Table 4.2-1)
# ─────────────────────────────────────────────

NR_NUMEROLOGY = {
    # μ: (Δf kHz, slot_dur_ms, sym/slot, CP_type, use_case)
    0: {'scs_khz': 15,   'slot_ms': 1.0,    'sym_per_slot': 14, 'cp': 'normal',   'use': 'Sub-6GHz FDD/TDD'},
    1: {'scs_khz': 30,   'slot_ms': 0.5,    'sym_per_slot': 14, 'cp': 'normal',   'use': 'Sub-6GHz TDD'},
    2: {'scs_khz': 60,   'slot_ms': 0.25,   'sym_per_slot': 14, 'cp': 'normal/extended', 'use': 'Sub-6GHz / FR2'},
    3: {'scs_khz': 120,  'slot_ms': 0.125,  'sym_per_slot': 14, 'cp': 'normal',   'use': 'mmWave FR2'},
    4: {'scs_khz': 240,  'slot_ms': 0.0625, 'sym_per_slot': 14, 'cp': 'normal',   'use': 'mmWave reference'},
}

# Bandwidth options per numerology (MHz): max PRBs
NR_BW_TABLE = {
    0: {5:25, 10:52, 15:79, 20:106, 25:133, 40:216, 50:270},        # 15 kHz SCS
    1: {5:11, 10:24, 15:38, 20:51,  25:65,  40:106, 50:133, 100:264},# 30 kHz SCS
    2: {10:11,15:18, 20:24, 25:31,  40:51,  50:65,  100:132,200:264},# 60 kHz SCS
    3: {50:32,100:66,200:132,400:264},                                 # 120 kHz SCS
    4: {50:16,100:32,200:66,400:132},                                  # 240 kHz SCS
}

FRAME_DUR_MS  = 10.0    # 10 ms radio frame
SUBFRAME_DUR  = 1.0     # 1 ms subframe
SUBFRAMES_PER_FRAME = 10


class NRConfig:
    """5G NR configuration for a given numerology and bandwidth."""
    def __init__(self, mu=1, bw_mhz=100):
        self.mu      = mu
        self.bw_mhz  = bw_mhz
        num          = NR_NUMEROLOGY[mu]
        self.scs_khz    = num['scs_khz']
        self.scs_hz     = num['scs_khz'] * 1e3
        self.slot_dur_ms= num['slot_ms']
        self.sym_per_slot = num['sym_per_slot']
        self.slots_per_subframe = int(1.0 / self.slot_dur_ms)
        self.slots_per_frame    = int(FRAME_DUR_MS / self.slot_dur_ms)
        self.n_prb = NR_BW_TABLE.get(mu, {}).get(bw_mhz, 52)
        self.n_sc  = self.n_prb * 12          # 12 SC per PRB
        self.fft_size = self._fft_size()
        self.cp_len   = self._cp_len()
        self.fs       = self.scs_hz * self.fft_size   # sample rate

    def _fft_size(self):
        """Minimum FFT size >= N_SC (next power of 2)."""
        n = self.n_sc
        p = 1
        while p < n: p <<= 1
        return max(p, 128)

    def _cp_len(self):
        """Normal CP length = 144/2048 * Ts * FFT_size (approx)."""
        return max(int(self.fft_size * 144 / 2048), 6)

    def sym_dur_us(self):
        return 1e6 / self.scs_hz

    def __str__(self):
        return (f"NRConfig(μ={self.mu}, SCS={self.scs_khz}kHz, "
                f"BW={self.bw_mhz}MHz, PRB={self.n_prb}, "
                f"SC={self.n_sc}, FFT={self.fft_size}, "
                f"slot={self.slot_dur_ms}ms)")


# ─────────────────────────────────────────────
# 2. SLOT FORMATS (3GPP TS 38.213 Table 11.1.1-1)
# ─────────────────────────────────────────────

# Format: list of 14 symbols as D=downlink, U=uplink, F=flexible/guard
SLOT_FORMATS = {
    0:  'D'*14,                   # All downlink
    1:  'U'*14,                   # All uplink
    2:  'D'*12 + 'F'*2,           # DL heavy
    7:  'D'*6 + 'F'*4 + 'U'*4,   # TDD mixed (common for NR)
    28: 'D'*3 + 'F'*8 + 'U'*3,   # Balanced TDD
    34: 'D'*13 + 'U'*1,           # Almost all DL
    55: 'D'*7 + 'F'*2 + 'U'*5,   # DDDSU pattern equivalent
}

TDD_PATTERNS = {
    'DDDSU':  ['D','D','D','S','U'],  # 5-slot TDD (LTE-like)
    'DDDDD_UUUUU': ['D']*5 + ['U']*5,
    'DL_heavy': ['D','D','D','D','S','U','U'],  # 7-slot pattern
    'balanced': ['D','D','S','U','U'],
}


# ─────────────────────────────────────────────
# 3. RESOURCE GRID BUILDER
# ─────────────────────────────────────────────

class NRResourceGrid:
    """
    5G NR resource grid for one slot.
    Dimensions: [n_sc × n_symbols] = [N_PRB*12 × 14]
    """

    # RE types
    DATA   = 0
    DMRS   = 1
    CSIRS  = 2
    PTRS   = 3
    GUARD  = 4
    CORESET= 5

    def __init__(self, cfg: NRConfig, dmrs_type=1):
        self.cfg       = cfg
        self.n_sc      = cfg.n_sc
        self.n_sym     = cfg.sym_per_slot
        self.dmrs_type = dmrs_type
        self.grid      = np.zeros((self.n_sc, self.n_sym), dtype=int)
        self._place_dmrs()
        self._place_coreset()

    def _place_dmrs(self):
        """Place DMRS (Demodulation Reference Signal) — Type 1, mapping type A."""
        # DMRS on symbol 2 (single-symbol front-loaded, 3GPP TS 38.211 §7.4.1)
        dmrs_syms = [2]                # additional at [2,3] for PDSCH mapping B
        if self.dmrs_type == 1:
            # Type 1: every other SC (δ=0,2,4,...) per CDM group
            dmrs_sc = np.arange(0, self.n_sc, 2)
        else:
            # Type 2: groups of 2 SC every 3rd
            dmrs_sc = np.concatenate([np.arange(k, self.n_sc, 6) for k in [0,1]])
        for s in dmrs_syms:
            self.grid[dmrs_sc, s] = self.DMRS

    def _place_coreset(self):
        """CORESET (Control Resource Set) on first 1-3 symbols."""
        # Simplified: 1 symbol CORESET across first 48 PRBs
        coreset_prbs  = min(self.cfg.n_prb, 48)
        coreset_sc    = coreset_prbs * 12
        self.grid[:coreset_sc, 0] = self.CORESET

    def place_csirs(self, row=1):
        """CSI-RS (Channel State Information RS) placement."""
        # Simplified row 1: single port, symbol 13, every 4th SC
        csirs_sc = np.arange(0, self.n_sc, 4)
        self.grid[csirs_sc, 13] = self.CSIRS

    def n_data_re(self):
        return int(np.sum(self.grid == self.DATA))

    def color_map(self):
        """Return RGB image of resource grid."""
        colors = {
            self.DATA:    [0.12, 0.47, 0.71],  # blue
            self.DMRS:    [0.17, 0.63, 0.17],  # green
            self.CSIRS:   [0.84, 0.15, 0.16],  # red
            self.PTRS:    [1.00, 0.50, 0.05],  # orange
            self.GUARD:   [0.50, 0.50, 0.50],  # grey
            self.CORESET: [0.58, 0.40, 0.74],  # purple
        }
        img = np.zeros((self.n_sc, self.n_sym, 3))
        for re_type, col in colors.items():
            mask = self.grid == re_type
            img[mask] = col
        return img


# ─────────────────────────────────────────────
# 4. 5G NR OFDM MOD / DEMOD
# ─────────────────────────────────────────────

def nr_ofdm_mod(grid_freq, cfg: NRConfig):
    """
    5G NR OFDM modulation for one slot (14 symbols).
    grid_freq: (n_sc × 14) complex array
    Returns time-domain signal.
    """
    fft_size = cfg.fft_size
    sc_start = (fft_size - cfg.n_sc) // 2
    symbols  = []
    for s in range(cfg.sym_per_slot):
        freq = np.zeros(fft_size, dtype=complex)
        freq[sc_start:sc_start+cfg.n_sc] = grid_freq[:, s]
        td   = np.fft.ifft(freq) * np.sqrt(fft_size)
        cp   = td[-cfg.cp_len:]
        symbols.append(np.concatenate([cp, td]))
    return np.concatenate(symbols)


def nr_ofdm_demod(signal, cfg: NRConfig):
    """5G NR OFDM demodulation. Returns (n_sc × 14) frequency-domain grid."""
    fft_size  = cfg.fft_size
    sym_len   = fft_size + cfg.cp_len
    sc_start  = (fft_size - cfg.n_sc) // 2
    grid_out  = np.zeros((cfg.n_sc, cfg.sym_per_slot), dtype=complex)
    for s in range(cfg.sym_per_slot):
        seg = signal[s*sym_len:(s+1)*sym_len]
        if len(seg) < sym_len:
            break
        td  = seg[cfg.cp_len:]
        fd  = np.fft.fft(td) / np.sqrt(fft_size)
        grid_out[:, s] = fd[sc_start:sc_start+cfg.n_sc]
    return grid_out


# ─────────────────────────────────────────────
# 5. SSB (Synchronisation Signal Block)
# ─────────────────────────────────────────────

def generate_ssb(cell_id=0, mu=1):
    """
    Generate 5G NR SSB (SS/PBCH block).
    Contains: PSS (Primary Sync), SSS (Secondary Sync), PBCH.
    SSB occupies 20 PRBs (240 SC), 4 OFDM symbols.
    """
    N_sc_ssb = 240   # 20 PRBs
    N_sym_ssb = 4

    # PSS: m-sequence based, length 127, centered in symbol 0
    pss = _generate_pss(cell_id % 3)
    # SSS: Gold sequence based, length 127, centered in symbol 2
    sss = _generate_sss(cell_id)
    # PBCH: QPSK on symbols 1,3 and sides of 2
    pbch = np.exp(1j * np.pi/4 * np.random.randint(0, 4, N_sc_ssb * 2))

    ssb_grid = np.zeros((N_sc_ssb, N_sym_ssb), dtype=complex)
    # Place PSS in symbol 0, centre 127 SC
    ssb_grid[56:56+127, 0] = pss
    # Place SSS in symbol 2
    ssb_grid[56:56+127, 2] = sss
    # PBCH in symbols 1,3
    ssb_grid[:, 1] = pbch[:N_sc_ssb]
    ssb_grid[:, 3] = pbch[N_sc_ssb:]

    return ssb_grid, pss, sss


def _generate_pss(N_id2):
    """PSS: BPSK m-sequence of length 127."""
    x = np.zeros(127)
    init = [0,1,1,0,1,1,1]   # initial state
    reg  = list(init)
    seq  = []
    for _ in range(127):
        seq.append(reg[0])
        fb = (reg[3] + reg[0]) % 2
        reg = [fb] + reg[:-1]
    seq = np.array(seq)
    n   = np.arange(127)
    pss = 1 - 2 * ((seq + N_id2) % 2)
    return pss.astype(complex)


def _generate_sss(cell_id):
    """SSS: Gold sequence of length 127."""
    N_id1 = cell_id // 3
    N_id2 = cell_id % 3

    def m_seq(init):
        reg = list(init)
        out = []
        for _ in range(127):
            out.append(reg[0])
            fb = (reg[3] + reg[0]) % 2
            reg = [fb] + reg[:-1]
        return np.array(out)

    x0 = m_seq([1,0,0,0,0,0,0])
    x1 = m_seq([1,0,0,0,0,0,0])
    m0 = 15 * (N_id1 // 112) + 5 * N_id2
    m1 = N_id1 % 112
    n  = np.arange(127)
    sss = (1-2*x0[(n+m0)%127]) * (1-2*x1[(n+m1)%127])
    return sss.astype(complex)


def ssb_detect(rx_signal, cfg: NRConfig, cell_id=0):
    """Detect SSB using PSS correlation."""
    pss, _ = _generate_pss(cell_id % 3), None
    # Cross-correlate with PSS
    corr  = np.abs(np.correlate(rx_signal[:5000].real, pss.real, mode='full'))
    peak  = np.argmax(corr)
    return peak, corr


# ─────────────────────────────────────────────
# 6. THROUGHPUT CALCULATOR
# ─────────────────────────────────────────────

def nr_peak_throughput(mu, bw_mhz, layers=4, mod_order=8, code_rate=948/1024):
    """
    Calculate 5G NR peak DL throughput (3GPP TS 38.306).
    mod_order: 2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM
    """
    cfg       = NRConfig(mu, bw_mhz)
    rg        = NRResourceGrid(cfg)
    n_data_re = rg.n_data_re()

    # Bits per slot per layer
    bits_per_slot = n_data_re * mod_order * code_rate

    # Total bits per second
    slots_per_sec = 1000 / cfg.slot_dur_ms
    tput_mbps     = bits_per_slot * slots_per_sec * layers / 1e6

    return {
        'mu':           mu,
        'scs_khz':      cfg.scs_khz,
        'bw_mhz':       bw_mhz,
        'n_prb':        cfg.n_prb,
        'n_data_re':    n_data_re,
        'layers':       layers,
        'modulation':   {2:'QPSK',4:'16QAM',6:'64QAM',8:'256QAM'}.get(mod_order,'?'),
        'code_rate':    code_rate,
        'tput_mbps':    tput_mbps,
        'slots_per_sec':slots_per_sec,
    }


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  5G NR — Numerology & Frame Structure (3GPP TS 38.211)")
    print("=" * 65)

    # ── 1. Numerology comparison ──────────────────────────────
    print("\n  NR Numerology Table:")
    print(f"  {'μ':>3}  {'SCS':>8}  {'Slot':>8}  {'Slots/frame':>12}  {'Use case'}")
    print("  " + "-"*65)
    for mu, n in NR_NUMEROLOGY.items():
        sps  = int(FRAME_DUR_MS / n['slot_ms'])
        print(f"  {mu:>3}  {n['scs_khz']:>5} kHz  {n['slot_ms']:>5.4f} ms  "
              f"{sps:>12}  {n['use']}")

    # ── 2. Throughput table ───────────────────────────────────
    print("\n  Peak DL Throughput (4 layers, 256QAM, R=0.926):")
    print(f"  {'Config':>20}  {'PRBs':>6}  {'Tput (Mbps)':>12}")
    print("  " + "-"*45)
    configs = [(0,20),(0,50),(1,100),(2,100),(3,100),(3,400)]
    for mu, bw in configs:
        r = nr_peak_throughput(mu, bw, layers=4, mod_order=8, code_rate=948/1024)
        print(f"  μ={mu} {bw}MHz {r['scs_khz']}kHz   {r['n_prb']:>6}  {r['tput_mbps']:>12.1f}")

    # ── Plots ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

    DARK = '#0d1117'; BG = '#161b22'; GRD = dict(alpha=0.15, color='white')
    def dark_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(BG)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.grid(**GRD)
        return ax

    # Plot 1: NR Frame structure timeline
    ax1 = dark_ax(gs[0, :2])
    colors_du = {'D':'#58a6ff', 'U':'#3fb950', 'F':'#e3b341', 'S':'#f85149'}
    labels_seen = set()
    for mu_p in [0, 1, 3]:
        num    = NR_NUMEROLOGY[mu_p]
        n_slots = int(2.0 / num['slot_ms'])  # show 2 ms
        y_base = mu_p * 1.2
        pattern = list('DDDDDSUUUU')   # simplified TDD
        for s in range(n_slots):
            slot_type = pattern[s % len(pattern)]
            col = colors_du.get(slot_type, '#aaa')
            lbl = slot_type if slot_type not in labels_seen else None
            labels_seen.add(slot_type)
            ax1.barh(y_base, num['slot_ms'], left=s*num['slot_ms'],
                     height=0.9, color=col, edgecolor='#0d1117', lw=0.5,
                     label=lbl, alpha=0.85)
        ax1.text(-0.15, y_base+0.45,
                 f"μ={mu_p}\n{num['scs_khz']}kHz",
                 ha='right', va='top', color='#8b949e', fontsize=8)

    ax1.set_xlabel('Time (ms)', color='#8b949e')
    ax1.set_title('5G NR TDD Slot Structure (2 ms window)', color='#58a6ff', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, facecolor=BG, labelcolor='white',
               labels=['Downlink','Uplink','Flexible','Special'])
    ax1.set_yticks([]); ax1.set_xlim(0, 2)

    # Plot 2: Resource grid (μ=1, 20 MHz)
    ax2 = dark_ax(gs[0, 2])
    cfg2  = NRConfig(mu=1, bw_mhz=20)
    rg2   = NRResourceGrid(cfg2)
    rg2.place_csirs()
    img   = rg2.color_map()
    ax2.imshow(img[:120, :], aspect='auto', origin='lower', interpolation='nearest')
    ax2.set_title('NR Resource Grid (μ=1, 20MHz)\nfirst 10 PRBs, 1 slot', color='#58a6ff', fontweight='bold')
    ax2.set_xlabel('OFDM symbol', color='#8b949e')
    ax2.set_ylabel('Subcarrier', color='#8b949e')
    legend_patches = [
        mpatches.Patch(color='#1e78b4', label='Data (PDSCH)'),
        mpatches.Patch(color='#2ca02c', label='DMRS'),
        mpatches.Patch(color='#d62728', label='CSI-RS'),
        mpatches.Patch(color='#9467bd', label='CORESET/PDCCH'),
    ]
    ax2.legend(handles=legend_patches, fontsize=7, facecolor=BG, labelcolor='white',
               loc='upper right')

    # Plot 3: SSB structure
    ax3 = dark_ax(gs[1, 0])
    ssb_grid, pss, sss = generate_ssb(cell_id=42, mu=1)
    ax3.imshow(np.abs(ssb_grid), aspect='auto', origin='lower', cmap='plasma')
    ax3.set_title('SSB (Synchronisation Signal Block)\nPSS+SSS+PBCH, 240 SC × 4 sym', color='#58a6ff', fontweight='bold')
    ax3.set_xlabel('SSB symbol', color='#8b949e'); ax3.set_ylabel('Subcarrier', color='#8b949e')
    for label, sym in [('PSS',0),('PBCH',1),('SSS+PBCH',2),('PBCH',3)]:
        ax3.text(sym, 120, label, ha='center', color='white', fontsize=7, fontweight='bold')

    # Plot 4: PSS correlation / cell detection
    ax4 = dark_ax(gs[1, 1])
    cfg4 = NRConfig(mu=1, bw_mhz=20)
    ssb4, pss4, _ = generate_ssb(cell_id=0)
    # Embed SSB in noise signal
    noise_sig = (np.random.randn(3000) + 1j*np.random.randn(3000)) * 0.3
    noise_sig[500:500+len(pss4)] += pss4 * 2
    corr = np.abs(np.correlate(noise_sig.real, pss4.real, mode='full'))
    ax4.plot(corr, color='#58a6ff', lw=0.8)
    ax4.axvline(np.argmax(corr), color='#f85149', lw=1.5, ls='--', label=f'Peak @ {np.argmax(corr)}')
    ax4.set_title('Cell Detection — PSS Correlation', color='#58a6ff', fontweight='bold')
    ax4.set_xlabel('Sample', color='#8b949e'); ax4.set_ylabel('|Correlation|', color='#8b949e')
    ax4.legend(fontsize=8, facecolor=BG, labelcolor='white')

    # Plot 5: Peak throughput vs bandwidth
    ax5 = dark_ax(gs[1, 2])
    bw_range = [5, 10, 20, 40, 50, 100]
    for mu_p, col_p in [(0,'#58a6ff'),(1,'#3fb950'),(3,'#e3b341')]:
        tputs = []
        bws   = []
        for bw in bw_range:
            try:
                r = nr_peak_throughput(mu_p, bw, layers=4, mod_order=8)
                tputs.append(r['tput_mbps'])
                bws.append(bw)
            except: pass
        if tputs:
            ax5.plot(bws, tputs, 'o-', color=col_p, lw=2, ms=5,
                     label=f"μ={mu_p} ({NR_NUMEROLOGY[mu_p]['scs_khz']}kHz)")
    ax5.set_xlabel('Bandwidth (MHz)', color='#8b949e')
    ax5.set_ylabel('Peak Throughput (Mbps)', color='#8b949e')
    ax5.set_title('5G NR Peak DL Throughput\n(4L, 256QAM, R=0.926)', color='#58a6ff', fontweight='bold')
    ax5.legend(fontsize=8, facecolor=BG, labelcolor='white')

    fig.text(0.5, 0.98, '5G NR — Numerology, Frame Structure & Resource Grid (3GPP TS 38.211)',
             ha='center', color='white', fontsize=14, fontweight='bold')
    plt.savefig('nr_numerology.png', dpi=130, bbox_inches='tight', facecolor=DARK)
    print("\n✓ Saved: nr_numerology.png")
    plt.close('all')
    print("\n✅  NR Numerology module demo complete.")
