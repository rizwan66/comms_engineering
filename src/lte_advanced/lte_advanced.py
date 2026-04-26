"""
src/lte_advanced/lte_advanced.py
==================================
LTE-Advanced Features (3GPP Release 10-13)
Covers: Carrier Aggregation, eICIC/HetNet, CoMP,
        LTE-A Pro (LAA, LWIP, NB-IoT, eMTC).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# 1. CARRIER AGGREGATION (CA)
# ─────────────────────────────────────────────

class ComponentCarrier:
    """One LTE Component Carrier."""
    CC_TYPES = {6:'1.4MHz',15:'3MHz',25:'5MHz',50:'10MHz',75:'15MHz',100:'20MHz'}

    def __init__(self, fc_mhz, bw_mhz, cell_id=0, cc_id=0):
        self.fc_mhz  = fc_mhz
        self.bw_mhz  = bw_mhz
        self.n_prb   = {1.4:6,3:15,5:25,10:50,15:75,20:100}.get(bw_mhz, 50)
        self.cell_id = cell_id
        self.cc_id   = cc_id
        self.is_pcc  = (cc_id == 0)   # Primary CC

    def peak_tput_mbps(self, layers=2, mcs=26):
        """Approximate DL peak throughput (Mbps)."""
        # 3GPP simplified: prbs × bits_per_rb × slots_per_s × layers
        n_re_per_prb = 12 * 14 - 24   # approx overhead removed
        mod_order = {0:2,10:4,16:4,17:6,20:6,26:6,28:6}.get(mcs, 6)
        code_rate = 0.93
        tput = self.n_prb * n_re_per_prb * mod_order * code_rate * 1000 * layers / 1e6
        return tput


class CarrierAggregator:
    """
    LTE-A Carrier Aggregation.
    Supports up to 5 CCs, 3 aggregation types:
    - Intra-band contiguous (same band, adjacent)
    - Intra-band non-contiguous (same band, separate)
    - Inter-band (different bands)
    """
    CA_TYPES = {
        'intra_contiguous':    'Intra-band Contiguous',
        'intra_non_contiguous':'Intra-band Non-Contiguous',
        'inter_band':          'Inter-band',
    }

    def __init__(self):
        self.carriers = []

    def add_carrier(self, fc_mhz, bw_mhz, cc_id=None):
        cc_id = cc_id if cc_id is not None else len(self.carriers)
        cc    = ComponentCarrier(fc_mhz, bw_mhz, cc_id=cc_id)
        self.carriers.append(cc)
        return cc

    def total_bandwidth(self):
        return sum(cc.bw_mhz for cc in self.carriers)

    def total_prbs(self):
        return sum(cc.n_prb for cc in self.carriers)

    def peak_tput(self, layers=4):
        return sum(cc.peak_tput_mbps(layers) for cc in self.carriers)

    def ca_type(self):
        if len(self.carriers) < 2: return 'single'
        bands = [int(cc.fc_mhz) for cc in self.carriers]
        bw    = max(bands) - min(bands)
        if bw > 100:   return 'inter_band'
        elif bw > 20:  return 'intra_non_contiguous'
        else:          return 'intra_contiguous'

    def spectrum_map(self):
        """Return (center_mhz, bw_mhz) for plotting."""
        return [(cc.fc_mhz, cc.bw_mhz) for cc in self.carriers]

    def report(self):
        print(f"\n  Carrier Aggregation Report:")
        print(f"  CCs           : {len(self.carriers)}")
        print(f"  CA type       : {self.CA_TYPES.get(self.ca_type(), self.ca_type())}")
        print(f"  Total BW      : {self.total_bandwidth()} MHz")
        print(f"  Total PRBs    : {self.total_prbs()}")
        print(f"  Est. Peak DL  : {self.peak_tput(layers=4):.0f} Mbps (4 layers, 64QAM)")
        for cc in self.carriers:
            pcc = "PCC" if cc.is_pcc else f"SCC{cc.cc_id}"
            print(f"  {pcc}: {cc.fc_mhz} MHz, BW={cc.bw_mhz}MHz, PRBs={cc.n_prb}")


# ─────────────────────────────────────────────
# 2. HETNET / eICIC (Enhanced ICIC)
# ─────────────────────────────────────────────

class Cell:
    TYPES = {
        'macro': {'power_dbm':46,'radius_m':1000,'color':'#58a6ff'},
        'pico':  {'power_dbm':30,'radius_m':200, 'color':'#3fb950'},
        'femto': {'power_dbm':20,'radius_m':50,  'color':'#f85149'},
    }

    def __init__(self, cell_type, pos_x, pos_y, cell_id=0):
        self.cell_type = cell_type
        self.pos       = np.array([pos_x, pos_y])
        self.cell_id   = cell_id
        params         = self.TYPES[cell_type]
        self.tx_power  = params['power_dbm']
        self.radius    = params['radius_m']
        self.color     = params['color']


def sinr_map_hetnet(cells, grid_size=1000, resolution=50, fc_ghz=2.1, noise_dbm=-104):
    """Compute SINR map for a HetNet deployment."""
    x  = np.linspace(-grid_size//2, grid_size//2, resolution)
    y  = np.linspace(-grid_size//2, grid_size//2, resolution)
    xx, yy = np.meshgrid(x, y)
    SINR = np.zeros((resolution, resolution))

    def path_loss(d_m):
        d = max(d_m, 1)
        return 128.1 + 37.6 * np.log10(max(d/1000, 0.001))   # UMa NLOS simplified

    noise_lin = 10**((noise_dbm)/10) * 1e-3   # W

    for i in range(resolution):
        for j in range(resolution):
            pt = np.array([xx[i,j], yy[i,j]])
            powers = []
            for cell in cells:
                d_m = max(np.linalg.norm(pt - cell.pos), 1)
                pl  = path_loss(d_m)
                rx  = cell.tx_power - pl   # dBm
                powers.append(10**(rx/10) * 1e-3)   # W
            if powers:
                sig  = max(powers)
                intf = sum(powers) - sig + noise_lin
                SINR[i,j] = 10*np.log10(sig / max(intf, 1e-20))

    return xx, yy, SINR


# ─────────────────────────────────────────────
# 3. CoMP (Coordinated Multi-Point)
# ─────────────────────────────────────────────

def comp_jt_gain(N_cells, N_bs_per_cell, snr_db=10):
    """
    Estimate capacity gain from CoMP Joint Transmission (JT).
    JT coherently combines signals from N_cells cooperating BSs.
    """
    snr = 10**(snr_db/10)
    N_total = N_cells * N_bs_per_cell

    # Non-cooperative: interference limited
    sinr_no_comp = snr / N_cells   # roughly
    C_no_comp    = np.log2(1 + sinr_no_comp)

    # CoMP JT: interference becomes signal (all BSs cooperate)
    # Effective: N_total×1 MISO → beamforming gain = N_total
    sinr_comp = snr * N_total
    C_comp    = np.log2(1 + sinr_comp)

    return C_no_comp, C_comp


# ─────────────────────────────────────────────
# 4. NB-IoT / eMTC LINK BUDGET
# ─────────────────────────────────────────────

IoT_MODES = {
    'LTE-M (eMTC)': {
        'bw_khz':      1400,
        'max_coupling_db': 156,
        'peak_dl_kbps':    4000,
        'peak_ul_kbps':    7000,
        'duplex':          'FDD',
        'repetitions':     2048,
        'use_case':        'Wearables, smart meters',
    },
    'NB-IoT': {
        'bw_khz':      180,
        'max_coupling_db': 164,
        'peak_dl_kbps':    250,
        'peak_ul_kbps':    250,
        'duplex':          'HD-FDD',
        'repetitions':     2048,
        'use_case':        'Deep indoor sensors, water meters',
    },
    'eMTC Cat-M2': {
        'bw_khz':      5000,
        'max_coupling_db': 154,
        'peak_dl_kbps':   4000,
        'peak_ul_kbps':   7000,
        'duplex':          'FDD',
        'repetitions':     32,
        'use_case':        'Industrial IoT',
    },
}


def coverage_extension_gain(base_snr_db=-3, repetitions=2048):
    """Coverage extension through repetition combining (gain ≈ 10·log10(N_rep))."""
    return 10 * np.log10(repetitions)


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  LTE-Advanced Features (3GPP Release 10–13)")
    print("=" * 65)

    DARK = '#0d1117'; BG = '#161b22'
    GRD  = dict(alpha=0.15, color='white')

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    def dark_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(BG)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.grid(**GRD)
        return ax

    # ── 1. Carrier Aggregation spectrum ──────────────────────
    ax1 = dark_ax(gs[0, 0])

    ca_configs = [
        {'name':'3CA Intra-Contiguous\n(B1: 2100 MHz)', 'ccs':[(2090,20),(2110,20),(2130,20)]},
        {'name':'2CA Inter-Band\n(B1+B7)', 'ccs':[(2110,20),(2660,20)]},
        {'name':'5CA Inter-Band\n(B3+B7+B20)', 'ccs':[(1820,20),(2660,20),(800,10),(1840,10),(2680,20)]},
    ]

    for row_i, cfg in enumerate(ca_configs):
        y_base = row_i * 1.5
        ca = CarrierAggregator()
        for fc, bw in cfg['ccs']:
            ca.add_carrier(fc, bw)
        colors_ca = ['#58a6ff','#3fb950','#e3b341','#f85149','#d2a8ff']
        for ci, (fc, bw) in enumerate(cfg['ccs']):
            ax1.barh(y_base, bw, left=fc-bw/2,
                     height=0.8, color=colors_ca[ci], alpha=0.85, edgecolor='#0d1117', lw=0.5)
            ax1.text(fc, y_base+0.4, f'{bw}M', ha='center', va='bottom', color='white', fontsize=6)
        ax1.text(0.02, y_base+0.4, f"{cfg['name']}\n{ca.total_bandwidth()}MHz → {ca.peak_tput():.0f}Mbps",
                 transform=ax1.get_yaxis_transform(), color='#8b949e', fontsize=7)

    ax1.set_xlabel('Frequency (MHz)', color='#8b949e')
    ax1.set_title('LTE-A Carrier Aggregation\nSpectrum Configurations', color='#58a6ff', fontweight='bold')
    ax1.set_yticks([]); ax1.set_xlim(700, 2800)

    # ── 2. HetNet deployment + SINR map ──────────────────────
    ax2 = dark_ax(gs[0, 1])
    cells = [
        Cell('macro',    0,    0, 0),
        Cell('pico',   200,  150, 1),
        Cell('pico',  -250, -100, 2),
        Cell('femto',  300, -250, 3),
        Cell('femto', -150,  200, 4),
    ]
    xx, yy, sinr = sinr_map_hetnet(cells, grid_size=800, resolution=60, fc_ghz=2.1)
    im = ax2.pcolormesh(xx, yy, sinr, cmap='RdYlGn', vmin=-10, vmax=30, shading='auto')
    plt.colorbar(im, ax=ax2, label='SINR (dB)').ax.tick_params(colors='#8b949e')
    for cell in cells:
        m = {'macro':'*','pico':'o','femto':'^'}[cell.cell_type]
        s = {'macro':200,'pico':80,'femto':50}[cell.cell_type]
        ax2.scatter(*cell.pos, marker=m, s=s, color=cell.color,
                    edgecolors='white', lw=0.8, zorder=5)
    legend_patches = [
        mpatches.Patch(color='#58a6ff', label='Macro (46 dBm)'),
        mpatches.Patch(color='#3fb950', label='Pico (30 dBm)'),
        mpatches.Patch(color='#f85149', label='Femto (20 dBm)'),
    ]
    ax2.legend(handles=legend_patches, fontsize=7, facecolor=BG, labelcolor='white')
    ax2.set_title('HetNet SINR Map (UMa NLOS, 2.1GHz)\nMacro + Pico + Femto', color='#58a6ff', fontweight='bold')
    ax2.set_xlabel('x (m)', color='#8b949e'); ax2.set_ylabel('y (m)', color='#8b949e')

    # ── 3. CA throughput scaling ──────────────────────────────
    ax3 = dark_ax(gs[0, 2])
    n_cc_range = range(1, 6)
    bw_per_cc  = 20
    for layers, col, ls in [(1,'#58a6ff','-'),(2,'#3fb950','--'),(4,'#f85149',':')]:
        tputs = []
        for n_cc in n_cc_range:
            ca2 = CarrierAggregator()
            for k in range(n_cc):
                ca2.add_carrier(2100+k*20, bw_per_cc, cc_id=k)
            tputs.append(ca2.peak_tput(layers=layers))
        ax3.plot(list(n_cc_range), tputs, 'o-', color=col, lw=2, ms=5, ls=ls, label=f'{layers} layers')
    ax3.set_xlabel('Number of Component Carriers', color='#8b949e')
    ax3.set_ylabel('Peak DL Throughput (Mbps)', color='#8b949e')
    ax3.set_title('LTE-A CA Throughput Scaling\n(20MHz per CC, 64QAM)', color='#58a6ff', fontweight='bold')
    ax3.legend(fontsize=8, facecolor=BG, labelcolor='white')
    ax3.set_xticks(list(n_cc_range))

    # ── 4. CoMP gain ──────────────────────────────────────────
    ax4 = dark_ax(gs[1, 0])
    snr_range = np.arange(-5, 25)
    for N_cells_c, col in [(1,'#8b949e'),(2,'#58a6ff'),(3,'#3fb950'),(4,'#f85149')]:
        caps_no = []; caps_jt = []
        for snr in snr_range:
            c_no, c_jt = comp_jt_gain(N_cells_c, 8, snr)
            caps_no.append(c_no); caps_jt.append(c_jt)
        lbl = f'{N_cells_c} cells'
        ax4.plot(snr_range, caps_jt, color=col, lw=1.8, label=f'CoMP JT {lbl}')
        if N_cells_c == 1:
            ax4.plot(snr_range, caps_no, color=col, lw=1.5, ls='--', label='No CoMP')
    ax4.set_xlabel('SNR (dB)', color='#8b949e')
    ax4.set_ylabel('Capacity (bits/s/Hz)', color='#8b949e')
    ax4.set_title('CoMP Joint Transmission Gain\nvs Non-Cooperative', color='#58a6ff', fontweight='bold')
    ax4.legend(fontsize=7, facecolor=BG, labelcolor='white')

    # ── 5. NB-IoT / eMTC comparison ──────────────────────────
    ax5 = dark_ax(gs[1, 1])
    modes = list(IoT_MODES.keys())
    x     = np.arange(len(modes))
    mcl   = [IoT_MODES[m]['max_coupling_db'] for m in modes]
    bw    = [IoT_MODES[m]['bw_khz'] / 1000 for m in modes]
    tput  = [IoT_MODES[m]['peak_dl_kbps'] / 1000 for m in modes]

    bars = ax5.bar(x, mcl, color=['#58a6ff','#3fb950','#e3b341'],
                   edgecolor='#30363d', lw=0.5, width=0.5)
    ax5.bar_label(bars, fmt='%.0f dB', padding=3, color='white', fontsize=8)
    ax5.set_xticks(x); ax5.set_xticklabels(modes, fontsize=7, color='#8b949e')
    ax5.set_ylabel('Max Coupling Loss (dB)', color='#8b949e')
    ax5.set_title('NB-IoT vs LTE-M Coverage\n(Max Coupling Loss)', color='#58a6ff', fontweight='bold')
    ax5.set_ylim([140, 172])

    ax5b = ax5.twinx()
    ax5b.plot(x, tput, 'o--', color='#d2a8ff', ms=8, lw=1.5, label='Peak DL (Mbps)')
    ax5b.set_ylabel('Peak DL (Mbps)', color='#d2a8ff')
    ax5b.tick_params(colors='#d2a8ff', labelsize=8)

    # ── 6. Coverage extension via repetition ─────────────────
    ax6 = dark_ax(gs[1, 2])
    reps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    gains = [coverage_extension_gain(repetitions=r) for r in reps]
    ax6.semilogx(reps, gains, 'o-', color='#3fb950', lw=2, ms=5)
    ax6.axhline(coverage_extension_gain(repetitions=2048), color='#f85149',
                ls='--', lw=1, label='NB-IoT max (+33dB)')
    ax6.axhline(0, color='white', lw=0.4, ls=':', alpha=0.3)
    ax6.fill_between(reps, 0, gains, alpha=0.15, color='#3fb950')
    ax6.set_xlabel('Repetitions (N)', color='#8b949e')
    ax6.set_ylabel('Coverage Extension Gain (dB)', color='#8b949e')
    ax6.set_title('IoT Coverage Extension\nRepetition Combining Gain', color='#58a6ff', fontweight='bold')
    ax6.legend(fontsize=8, facecolor=BG, labelcolor='white')

    fig.text(0.5, 0.98, 'LTE-Advanced — Carrier Aggregation, HetNet, CoMP, NB-IoT',
             ha='center', color='white', fontsize=14, fontweight='bold')
    plt.savefig('lte_advanced.png', dpi=130, bbox_inches='tight', facecolor=DARK)
    print("\n✓ Saved: lte_advanced.png")

    # Print CA report
    ca_demo = CarrierAggregator()
    for fc, bw in [(1820,20),(2660,20),(800,10)]:
        ca_demo.add_carrier(fc, bw)
    ca_demo.report()

    # IoT summary
    print("\n  IoT Technology Comparison:")
    print(f"  {'Mode':>20}  {'MCL(dB)':>8}  {'BW':>8}  {'Peak DL':>10}  {'Use Case'}")
    print("  " + "-"*75)
    for name, params in IoT_MODES.items():
        print(f"  {name:>20}  {params['max_coupling_db']:>8}  "
              f"{params['bw_khz']:>5}kHz  {params['peak_dl_kbps']:>7}kbps  {params['use_case']}")

    plt.close('all')
    print("\n✅  LTE-Advanced module demo complete.")
