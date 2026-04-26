"""
src/channel_models/propagation.py
===================================
3GPP Channel Models for LTE & 5G NR
Covers: AWGN, EPA, EVA, ETU (LTE), CDL-A/B/C/D/E (NR), TDL,
        Path loss (UMa, UMi, RMa, InH), Shadowing, Doppler.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp

# ─────────────────────────────────────────────
# 1. LTE CHANNEL MODELS (3GPP TR 36.104)
# ─────────────────────────────────────────────

# (delay_ns, power_dB, Doppler_Hz) per tap
LTE_CHANNELS = {
    'EPA': {   # Extended Pedestrian A — low Doppler (5 Hz)
        'fd_hz': 5,
        'taps': [
            {'delay_ns':   0, 'power_db':  0.0},
            {'delay_ns':  30, 'power_db': -1.0},
            {'delay_ns':  70, 'power_db': -2.0},
            {'delay_ns':  90, 'power_db': -3.0},
            {'delay_ns': 110, 'power_db': -8.0},
            {'delay_ns': 190, 'power_db':-17.2},
            {'delay_ns': 410, 'power_db':-20.8},
        ]
    },
    'EVA': {   # Extended Vehicular A — medium Doppler (70 Hz)
        'fd_hz': 70,
        'taps': [
            {'delay_ns':   0, 'power_db':  0.0},
            {'delay_ns':  30, 'power_db': -1.5},
            {'delay_ns': 150, 'power_db': -1.4},
            {'delay_ns': 310, 'power_db': -3.6},
            {'delay_ns': 370, 'power_db': -0.6},
            {'delay_ns': 710, 'power_db': -9.1},
            {'delay_ns':1090, 'power_db': -7.0},
            {'delay_ns':1730, 'power_db':-12.0},
            {'delay_ns':2510, 'power_db':-16.9},
        ]
    },
    'ETU': {   # Extended Typical Urban — high Doppler (300 Hz)
        'fd_hz': 300,
        'taps': [
            {'delay_ns':   0, 'power_db': -1.0},
            {'delay_ns':  50, 'power_db': -1.0},
            {'delay_ns': 120, 'power_db': -1.0},
            {'delay_ns': 200, 'power_db':  0.0},
            {'delay_ns': 230, 'power_db':  0.0},
            {'delay_ns': 500, 'power_db':  0.0},
            {'delay_ns': 600, 'power_db': -3.0},
            {'delay_ns': 900, 'power_db': -5.0},
            {'delay_ns':1200, 'power_db': -7.0},
        ]
    },
}

# ─────────────────────────────────────────────
# 2. 5G NR CDL CHANNEL MODELS (3GPP TR 38.901)
# ─────────────────────────────────────────────

CDL_CHANNELS = {
    'CDL-A': {   # NLOS, indoor/urban, delay spread ~100 ns
        'name': 'CDL-A (NLOS, Cluster Delay Line)',
        'taps': [
            {'delay_ns':   0, 'power_db':  0.0, 'aoa_deg':  52},
            {'delay_ns':  30, 'power_db': -4.8, 'aoa_deg': -54},
            {'delay_ns':  75, 'power_db': -6.4, 'aoa_deg':  57},
            {'delay_ns': 110, 'power_db': -9.5, 'aoa_deg': -84},
            {'delay_ns': 195, 'power_db':-12.3, 'aoa_deg': -45},
            {'delay_ns': 290, 'power_db':-12.9, 'aoa_deg':  81},
        ]
    },
    'CDL-B': {   # NLOS, urban, delay spread ~200 ns
        'name': 'CDL-B (NLOS, Urban)',
        'taps': [
            {'delay_ns':   0, 'power_db':  0.0, 'aoa_deg':  -2},
            {'delay_ns':  10, 'power_db': -2.2, 'aoa_deg': -70},
            {'delay_ns':  20, 'power_db': -4.0, 'aoa_deg': -83},
            {'delay_ns':  85, 'power_db': -7.6, 'aoa_deg':  89},
            {'delay_ns': 245, 'power_db':-12.0, 'aoa_deg': -45},
            {'delay_ns': 500, 'power_db':-20.0, 'aoa_deg':  -9},
        ]
    },
    'CDL-C': {   # NLOS, suburban, delay spread ~500 ns
        'name': 'CDL-C (NLOS, Suburban)',
        'taps': [
            {'delay_ns':   0, 'power_db':  0.0, 'aoa_deg':  13},
            {'delay_ns':  65, 'power_db': -2.4, 'aoa_deg': -89},
            {'delay_ns': 150, 'power_db': -3.6, 'aoa_deg':  14},
            {'delay_ns': 228, 'power_db': -6.6, 'aoa_deg': -60},
            {'delay_ns': 490, 'power_db':-11.9, 'aoa_deg':  60},
            {'delay_ns': 900, 'power_db':-18.0, 'aoa_deg':  48},
        ]
    },
    'CDL-D': {   # LOS, rural, strong direct path (K=7dB)
        'name': 'CDL-D (LOS, K=7dB)',
        'K_factor_db': 7.0,
        'taps': [
            {'delay_ns':   0, 'power_db':  0.0, 'aoa_deg':   0, 'los': True},
            {'delay_ns':  30, 'power_db': -2.1, 'aoa_deg': -60},
            {'delay_ns': 100, 'power_db': -7.5, 'aoa_deg':  90},
            {'delay_ns': 170, 'power_db':-15.1, 'aoa_deg': -89},
        ]
    },
    'CDL-E': {   # LOS, highway/outdoor, K=22dB (almost flat)
        'name': 'CDL-E (LOS, K=22dB, Highway)',
        'K_factor_db': 22.0,
        'taps': [
            {'delay_ns':   0, 'power_db':  0.0, 'aoa_deg':   0, 'los': True},
            {'delay_ns':  10, 'power_db': -4.9, 'aoa_deg':  40},
            {'delay_ns':  45, 'power_db': -8.0, 'aoa_deg': -87},
        ]
    },
}

# ─────────────────────────────────────────────
# 3. CHANNEL FILTER IMPLEMENTATION
# ─────────────────────────────────────────────

def make_channel_filter(taps_config, fs, rng=None):
    """
    Build a time-domain FIR channel filter from tap config.
    Returns (h, delays_samples, gains)
    """
    rng = rng or np.random.default_rng(0)
    max_delay_ns = max(t['delay_ns'] for t in taps_config)
    max_samp     = int(np.ceil(max_delay_ns * 1e-9 * fs)) + 1
    h = np.zeros(max(max_samp, 1), dtype=complex)

    for tap in taps_config:
        d_samp = int(np.round(tap['delay_ns'] * 1e-9 * fs))
        g_lin  = 10 ** (tap['power_db'] / 20)
        # Rayleigh fading per tap (complex Gaussian)
        is_los = tap.get('los', False)
        if is_los:
            # Rician: deterministic LOS + scattered
            K   = 10 ** (7.0/10)   # K-factor
            los_amp   = np.sqrt(K/(K+1)) * g_lin
            nlos_amp  = np.sqrt(1/(K+1)) * g_lin
            h[d_samp] += los_amp + nlos_amp * (rng.standard_normal() + 1j*rng.standard_normal())/np.sqrt(2)
        else:
            h[d_samp] += g_lin * (rng.standard_normal() + 1j*rng.standard_normal()) / np.sqrt(2)

    # Normalise total power
    p = np.sum(np.abs(h)**2)
    if p > 0:
        h /= np.sqrt(p)
    return h


def apply_channel(signal, h, snr_db=20, rng=None):
    """Apply channel filter h and add AWGN."""
    rng = rng or np.random.default_rng()
    y   = np.convolve(signal, h, mode='full')[:len(signal)]
    P   = np.mean(np.abs(y)**2)
    N0  = P / max(10**(snr_db/10), 1e-10)
    n   = np.sqrt(N0/2) * (rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y)))
    return y + n


def time_varying_channel(signal, taps_config, fs, fd_hz, snr_db=20):
    """
    Time-varying multipath channel with Jakes Doppler spectrum.
    Re-draws tap coefficients every coherence_time samples.
    """
    if fd_hz == 0:
        h = make_channel_filter(taps_config, fs)
        return apply_channel(signal, h, snr_db)

    coh_time  = int(fs / (4 * max(fd_hz, 1)))   # coherence time samples
    N         = len(signal)
    output    = np.zeros(N, dtype=complex)
    rng       = np.random.default_rng(42)

    for start in range(0, N, coh_time):
        end = min(start + coh_time, N)
        h   = make_channel_filter(taps_config, fs, rng)
        seg = signal[start:end]
        y   = np.convolve(seg, h, mode='full')[:len(seg)]
        output[start:end] = y

    P  = np.mean(np.abs(output)**2)
    N0 = P / max(10**(snr_db/10), 1e-10)
    output += np.sqrt(N0/2) * (rng.standard_normal(N) + 1j*rng.standard_normal(N))
    return output


# ─────────────────────────────────────────────
# 4. PATH LOSS MODELS (3GPP TR 38.901)
# ─────────────────────────────────────────────

def path_loss_uma_nlos(d_m, fc_ghz, h_bs=25, h_ut=1.5):
    """
    UMa (Urban Macro) NLOS path loss.
    d_m: distance in metres, fc_ghz: frequency in GHz
    Returns PL in dB.
    """
    d     = np.atleast_1d(d_m).astype(float)
    d_3d  = np.sqrt(d**2 + (h_bs - h_ut)**2)
    PL    = 32.4 + 20*np.log10(fc_ghz) + 30*np.log10(d_3d)
    return PL


def path_loss_uma_los(d_m, fc_ghz, h_bs=25, h_ut=1.5):
    """UMa LOS path loss (dual-slope)."""
    d      = np.atleast_1d(d_m).astype(float)
    # Break point distance
    d_bp   = 4 * h_bs * h_ut * fc_ghz * 1e9 / 3e8
    d_3d   = np.sqrt(d**2 + (h_bs - h_ut)**2)

    PL1 = 28.0 + 22*np.log10(d_3d) + 20*np.log10(fc_ghz)
    PL2 = 28.0 + 40*np.log10(d_3d) + 20*np.log10(fc_ghz) \
        - 9*np.log10(d_bp**2 + (h_bs-h_ut)**2)

    return np.where(d_3d < d_bp, PL1, PL2)


def path_loss_umi_nlos(d_m, fc_ghz, h_bs=10, h_ut=1.5):
    """UMi (Urban Micro) NLOS — street canyon."""
    d    = np.atleast_1d(d_m).astype(float)
    d_3d = np.sqrt(d**2 + (h_bs - h_ut)**2)
    return 35.3*np.log10(d_3d) + 22.4 + 21.3*np.log10(fc_ghz) - 0.3*(h_ut-1.5)


def path_loss_rma_los(d_m, fc_ghz, h_bs=35, h_ut=1.5, W=20, h=5):
    """RMa (Rural Macro) LOS path loss."""
    d    = np.atleast_1d(d_m).astype(float)
    d_3d = np.sqrt(d**2 + (h_bs - h_ut)**2)
    A    = 161.04 - 7.1*np.log10(W) + 7.5*np.log10(h) \
           - (24.37 - 3.7*(h/h_bs)**2)*np.log10(h_bs)
    B    = (43.42 - 3.1*np.log10(h_bs)) * (np.log10(d_3d) - 3)
    C    = 20*np.log10(fc_ghz) - (3.2*(np.log10(11.75*h_ut))**2 - 4.97)
    return A + B + C


def shadowing(sigma_db, n_samples):
    """Log-normal shadowing (dB)."""
    return np.random.randn(n_samples) * sigma_db


def link_budget(eirp_dbm, path_loss_db, rx_gain_db=0,
                rx_nf_db=7, bw_hz=100e6, margin_db=10):
    """
    Complete link budget calculation.
    Returns received SNR and link margin.
    """
    k      = 1.38e-23
    T      = 290   # K
    N_dBm  = 10*np.log10(k*T*bw_hz) + 30 + rx_nf_db
    Pr_dBm = eirp_dbm - path_loss_db + rx_gain_db
    SNR_dB = Pr_dBm - N_dBm
    return {
        'Pr_dBm':    Pr_dBm,
        'N_dBm':     N_dBm,
        'SNR_dB':    SNR_dB,
        'margin_dB': SNR_dB - margin_db,
        'viable':    SNR_dB > margin_db,
    }


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)
    fs = 30.72e6   # LTE/NR 20 MHz sample rate

    print("=" * 65)
    print("  3GPP Channel Models — LTE & 5G NR (TR 36.104 / TR 38.901)")
    print("=" * 65)

    DARK = '#0d1117'; BG = '#161b22'
    GRD  = dict(alpha=0.15, color='white')

    def dark_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(BG)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.grid(**GRD)
        return ax

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. LTE power delay profiles ──────────────────────────
    ax1 = dark_ax(gs[0, 0])
    colors_ch = {'EPA':'#58a6ff', 'EVA':'#3fb950', 'ETU':'#f85149'}
    for name, ch in LTE_CHANNELS.items():
        delays = [t['delay_ns'] for t in ch['taps']]
        powers = [t['power_db'] for t in ch['taps']]
        col = colors_ch[name]
        ax1.vlines(delays, -25, powers, colors=col, lw=2, alpha=0.8)
        ax1.plot(delays, powers, 'o', color=col, ms=6, label=f"{name} (fd={ch['fd_hz']}Hz)")
        ax1.axhline(-25, color='#333', lw=0.5)
    ax1.set_xlabel('Delay (ns)', color='#8b949e')
    ax1.set_ylabel('Power (dB)', color='#8b949e')
    ax1.set_title('LTE Channel Models\nEPA / EVA / ETU Power Delay Profile', color='#58a6ff', fontweight='bold')
    ax1.legend(fontsize=8, facecolor=BG, labelcolor='white')

    # ── 2. NR CDL power delay profiles ───────────────────────
    ax2 = dark_ax(gs[0, 1])
    cdl_colors = {'CDL-A':'#58a6ff','CDL-B':'#3fb950','CDL-C':'#e3b341',
                  'CDL-D':'#f85149','CDL-E':'#d2a8ff'}
    for name, ch in CDL_CHANNELS.items():
        delays = [t['delay_ns'] for t in ch['taps']]
        powers = [t['power_db'] for t in ch['taps']]
        col = cdl_colors[name]
        ax2.vlines(delays, -25, powers, colors=col, lw=2, alpha=0.8)
        ax2.plot(delays, powers, 'o', color=col, ms=5, label=name)
    ax2.set_xlabel('Delay (ns)', color='#8b949e')
    ax2.set_ylabel('Power (dB)', color='#8b949e')
    ax2.set_title('5G NR CDL Channel Models\nPower Delay Profiles (TR 38.901)', color='#58a6ff', fontweight='bold')
    ax2.legend(fontsize=7, facecolor=BG, labelcolor='white')

    # ── 3. Frequency response of LTE channels ────────────────
    ax3 = dark_ax(gs[0, 2])
    test_tx = np.ones(512, dtype=complex)
    for name, ch in LTE_CHANNELS.items():
        h = make_channel_filter(ch['taps'], fs)
        w, H = sp.freqz(h, worN=512, fs=fs/1e6)
        ax3.plot(w, 20*np.log10(np.abs(H)+1e-6),
                 color=colors_ch[name], lw=1.5, label=name)
    ax3.set_xlabel('Frequency (MHz)', color='#8b949e')
    ax3.set_ylabel('|H(f)| (dB)', color='#8b949e')
    ax3.set_title('LTE Channel Frequency Response\n(one realisation per model)', color='#58a6ff', fontweight='bold')
    ax3.legend(fontsize=8, facecolor=BG, labelcolor='white')

    # ── 4. Path loss vs distance ──────────────────────────────
    ax4 = dark_ax(gs[1, 0])
    d_range = np.logspace(1, 4, 200)   # 10m to 10km
    for fc_ghz, col, ls in [(0.7,'#58a6ff','-'),(2.1,'#3fb950','--'),(28,'#f85149',':')]:
        pl_los  = path_loss_uma_los(d_range, fc_ghz)
        pl_nlos = path_loss_uma_nlos(d_range, fc_ghz)
        lbl = f'{fc_ghz}GHz'
        ax4.semilogx(d_range, pl_los,  color=col, lw=1.5, ls='-',  label=f'{lbl} LOS')
        ax4.semilogx(d_range, pl_nlos, color=col, lw=1.5, ls='--', label=f'{lbl} NLOS')
    ax4.set_xlabel('Distance (m)', color='#8b949e')
    ax4.set_ylabel('Path Loss (dB)', color='#8b949e')
    ax4.set_title('UMa Path Loss — 0.7/2.1/28 GHz\n(LOS and NLOS)', color='#58a6ff', fontweight='bold')
    ax4.legend(fontsize=7, facecolor=BG, labelcolor='white', ncol=2)
    ax4.set_ylim([50, 180])

    # ── 5. Coverage comparison (sub-6 vs mmWave) ─────────────
    ax5 = dark_ax(gs[1, 1])
    d_cov = np.linspace(10, 2000, 300)
    eirp  = 46   # dBm (typical gNB)
    rx_g  = 15   # dB (UE antenna gain)
    nf    = 7    # dB
    for fc_ghz, bw_hz, col, lab in [
        (3.5, 100e6, '#58a6ff', '5G NR 3.5GHz 100MHz'),
        (28,   400e6, '#f85149', '5G NR 28GHz 400MHz'),
        (0.8,   10e6, '#3fb950', 'LTE 800MHz 10MHz'),
    ]:
        pl   = path_loss_uma_nlos(d_cov, fc_ghz)
        snrs = [link_budget(eirp, p, rx_g, nf, bw_hz)['SNR_dB'] for p in pl]
        ax5.plot(d_cov, snrs, color=col, lw=1.8, label=lab)
    ax5.axhline(3,  color='white', lw=0.6, ls=':', alpha=0.5, label='QPSK threshold (~3dB)')
    ax5.axhline(18, color='white', lw=0.6, ls='--', alpha=0.5, label='64QAM threshold (~18dB)')
    ax5.set_xlabel('Distance (m)', color='#8b949e')
    ax5.set_ylabel('Received SNR (dB)', color='#8b949e')
    ax5.set_title('Coverage Comparison\nLTE vs 5G NR sub-6 vs mmWave', color='#58a6ff', fontweight='bold')
    ax5.legend(fontsize=7, facecolor=BG, labelcolor='white')
    ax5.set_ylim([-10, 60])

    # ── 6. Delay spread / coherence bandwidth ────────────────
    ax6 = dark_ax(gs[1, 2])
    all_channels = {**{f'LTE_{k}': v['taps'] for k,v in LTE_CHANNELS.items()},
                    **{k: v['taps'] for k,v in CDL_CHANNELS.items()}}
    names_all, rms_ds, Bc_all = [], [], []
    for name, taps in all_channels.items():
        delays = np.array([t['delay_ns'] for t in taps])
        powers = 10**np.array([t['power_db']/10 for t in taps])
        powers /= powers.sum()
        mean_d = np.sum(delays * powers)
        rms    = np.sqrt(np.sum(powers * (delays - mean_d)**2))
        rms_ds.append(rms)
        Bc_all.append(1/(5*max(rms*1e-9, 1e-12)) / 1e6)   # MHz
        names_all.append(name.replace('LTE_','LTE\n').replace('CDL-','CDL-'))

    x_pos = np.arange(len(names_all))
    bars = ax6.bar(x_pos, rms_ds,
                   color=['#58a6ff']*3 + ['#3fb950']*5,
                   edgecolor='#30363d', lw=0.5)
    ax6.set_xticks(x_pos); ax6.set_xticklabels(names_all, fontsize=7, color='#8b949e')
    ax6.set_ylabel('RMS Delay Spread (ns)', color='#8b949e')
    ax6.set_title('RMS Delay Spread\nLTE (blue) vs NR CDL (green)', color='#58a6ff', fontweight='bold')

    ax6b = ax6.twinx()
    ax6b.plot(x_pos, Bc_all, 'o--', color='#e3b341', ms=5, lw=1.5, label='Coherence BW (MHz)')
    ax6b.set_ylabel('Coherence BW (MHz)', color='#e3b341')
    ax6b.tick_params(colors='#e3b341', labelsize=8)
    ax6b.legend(fontsize=7, facecolor=BG, labelcolor='white', loc='upper right')

    fig.text(0.5, 0.98, '3GPP Channel Models — LTE (EPA/EVA/ETU) & 5G NR (CDL-A/B/C/D/E)',
             ha='center', color='white', fontsize=14, fontweight='bold')
    plt.savefig('channel_models.png', dpi=130, bbox_inches='tight', facecolor=DARK)
    print("\n✓ Saved: channel_models.png")

    # ── Link budget printout ─────────────────────────────────
    print("\n  Link Budget — 5G NR gNB (EIRP=46dBm, UE @500m):")
    for fc, bw, label in [(3.5,100e6,'NR 3.5GHz'),(28,400e6,'NR 28GHz')]:
        pl = float(path_loss_uma_nlos(np.array([500]), fc)[0])
        lb = link_budget(46, pl, 15, 7, bw)
        print(f"  {label:>12}: PL={pl:.1f}dB, Pr={lb['Pr_dBm']:.1f}dBm, "
              f"SNR={lb['SNR_dB']:.1f}dB {'✓' if lb['viable'] else '✗'}")

    plt.close('all')
    print("\n✅  Channel Models demo complete.")
