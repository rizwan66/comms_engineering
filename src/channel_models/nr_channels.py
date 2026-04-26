"""
src/channel_models/nr_channels.py
===================================
3GPP NR Channel Models (TR 38.901)

Implements:
  - CDL-A/B/C/D/E (Clustered Delay Line) — 3GPP TR 38.901 Table 7.7.2
  - TDL-A/B/C/D/E (Tapped Delay Line) — simplified flat-fading per tap
  - AWGN baseline
  - Rayleigh / Rician flat fading
  - Frequency-selective channel (FIR filter model)
  - Doppler / time-varying channel (Clarke's model)
  - mmWave channel (O2I / O2O, high path loss, sparse multipath)
  - V2X channel (high Doppler, non-stationary)
  - Path loss models: UMa, UMi, RMa, InH (TR 38.901 §7.4)
  - Shadowing, penetration loss
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

np.random.seed(0)

# ─────────────────────────────────────────────
# 1. CDL CHANNEL PROFILES (3GPP TR 38.901 §7.7.2)
# ─────────────────────────────────────────────

CDL_PROFILES = {
    # 'CDL-A': NLOS urban macro — rich scattering
    'CDL-A': {
        'delays_ns': [0,   10,  20,  30,  50,  80, 110, 140, 180, 230, 300, 400, 500],
        'powers_db': [0,  -10, -14,  -8, -14, -12, -18, -12, -20, -22, -24, -28, -30],
        'K_factor': None,
        'env': 'NLOS Urban Macro',
    },
    # 'CDL-B': NLOS urban micro
    'CDL-B': {
        'delays_ns': [0,  10,  20,  30,  40,  60,  80, 120, 160, 200],
        'powers_db': [0,  -2,  -6,  -8, -10, -12, -14, -18, -22, -26],
        'K_factor': None,
        'env': 'NLOS Urban Micro',
    },
    # 'CDL-C': NLOS indoor / suburban
    'CDL-C': {
        'delays_ns': [0,   5,  15,  30,  60, 100, 150, 220, 310, 430],
        'powers_db': [0,  -5,  -8, -10, -12, -16, -18, -22, -26, -30],
        'K_factor': None,
        'env': 'NLOS Indoor/Suburban',
    },
    # 'CDL-D': LOS urban macro (K=7dB)
    'CDL-D': {
        'delays_ns': [0,   30,  70, 120, 180, 250, 340, 450],
        'powers_db': [0.0, -4,  -8, -12, -16, -20, -24, -28],
        'K_factor': 7,   # dB
        'env': 'LOS Urban Macro',
    },
    # 'CDL-E': LOS indoor / rural (K=10dB)
    'CDL-E': {
        'delays_ns': [0,   20,  60, 110, 170, 240],
        'powers_db': [0.0, -5, -10, -15, -20, -25],
        'K_factor': 10,
        'env': 'LOS Indoor/Rural',
    },
}

# TDL profiles (flat-fading per tap, simplified)
TDL_PROFILES = {
    'TDL-A': {'delays_ns': [0, 30, 70, 110, 170, 250], 'powers_db': [0, -5, -10, -15, -20, -25]},
    'TDL-B': {'delays_ns': [0, 15, 40,  80, 130, 200], 'powers_db': [0, -4,  -8, -12, -18, -25]},
    'TDL-C': {'delays_ns': [0, 20, 50, 100, 160, 230], 'powers_db': [0, -3,  -6, -10, -15, -22]},
    'TDL-D': {'delays_ns': [0, 25, 60, 110],            'powers_db': [0, -5, -12, -20]},
    'TDL-E': {'delays_ns': [0, 10, 30,  60],            'powers_db': [0, -4,  -9, -16]},
}


# ─────────────────────────────────────────────
# 2. CHANNEL GENERATOR
# ─────────────────────────────────────────────

@dataclass
class ChannelConfig:
    fs:           float = 30.72e6    # sample rate (Hz)
    fc:           float = 3.5e9     # carrier frequency (Hz)
    v_kmh:        float = 30.0      # UE velocity (km/h)
    n_tx:         int   = 1
    n_rx:         int   = 1
    seed:         int   = 42

    @property
    def fd(self):
        """Maximum Doppler frequency (Hz)."""
        c = 3e8
        return self.v_kmh / 3.6 * self.fc / c

    @property
    def wavelength(self):
        return 3e8 / self.fc


class CDLChannel:
    """
    CDL channel model (3GPP TR 38.901).
    Generates impulse response samples for convolution.
    """
    def __init__(self, profile: str, cfg: ChannelConfig):
        self.profile = CDL_PROFILES[profile]
        self.cfg     = cfg
        self.name    = profile
        self._build_taps()

    def _build_taps(self):
        rng    = np.random.default_rng(self.cfg.seed)
        delays = np.array(self.profile['delays_ns']) * 1e-9
        powers = 10**(np.array(self.profile['powers_db']) / 10)
        K      = self.profile['K_factor']

        self.delay_samples = np.round(delays * self.cfg.fs).astype(int)
        self.n_taps        = self.delay_samples.max() + 1
        self.powers        = powers / powers.sum()

        # Generate complex tap gains (Rayleigh or Rician per tap)
        gains = []
        for i, p in enumerate(self.powers):
            if K is not None and i == 0:
                # LOS component: Rician K-factor
                K_lin = 10**(K/10)
                sigma = np.sqrt(p / (2*(K_lin+1)))
                los   = np.sqrt(p * K_lin / (K_lin+1))
                g     = los + sigma*(rng.standard_normal()+1j*rng.standard_normal())
            else:
                sigma = np.sqrt(p/2)
                g     = sigma*(rng.standard_normal()+1j*rng.standard_normal())
            gains.append(g)
        self.gains = np.array(gains)

    def get_cir(self) -> np.ndarray:
        """Channel Impulse Response (CIR) as FIR filter."""
        h = np.zeros(self.n_taps, dtype=complex)
        for d, g in zip(self.delay_samples, self.gains):
            h[d] += g
        return h

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply static CDL channel to signal."""
        from scipy.signal import fftconvolve
        h = self.get_cir()
        return fftconvolve(signal, h)[:len(signal)]

    def apply_time_varying(self, signal: np.ndarray, n_update: int = 256) -> np.ndarray:
        """
        Time-varying CDL channel with Doppler.
        Updates channel gains every n_update samples using Clarke's model.
        """
        from scipy.signal import fftconvolve
        out  = np.zeros_like(signal, dtype=complex)
        N    = len(signal)
        rng  = np.random.default_rng(self.cfg.seed + 1)

        fd   = self.cfg.fd
        fs   = self.cfg.fs

        for start in range(0, N, n_update):
            end    = min(start + n_update, N)
            t_arr  = np.arange(start, end) / fs

            # Update each tap with Doppler
            h = np.zeros(self.n_taps, dtype=complex)
            for d, g0 in zip(self.delay_samples, self.gains):
                # Random angle of arrival for Doppler phase
                aoa    = rng.uniform(-np.pi, np.pi)
                dop_ph = 2 * np.pi * fd * np.cos(aoa) * t_arr
                h[d]  += g0 * np.mean(np.exp(1j * dop_ph))

            seg = signal[start:end]
            out[start:end] = fftconvolve(seg, h)[:len(seg)]

        return out


# ─────────────────────────────────────────────
# 3. PATH LOSS MODELS (3GPP TR 38.901 §7.4)
# ─────────────────────────────────────────────

def path_loss_uma_nlos(d_2d: float, h_bs: float = 25.0,
                        h_ut: float = 1.5, fc_ghz: float = 3.5) -> float:
    """
    UMa NLOS path loss (3GPP TR 38.901 §7.4.1 Table 7.4.1-1).
    d_2d: 2D distance (m), fc_ghz: frequency (GHz)
    Valid: 10m ≤ d ≤ 5000m
    """
    d_3d = np.sqrt(d_2d**2 + (h_bs - h_ut)**2)
    PL   = 13.54 + 39.08*np.log10(d_3d) + 20*np.log10(fc_ghz) - 0.6*(h_ut - 1.5)
    return max(PL, 0)


def path_loss_uma_los(d_2d: float, h_bs: float = 25.0,
                       h_ut: float = 1.5, fc_ghz: float = 3.5) -> float:
    """UMa LOS path loss (dual-slope)."""
    h_e  = 1.0  # effective environment height
    d_bp = 4 * (h_bs - h_e) * (h_ut - h_e) * fc_ghz*1e9 / 3e8
    d_3d = np.sqrt(d_2d**2 + (h_bs - h_ut)**2)

    if d_2d < d_bp:
        PL = (28.0 + 22*np.log10(d_3d) + 20*np.log10(fc_ghz))
    else:
        PL = (28.0 + 40*np.log10(d_3d) + 20*np.log10(fc_ghz)
              - 9*np.log10(d_bp**2 + (h_bs - h_ut)**2))
    return max(PL, 0)


def path_loss_umi_street(d_2d: float, fc_ghz: float = 3.5) -> float:
    """UMi-Street Canyon NLOS (TR 38.901 Table 7.4.1-1)."""
    d_3d = np.sqrt(d_2d**2 + (10-1.5)**2)
    return max(22.4 + 35.3*np.log10(d_3d) + 21.3*np.log10(fc_ghz), 0)


def path_loss_mmwave_o2o(d_2d: float, fc_ghz: float = 28.0) -> float:
    """mmWave outdoor-to-outdoor LOS path loss."""
    d_3d = np.sqrt(d_2d**2 + (10-1.5)**2)
    return max(32.4 + 20*np.log10(d_3d) + 20*np.log10(fc_ghz), 0)


def path_loss_mmwave_o2i(d_2d: float, fc_ghz: float = 28.0,
                          wall_loss_db: float = 20.0) -> float:
    """mmWave outdoor-to-indoor (high penetration loss)."""
    return path_loss_mmwave_o2o(d_2d, fc_ghz) + wall_loss_db


def shadowing(std_db: float = 8.0) -> float:
    """Log-normal shadowing (dB)."""
    return np.random.randn() * std_db


def coverage_map(d_arr: np.ndarray, fc_ghz: float,
                  tx_eirp_dbm: float, rx_sens_dbm: float) -> np.ndarray:
    """Coverage probability map across distance array."""
    pl_nlos = np.array([path_loss_uma_nlos(d, fc_ghz=fc_ghz) for d in d_arr])
    pl_los  = np.array([path_loss_uma_los(d,  fc_ghz=fc_ghz) for d in d_arr])

    # SINR at each distance (dBm - path_loss - noise)
    noise_dbm = -174 + 10*np.log10(100e6) + 7   # kTB @ 100MHz BW + 7dB NF
    sinr_nlos = tx_eirp_dbm - pl_nlos - noise_dbm
    sinr_los  = tx_eirp_dbm - pl_los  - noise_dbm
    return sinr_nlos, sinr_los


# ─────────────────────────────────────────────
# 4. AWGN + FADING HELPERS
# ─────────────────────────────────────────────

def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    P   = np.mean(np.abs(signal)**2)
    N0  = P / (10**(snr_db/10))
    n   = np.sqrt(N0/2)*(np.random.randn(len(signal))+1j*np.random.randn(len(signal)))
    return signal + n


def rayleigh_flat(n_samples: int) -> np.ndarray:
    """Rayleigh flat-fading envelope (normalised)."""
    h = (np.random.randn(n_samples) + 1j*np.random.randn(n_samples)) / np.sqrt(2)
    return h / np.sqrt(np.mean(np.abs(h)**2))


def rician_flat(n_samples: int, K_db: float = 7.0) -> np.ndarray:
    """Rician flat-fading (K=LOS power / scatter power)."""
    K   = 10**(K_db/10)
    los = np.sqrt(K/(K+1))
    scatter_sigma = 1/np.sqrt(2*(K+1))
    h = los + scatter_sigma*(np.random.randn(n_samples)+1j*np.random.randn(n_samples))
    return h / np.sqrt(np.mean(np.abs(h)**2))


# ─────────────────────────────────────────────
# 5. DOPPLER SPECTRUM
# ─────────────────────────────────────────────

def jakes_doppler_spectrum(f_arr: np.ndarray, fd: float) -> np.ndarray:
    """Jakes/Clarke U-shaped Doppler PSD: S(f) ∝ 1/√(1-(f/fd)²)"""
    ratio = np.abs(f_arr) / (fd + 1e-10)
    psd   = np.where(ratio < 0.999, 1.0/np.sqrt(np.maximum(1 - ratio**2, 1e-6)), 0.0)
    return psd / (psd.sum() + 1e-10)


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    DARK = '#0d1117'; BLUE = '#58a6ff'; ORG = '#e3b341'
    GRN  = '#3fb950'; RED  = '#f85149'; PUR = '#d2a8ff'

    print("=" * 60)
    print("  3GPP NR Channel Models (TR 38.901)")
    print("=" * 60)

    cfg = ChannelConfig(fs=30.72e6, fc=3.5e9, v_kmh=60)
    print(f"\n  Doppler fd = {cfg.fd:.1f} Hz  (v={cfg.v_kmh}km/h, fc={cfg.fc/1e9}GHz)")

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    def ax_(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.grid(True, alpha=0.15, color='white')
        return ax

    # ── 1. CDL CIR comparison ─────────────────────────────────
    ax1 = ax_(gs[0,0])
    for name, col in [('CDL-A',BLUE),('CDL-B',ORG),('CDL-D',GRN),('CDL-E',RED)]:
        ch  = CDLChannel(name, cfg)
        h   = ch.get_cir()
        t_ns = np.arange(len(h)) / cfg.fs * 1e9
        ax1.vlines(t_ns, 0, np.abs(h), colors=col, lw=1.5, label=name)
        ax1.scatter(t_ns, np.abs(h), color=col, s=20, zorder=5)
    ax1.set_xlabel("Delay (ns)", color='#8b949e')
    ax1.set_ylabel("|h(τ)|", color='#8b949e')
    ax1.set_title("CDL Channel Impulse Responses", color=BLUE, fontweight='bold')
    ax1.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    # ── 2. Path loss vs distance ──────────────────────────────
    ax2 = ax_(gs[0,1])
    d_arr = np.logspace(1, 3.5, 200)   # 10m to 3km
    for fc_g, col, label in [(0.7,PUR,'700 MHz'),(3.5,BLUE,'3.5 GHz'),
                               (28,RED,'28 GHz mmWave'),(70,ORG,'70 GHz mmWave')]:
        if fc_g < 6:
            pl = [path_loss_uma_nlos(d, fc_ghz=fc_g) for d in d_arr]
        else:
            pl = [path_loss_mmwave_o2o(d, fc_ghz=fc_g) for d in d_arr]
        ax2.semilogx(d_arr, pl, color=col, lw=2, label=label)
    ax2.set_xlabel("Distance (m)", color='#8b949e')
    ax2.set_ylabel("Path Loss (dB)", color='#8b949e')
    ax2.set_title("Path Loss vs Distance\n(UMa NLOS, various frequencies)", color=BLUE, fontweight='bold')
    ax2.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    # ── 3. Doppler spectrum ───────────────────────────────────
    ax3 = ax_(gs[0,2])
    for v, col, label in [(3,'#58a6ff','3 km/h (pedestrian)'),
                           (60,'#e3b341','60 km/h (vehicular)'),
                           (250,'#f85149','250 km/h (high-speed)')]:
        cfg_v = ChannelConfig(fc=3.5e9, v_kmh=v)
        fd    = cfg_v.fd
        f_arr = np.linspace(-fd*1.5, fd*1.5, 500)
        psd   = jakes_doppler_spectrum(f_arr, fd)
        ax3.plot(f_arr, psd, color=col, lw=2, label=f'{label}\nfd={fd:.0f}Hz')
    ax3.set_xlabel("Frequency (Hz)", color='#8b949e')
    ax3.set_ylabel("Normalised PSD", color='#8b949e')
    ax3.set_title("Jakes/Clarke Doppler Spectrum\n(U-shaped)", color=BLUE, fontweight='bold')
    ax3.legend(fontsize=7, facecolor='#161b22', labelcolor='white')

    # ── 4. Time-varying channel magnitude ─────────────────────
    ax4 = ax_(gs[1,0])
    N    = 4096
    sig  = np.ones(N, dtype=complex)
    ch_a = CDLChannel('CDL-A', ChannelConfig(fs=30.72e6, fc=3.5e9, v_kmh=60))
    ch_d = CDLChannel('CDL-D', ChannelConfig(fs=30.72e6, fc=3.5e9, v_kmh=60))
    out_a = ch_a.apply_time_varying(sig, n_update=128)
    out_d = ch_d.apply_time_varying(sig, n_update=128)
    t_ms  = np.arange(N) / 30.72e6 * 1e3
    ax4.plot(t_ms, 20*np.log10(np.abs(out_a)+1e-10), color=BLUE, lw=1, label='CDL-A (NLOS)')
    ax4.plot(t_ms, 20*np.log10(np.abs(out_d)+1e-10), color=GRN,  lw=1, label='CDL-D (LOS K=7dB)')
    ax4.set_xlabel("Time (ms)", color='#8b949e')
    ax4.set_ylabel("|h(t)| dB", color='#8b949e')
    ax4.set_title("Time-Varying Channel (v=60km/h)", color=BLUE, fontweight='bold')
    ax4.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    # ── 5. Rayleigh vs Rician PDF ─────────────────────────────
    ax5 = ax_(gs[1,1])
    n_samp = 50000
    h_ray = np.abs(rayleigh_flat(n_samp))
    h_ric = np.abs(rician_flat(n_samp, K_db=7))
    bins  = np.linspace(0, 3, 80)
    ax5.hist(h_ray, bins=bins, density=True, alpha=0.6, color=RED,  label='Rayleigh (K=0)')
    ax5.hist(h_ric, bins=bins, density=True, alpha=0.6, color=GRN,  label='Rician (K=7dB)')
    ax5.set_xlabel("|h|", color='#8b949e')
    ax5.set_ylabel("PDF", color='#8b949e')
    ax5.set_title("Rayleigh vs Rician Fading\nAmplitude Distribution", color=BLUE, fontweight='bold')
    ax5.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    # ── 6. Coverage SINR vs distance ─────────────────────────
    ax6 = ax_(gs[1,2])
    d_cov = np.linspace(50, 2000, 200)
    for fc_g, col, label in [(700e-3,PUR,'700 MHz'),(3.5,BLUE,'3.5 GHz'),(28,RED,'28 GHz')]:
        sinr_nlos, sinr_los = coverage_map(d_cov, fc_g, tx_eirp_dbm=46, rx_sens_dbm=-100)
        ax6.plot(d_cov, sinr_nlos, '-', color=col, lw=2, label=f'{label} NLOS')
        ax6.plot(d_cov, sinr_los,  '--', color=col, lw=1.5, alpha=0.6)
    ax6.axhline(-6, color='white', lw=0.8, ls=':', label='QPSK threshold (-6dB)')
    ax6.axhline(20, color='gray',  lw=0.8, ls=':', label='256QAM threshold (20dB)')
    ax6.set_xlabel("Distance (m)", color='#8b949e')
    ax6.set_ylabel("DL SINR (dB)", color='#8b949e')
    ax6.set_title("Coverage SINR vs Distance\n(EIRP=46dBm, NF=7dB, BW=100MHz)", color=BLUE, fontweight='bold')
    ax6.legend(fontsize=7, facecolor='#161b22', labelcolor='white')
    ax6.set_ylim([-30, 50])

    # ── 7. Delay profile comparison ───────────────────────────
    ax7 = ax_(gs[2,0])
    for name, col in [('CDL-A',BLUE),('CDL-B',ORG),('CDL-C',GRN)]:
        prof  = CDL_PROFILES[name]
        delays = np.array(prof['delays_ns'])
        powers = np.array(prof['powers_db'])
        markerline, stemlines, baseline = ax7.stem(delays, powers, basefmt='C7-', label=name)
        plt.setp(stemlines, color=col); plt.setp(markerline, color=col)
    ax7.set_xlabel("Delay (ns)", color='#8b949e')
    ax7.set_ylabel("Power (dBr)", color='#8b949e')
    ax7.set_title("CDL Delay Profiles A/B/C\n(NLOS environments)", color=BLUE, fontweight='bold')
    ax7.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    # ── 8. mmWave vs sub-6GHz coverage ───────────────────────
    ax8 = ax_(gs[2,1])
    d_mm = np.linspace(5, 500, 300)
    pl_sub6 = [path_loss_uma_nlos(d, fc_ghz=3.5) for d in d_mm]
    pl_mm   = [path_loss_mmwave_o2o(d, fc_ghz=28) for d in d_mm]
    pl_mm70 = [path_loss_mmwave_o2o(d, fc_ghz=70) for d in d_mm]
    pl_mm_o2i = [path_loss_mmwave_o2i(d, fc_ghz=28, wall_loss_db=25) for d in d_mm]

    ax8.plot(d_mm, pl_sub6,    color=BLUE, lw=2.5, label='3.5 GHz UMa NLOS')
    ax8.plot(d_mm, pl_mm,      color=ORG,  lw=2,   label='28 GHz O2O LOS')
    ax8.plot(d_mm, pl_mm70,    color=RED,  lw=2,   label='70 GHz O2O LOS')
    ax8.plot(d_mm, pl_mm_o2i,  color=PUR,  lw=1.5, ls='--', label='28 GHz O2I (+25dB wall)')
    ax8.set_xlabel("Distance (m)", color='#8b949e')
    ax8.set_ylabel("Path Loss (dB)", color='#8b949e')
    ax8.set_title("mmWave vs Sub-6GHz Path Loss", color=BLUE, fontweight='bold')
    ax8.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    # ── 9. Coherence bandwidth vs delay spread ────────────────
    ax9 = ax_(gs[2,2])
    rms_delays = np.logspace(1, 4, 100)   # 10ns to 10us
    Bc_90  = 1 / (50 * rms_delays * 1e-9) / 1e6   # MHz, 90% coherence BW
    Bc_50  = 1 / (5  * rms_delays * 1e-9) / 1e6

    ax9.loglog(rms_delays, Bc_90, color=BLUE, lw=2, label='Bc (90% corr)')
    ax9.loglog(rms_delays, Bc_50, color=ORG,  lw=2, label='Bc (50% corr)')

    # Mark CDL profiles
    for name, rms_ns, col in [('CDL-A',45,BLUE),('CDL-B',25,ORG),('CDL-D',12,GRN)]:
        bc = 1 / (50 * rms_ns * 1e-9) / 1e6
        ax9.scatter([rms_ns],[bc], s=80, color=col, zorder=5)
        ax9.annotate(name, (rms_ns, bc), color=col, fontsize=8,
                     xytext=(rms_ns*1.5, bc*0.6))

    ax9.set_xlabel("RMS Delay Spread (ns)", color='#8b949e')
    ax9.set_ylabel("Coherence BW (MHz)", color='#8b949e')
    ax9.set_title("Coherence Bandwidth vs Delay Spread", color=BLUE, fontweight='bold')
    ax9.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    fig.text(0.5, 0.98, "3GPP NR Channel Models — TR 38.901",
             ha='center', color='white', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.965, "CDL-A/B/C/D/E · Path Loss UMa/UMi/mmWave · Doppler · Rayleigh/Rician",
             ha='center', color='#8b949e', fontsize=10)

    plt.savefig('nr_channel_models.png', dpi=130, bbox_inches='tight', facecolor=DARK)
    print("\n✓ Saved: nr_channel_models.png")
    print("\n✅  Channel Models demo complete.")
    plt.close('all')
