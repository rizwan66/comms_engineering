"""
src/otfs/otfs_system.py
========================
OTFS — Orthogonal Time Frequency Space Modulation

Key concept: OTFS places symbols in the Delay-Doppler (DD) domain.
The DD domain provides a compact, stable representation of doubly-dispersive
(time-varying multipath) channels, enabling robust high-mobility communications.

Transforms:
  DD → TF  : ISFFT (Inverse Symplectic Finite Fourier Transform)
  TF → DD  : SFFT  (Symplectic Finite Fourier Transform)
  TF → time: Heisenberg (modulate with pulse g_tx)
  time → TF: Wigner (demodulate with pulse g_rx)

Pipeline:
  X_DD → ISFFT → X_TF → Heisenberg → s(t) → channel → r(t)
  → Wigner → Y_TF → SFFT → Y_DD → equalise → X̂_DD
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class OTFSConfig:
    M: int   = 32     # delay bins (subcarriers)
    N: int   = 32     # Doppler bins (time slots)
    df: float = 15e3  # subcarrier spacing (Hz)

    @property
    def T(self):       return 1.0 / self.df           # OFDM symbol duration (s)
    @property
    def delay_res(self): return 1.0 / (self.M * self.df)  # delay resolution (s)
    @property
    def dop_res(self):   return self.df / self.N          # Doppler resolution (Hz)
    @property
    def BW(self):        return self.M * self.df          # bandwidth (Hz)
    @property
    def Tf(self):        return self.N * self.T            # frame duration (s)

    def __str__(self):
        return (f"OTFSConfig(M={self.M}, N={self.N}, "
                f"df={self.df/1e3:.1f}kHz, BW={self.BW/1e6:.2f}MHz, "
                f"Tf={self.Tf*1e3:.2f}ms)")


# ─────────────────────────────────────────────
# 2. SYMPLECTIC TRANSFORMS
# ─────────────────────────────────────────────

def isfft(X_DD):
    """
    Inverse SFFT: Delay-Doppler → Time-Frequency
    X_DD[l,k]: l=delay bin, k=Doppler bin
    X_TF[m,n]: m=subcarrier, n=time slot
    """
    return np.fft.ifft(np.fft.fft(X_DD, axis=1), axis=0) * np.sqrt(X_DD.shape[0])

def sfft(X_TF):
    """
    SFFT: Time-Frequency → Delay-Doppler
    """
    return np.fft.ifft(np.fft.fft(X_TF, axis=0), axis=1) / np.sqrt(X_TF.shape[0])


# ─────────────────────────────────────────────
# 3. HEISENBERG / WIGNER (pulse shaping)
# ─────────────────────────────────────────────

def heisenberg(X_TF, M, N):
    """
    Heisenberg transform: TF frame → time-domain signal.
    Uses rectangular pulse (CP-OFDM style).
    """
    s = []
    for n in range(N):
        # IFFT of n-th time slot (subcarriers → time samples)
        sym = np.fft.ifft(X_TF[:, n], n=M) * np.sqrt(M)
        s.append(sym)
    return np.concatenate(s)   # length M*N


def wigner(r, M, N):
    """
    Wigner transform: time-domain signal → TF frame.
    """
    Y_TF = np.zeros((M, N), dtype=complex)
    for n in range(N):
        seg = r[n*M : n*M + M]
        if len(seg) < M:
            seg = np.pad(seg, (0, M - len(seg)))
        Y_TF[:, n] = np.fft.fft(seg, n=M) / np.sqrt(M)
    return Y_TF


# ─────────────────────────────────────────────
# 4. FULL TX / RX
# ─────────────────────────────────────────────

def otfs_tx(X_DD, cfg: OTFSConfig):
    X_TF = isfft(X_DD)
    s    = heisenberg(X_TF, cfg.M, cfg.N)
    return s, X_TF


def otfs_rx(r, cfg: OTFSConfig):
    Y_TF = wigner(r, cfg.M, cfg.N)
    Y_DD = sfft(Y_TF)
    return Y_DD, Y_TF


# ─────────────────────────────────────────────
# 5. CHANNEL MODEL (frequency-selective fading)
# ─────────────────────────────────────────────

def apply_channel_tf(X_TF, paths, cfg: OTFSConfig, snr_db=20):
    """
    Apply doubly-dispersive channel in TF domain.
    Each path contributes a time-varying gain per TF bin.
    H[m,n] = sum_p g_p * exp(-j2pi*tau_p*m*df) * exp(j2pi*nu_p*n*T)
    """
    M, N = cfg.M, cfg.N
    H    = np.zeros((M, N), dtype=complex)
    for p in paths:
        tau = p['delay_bins'] * cfg.delay_res
        nu  = p['dop_bins']   * cfg.dop_res
        m_arr = np.arange(M); n_arr = np.arange(N)
        H += p['gain'] * (np.exp(-1j*2*np.pi*tau*m_arr*cfg.df)[:,None]
                        * np.exp( 1j*2*np.pi*nu *n_arr*cfg.T)[None,:])

    Y_TF = H * X_TF   # element-wise (valid for OTFS with rectangular pulse)

    # Add AWGN
    P   = np.mean(np.abs(Y_TF)**2)
    N0  = P / (10**(snr_db/10))
    Y_TF = Y_TF + np.sqrt(N0/2) * (np.random.randn(M,N) + 1j*np.random.randn(M,N))
    return Y_TF, H


def mmse_eq_tf(Y_TF, H, snr_lin):
    """MMSE equalisation in TF domain."""
    return Y_TF * np.conj(H) / (np.abs(H)**2 + 1/max(snr_lin, 0.01))


# ─────────────────────────────────────────────
# 6. OFDM REFERENCE (same channel)
# ─────────────────────────────────────────────

def ofdm_sim(X_TF_data, paths, cfg: OTFSConfig, snr_db=20):
    """
    OFDM over the same TF channel (no DD processing).
    OFDM treats each TF bin independently — severe ICI under Doppler.
    """
    Y_TF, H = apply_channel_tf(X_TF_data, paths, cfg, snr_db)
    # OFDM equalises each subcarrier per time slot (ignores Doppler cross-terms)
    Y_eq = mmse_eq_tf(Y_TF, H, 10**(snr_db/10))
    return Y_eq


# ─────────────────────────────────────────────
# 7. QPSK
# ─────────────────────────────────────────────

def qpsk_map(n):
    bits   = np.random.randint(0, 4, n)
    return np.exp(1j * (np.pi/4 + np.pi/2 * bits))

def qpsk_demod(syms):
    return (np.floor(np.angle(syms) % (2*np.pi) / (np.pi/2)).astype(int)) % 4

def ser(tx, rx):
    return np.mean(qpsk_demod(tx.flatten()) != qpsk_demod(rx.flatten()))


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)

    DARK  = '#0d1117'
    BLUE  = '#58a6ff'
    ORG   = '#e3b341'
    GRN   = '#3fb950'
    RED   = '#f85149'

    print("=" * 60)
    print("  OTFS — Orthogonal Time Frequency Space Modulation")
    print("  Delay-Doppler domain signal processing")
    print("=" * 60)

    cfg = OTFSConfig(M=32, N=32)
    print(f"\n  {cfg}")
    print(f"  Delay resolution  : {cfg.delay_res*1e6:.2f} us")
    print(f"  Doppler resolution: {cfg.dop_res:.1f} Hz")

    # High-mobility channel (4 paths with significant Doppler)
    paths = [
        {'delay_bins': 0, 'dop_bins':  0,  'gain': 1.00 + 0j},
        {'delay_bins': 2, 'dop_bins':  3,  'gain': 0.60 * np.exp(1j*0.7)},
        {'delay_bins': 5, 'dop_bins': -4,  'gain': 0.40 * np.exp(1j*1.3)},
        {'delay_bins': 8, 'dop_bins':  6,  'gain': 0.25 * np.exp(1j*2.1)},
    ]

    SNR = 20
    snr_lin = 10**(SNR/10)

    # TX
    syms = qpsk_map(cfg.M * cfg.N)
    X_DD = syms.reshape(cfg.M, cfg.N)

    # OTFS: DD → TF → channel → TF eq → DD
    s_tx, X_TF = otfs_tx(X_DD, cfg)
    Y_TF_ch, H_TF = apply_channel_tf(X_TF, paths, cfg, SNR)
    Y_TF_eq = mmse_eq_tf(Y_TF_ch, H_TF, snr_lin)
    Y_DD, _ = otfs_rx(heisenberg(Y_TF_eq, cfg.M, cfg.N), cfg)
    # After TF equalisation, convert back through SFFT
    Y_DD_eq = sfft(Y_TF_eq)

    ber_otfs = ser(X_DD, Y_DD_eq)

    # OFDM: same channel, no DD processing
    Y_ofdm = ofdm_sim(X_TF, paths, cfg, SNR)
    ber_ofdm = ser(X_DD, Y_ofdm)

    print(f"\n  SNR = {SNR} dB  |  4-path Doppler channel")
    print(f"  OTFS SER : {ber_otfs:.4f}")
    print(f"  OFDM SER : {ber_ofdm:.4f}")

    # BER vs SNR
    snr_range = range(-5, 28, 3)
    sers_otfs, sers_ofdm = [], []
    for snr in snr_range:
        sl = 10**(snr/10)
        Y_tf_c, H_tf = apply_channel_tf(X_TF, paths, cfg, snr)
        Y_tf_e = mmse_eq_tf(Y_tf_c, H_tf, sl)
        Y_dd   = sfft(Y_tf_e)
        sers_otfs.append(max(ser(X_DD, Y_dd), 1e-4))

        Y_o = ofdm_sim(X_TF, paths, cfg, snr)
        sers_ofdm.append(max(ser(X_DD, Y_o), 1e-4))

    # ── PLOTS ─────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    def dark_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.grid(True, alpha=0.15, color='white')
        return ax

    # 1. DD channel (delay-Doppler spread)
    ax1 = dark_ax(gs[0,0])
    H_DD_vis = np.zeros((cfg.M, cfg.N))
    for p in paths:
        H_DD_vis[p['delay_bins']%cfg.M, p['dop_bins']%cfg.N] = abs(p['gain'])
    im = ax1.imshow(H_DD_vis, aspect='auto', cmap='hot', origin='lower')
    ax1.set_title("Delay-Doppler Channel Spread", color=BLUE, fontweight='bold')
    ax1.set_xlabel("Doppler bin", color='#8b949e')
    ax1.set_ylabel("Delay bin", color='#8b949e')
    plt.colorbar(im, ax=ax1).ax.tick_params(colors='#8b949e')

    # 2. TF channel magnitude
    ax2 = dark_ax(gs[0,1])
    im2 = ax2.imshow(np.abs(H_TF), aspect='auto', cmap='plasma', origin='lower')
    ax2.set_title("TF Channel |H[m,n]|", color=BLUE, fontweight='bold')
    ax2.set_xlabel("Time slot n", color='#8b949e')
    ax2.set_ylabel("Subcarrier m", color='#8b949e')
    plt.colorbar(im2, ax=ax2).ax.tick_params(colors='#8b949e')

    # 3. OTFS DD received (before eq)
    ax3 = dark_ax(gs[0,2])
    Y_DD_raw = sfft(Y_TF_ch)
    im3 = ax3.imshow(np.abs(Y_DD_raw), aspect='auto', cmap='viridis', origin='lower')
    ax3.set_title("Received DD Frame |Y_DD| (before eq)", color=BLUE, fontweight='bold')
    ax3.set_xlabel("Doppler bin", color='#8b949e')
    ax3.set_ylabel("Delay bin", color='#8b949e')
    plt.colorbar(im3, ax=ax3).ax.tick_params(colors='#8b949e')

    # 4. OTFS constellation
    ax4 = dark_ax(gs[1,0])
    flat = Y_DD_eq.flatten()
    ax4.scatter(flat.real, flat.imag, s=3, alpha=0.4, color=BLUE)
    ax4.set_title(f"OTFS Constellation (SER={ber_otfs:.4f})", color=BLUE, fontweight='bold')
    ax4.set_xlabel("I", color='#8b949e'); ax4.set_ylabel("Q", color='#8b949e')
    ax4.axhline(0,color='white',lw=0.3); ax4.axvline(0,color='white',lw=0.3)
    ax4.set_aspect('equal')

    # 5. OFDM constellation
    ax5 = dark_ax(gs[1,1])
    flat_o = Y_ofdm.flatten()
    ax5.scatter(flat_o.real, flat_o.imag, s=3, alpha=0.4, color=RED)
    ax5.set_title(f"OFDM Constellation (SER={ber_ofdm:.4f})", color=RED, fontweight='bold')
    ax5.set_xlabel("I", color='#8b949e'); ax5.set_ylabel("Q", color='#8b949e')
    ax5.axhline(0,color='white',lw=0.3); ax5.axvline(0,color='white',lw=0.3)
    ax5.set_aspect('equal')

    # 6. SER vs SNR
    ax6 = dark_ax(gs[1,2])
    snr_arr = np.array(list(snr_range))
    ax6.semilogy(snr_arr, sers_otfs, 'o-', color=BLUE, lw=2, ms=5, label='OTFS (DD domain)')
    ax6.semilogy(snr_arr, sers_ofdm, 's--', color=RED, lw=2, ms=5, label='OFDM (TF domain)')
    ax6.set_title("SER vs SNR — Doppler Channel", color=BLUE, fontweight='bold')
    ax6.set_xlabel("SNR (dB)", color='#8b949e')
    ax6.set_ylabel("SER", color='#8b949e')
    ax6.legend(fontsize=9, facecolor='#161b22', labelcolor='white')
    ax6.set_ylim([1e-4, 1])

    fig.text(0.5, 0.98, "OTFS Modulation — Delay-Doppler Domain Signal Processing",
             ha='center', color='white', fontsize=14, fontweight='bold')

    plt.savefig('otfs_system.png', dpi=130, bbox_inches='tight', facecolor=DARK)
    print("\n✓ Saved: otfs_system.png")

    print(f"\n  Summary (SNR={SNR}dB, 4-path Doppler channel):")
    print(f"  OTFS SER : {ber_otfs:.4f}")
    print(f"  OFDM SER : {ber_ofdm:.4f}")
    print("\n✅  OTFS module demo complete — 1 figure saved.")
    plt.close('all')
