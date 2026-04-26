"""
src/massive_mimo/massive_mimo.py
==================================
Massive MIMO & Beamforming for 5G NR
Covers: ULA/UPA arrays, beamsteering, hybrid precoding,
        MRT/ZF/MMSE precoders, spatial multiplexing, capacity,
        channel estimation, FDD/TDD reciprocity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd

# ─────────────────────────────────────────────
# 1. ARRAY GEOMETRY
# ─────────────────────────────────────────────

def ula_steering_vector(N, theta_deg, d_lambda=0.5):
    """
    ULA (Uniform Linear Array) steering vector.
    N: antennas, theta: azimuth angle (deg), d_lambda: element spacing in wavelengths.
    a(θ) = [1, e^{j2πd·sin(θ)}, ..., e^{j2πd(N-1)sin(θ)}]
    """
    theta = np.deg2rad(theta_deg)
    n = np.arange(N)
    return np.exp(1j * 2 * np.pi * d_lambda * n * np.sin(theta))


def upa_steering_vector(Nh, Nv, az_deg, el_deg, d_lambda=0.5):
    """
    UPA (Uniform Planar Array) steering vector.
    Nh × Nv elements, azimuth and elevation angles.
    """
    az = np.deg2rad(az_deg); el = np.deg2rad(el_deg)
    nh = np.arange(Nh); nv = np.arange(Nv)
    a_h = np.exp(1j * 2*np.pi * d_lambda * nh * np.sin(az) * np.cos(el))
    a_v = np.exp(1j * 2*np.pi * d_lambda * nv * np.sin(el))
    return np.kron(a_v, a_h)   # Nh*Nv × 1


def array_pattern(N, theta_range, weights=None, d_lambda=0.5):
    """
    Compute array pattern |w^H a(θ)|² for range of angles.
    weights: beamforming vector (default: uniform → broadside)
    """
    if weights is None:
        weights = np.ones(N) / np.sqrt(N)
    patterns = []
    for theta in theta_range:
        a = ula_steering_vector(N, theta, d_lambda)
        patterns.append(np.abs(np.dot(weights.conj(), a))**2)
    return np.array(patterns)


# ─────────────────────────────────────────────
# 2. CHANNEL MODEL
# ─────────────────────────────────────────────

def geometric_channel(N_bs, N_ue, n_paths=10, az_range=(-60, 60)):
    """
    Geometric (ray-based) MIMO channel matrix.
    H = sqrt(N_bs*N_ue/L) * sum_l g_l * a_bs(φ_l) * a_ue(θ_l)^H
    """
    L = n_paths
    H = np.zeros((N_ue, N_bs), dtype=complex)
    rng = np.random.default_rng(42)
    az_min, az_max = az_range

    for _ in range(L):
        g      = (rng.standard_normal() + 1j*rng.standard_normal()) / np.sqrt(2)
        phi_bs = rng.uniform(az_min, az_max)   # DoD at BS
        phi_ue = rng.uniform(az_min, az_max)   # DoA at UE
        a_bs   = ula_steering_vector(N_bs, phi_bs)
        a_ue   = ula_steering_vector(N_ue, phi_ue)
        H     += g * np.outer(a_ue, a_bs.conj())

    H *= np.sqrt(N_bs * N_ue / L)
    return H


def iid_rayleigh_channel(N_rx, N_tx):
    """i.i.d. Rayleigh MIMO channel — no spatial correlation."""
    return (np.random.randn(N_rx, N_tx) + 1j*np.random.randn(N_rx, N_tx)) / np.sqrt(2)


# ─────────────────────────────────────────────
# 3. PRECODERS / BEAMFORMERS
# ─────────────────────────────────────────────

def mrt_precoder(H):
    """
    MRT (Maximum Ratio Transmission) / conjugate beamforming.
    W = H^H / ||H||_F
    """
    W = H.conj().T
    return W / np.linalg.norm(W, 'fro')


def zf_precoder(H):
    """
    Zero-Forcing precoder.
    W = H^H (H H^H)^{-1}
    """
    W = H.conj().T @ np.linalg.pinv(H @ H.conj().T)
    return W / np.linalg.norm(W, 'fro')


def mmse_precoder(H, snr_lin):
    """
    MMSE precoder (regularised ZF).
    W = H^H (H H^H + N_ue/SNR * I)^{-1}
    """
    N_ue = H.shape[0]
    W = H.conj().T @ np.linalg.inv(H @ H.conj().T + (N_ue/snr_lin) * np.eye(N_ue))
    return W / np.linalg.norm(W, 'fro')


def svd_precoder(H, n_streams=None):
    """
    SVD-based precoder (optimal for MIMO capacity).
    W = V[:, :n_streams]  (right singular vectors)
    """
    U, S, Vh = svd(H, full_matrices=False)
    n_streams = n_streams or min(H.shape)
    V = Vh.conj().T
    return V[:, :n_streams], S[:n_streams]


def dft_codebook(N, n_beams=None):
    """DFT-based beamforming codebook (discrete angles)."""
    n_beams = n_beams or N
    beams   = []
    for k in range(n_beams):
        theta = np.arcsin(2*k/n_beams - 1) * 180 / np.pi
        w     = ula_steering_vector(N, theta) / np.sqrt(N)
        beams.append(w)
    return np.array(beams)   # n_beams × N


def hybrid_precoder(H, N_rf, N_tx):
    """
    Hybrid Beamforming: N_rf RF chains, N_tx antennas.
    Splits full-digital precoder into F_RF (analog) × F_BB (digital).
    Uses OMP-inspired greedy approach.
    """
    # Target: full SVD precoder
    W_opt, S = svd_precoder(H, n_streams=N_rf)  # N_tx × N_rf

    # Analog codebook (DFT)
    codebook = dft_codebook(N_tx, N_tx)    # N_tx × N_tx

    F_RF = np.zeros((N_tx, N_rf), dtype=complex)
    residual = W_opt.copy()

    for i in range(N_rf):
        # Find best beam from codebook
        col_i = min(i, residual.shape[1]-1)
        correlations = np.abs(codebook @ residual[:, col_i])
        best_idx     = np.argmax(correlations)
        F_RF[:, i]   = codebook[best_idx]
        # Update residual
        F_RF_sub = F_RF[:, :i+1]
        F_BB_sub = np.linalg.pinv(F_RF_sub) @ W_opt[:, :i+1]
        residual[:, :i+1] = W_opt[:, :i+1] - F_RF_sub @ F_BB_sub

    # Digital precoder
    F_BB = np.linalg.pinv(F_RF) @ W_opt
    # Normalise
    scale = np.linalg.norm(F_RF @ F_BB, 'fro')
    F_BB /= max(scale, 1e-10)

    return F_RF, F_BB


# ─────────────────────────────────────────────
# 4. CAPACITY ANALYSIS
# ─────────────────────────────────────────────

def mimo_capacity(H, snr_db):
    """Shannon capacity of MIMO channel (bits/s/Hz)."""
    N_tx = H.shape[1]
    snr  = 10**(snr_db/10)
    G    = H @ H.conj().T   # N_rx × N_rx Gram matrix
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.maximum(eigvals, 0)
    C = np.sum(np.log2(1 + snr/N_tx * eigvals))
    return C


def massive_mimo_capacity_vs_antennas(K=8, snr_db=10, N_max=256):
    """
    Capacity vs number of BS antennas (K single-antenna UEs).
    Shows massive MIMO gain.
    """
    N_range = np.arange(K, N_max+1, 4)
    caps    = []
    for N in N_range:
        H = geometric_channel(N, K, n_paths=10)
        C = 0
        for k in range(K):
            h_k   = H[k:k+1, :]   # 1 × N_bs
            snr_u = 10**(snr_db/10) * N / K   # array gain
            g     = np.abs(h_k @ h_k.conj().T)[0,0]
            C    += np.log2(1 + snr_u * g / N)
        caps.append(C)
    return N_range, np.array(caps)


def water_filling(eigenvalues, P_total):
    """
    Water-filling power allocation across MIMO sub-channels.
    Returns power per sub-channel.
    """
    lam    = np.sort(eigenvalues)[::-1]   # largest first
    n      = len(lam)
    P      = np.zeros(n)
    for k in range(n, 0, -1):
        mu = (P_total + np.sum(1/lam[:k])) / k
        P_k = mu - 1/lam[:k]
        if np.all(P_k >= 0):
            P[:k] = P_k
            break
    return P


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 65)
    print("  Massive MIMO & Beamforming — 5G NR (3GPP TR 38.901)")
    print("=" * 65)

    DARK = '#0d1117'; BG = '#161b22'
    GRD  = dict(alpha=0.15, color='white')

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    def dark_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(BG)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.grid(**GRD)
        return ax

    theta_range = np.linspace(-90, 90, 361)

    # ── 1. Array pattern — N = 4, 16, 64 ─────────────────────
    ax1 = dark_ax(gs[0, 0])
    for N, col in [(4,'#58a6ff'),(16,'#3fb950'),(64,'#f85149')]:
        # Steer to 30 degrees
        w  = ula_steering_vector(N, 30) / np.sqrt(N)
        P  = array_pattern(N, theta_range, w)
        P_db = 10*np.log10(P + 1e-10)
        ax1.plot(theta_range, P_db - P_db.max(), color=col, lw=1.5, label=f'N={N}')
    ax1.axvline(30, color='white', ls='--', lw=0.8, alpha=0.5, label='Beam target 30°')
    ax1.set_ylim([-40, 2]); ax1.set_xlabel('Angle (deg)', color='#8b949e')
    ax1.set_ylabel('Normalised Gain (dB)', color='#8b949e')
    ax1.set_title('ULA Beam Pattern — N=4,16,64\nSteered to 30°', color='#58a6ff', fontweight='bold')
    ax1.legend(fontsize=8, facecolor=BG, labelcolor='white')

    # ── 2. Simultaneous multi-beam (spatial multiplexing) ─────
    ax2 = dark_ax(gs[0, 1])
    N_bs   = 64
    angles = [-40, -10, 20, 50]   # 4 users
    colors_u = ['#58a6ff','#3fb950','#e3b341','#f85149']
    for i, (ang, col) in enumerate(zip(angles, colors_u)):
        w   = ula_steering_vector(N_bs, ang) / np.sqrt(N_bs)
        P   = array_pattern(N_bs, theta_range, w)
        P_db = 10*np.log10(P + 1e-10)
        ax2.plot(theta_range, P_db - P_db.max(), color=col, lw=1.2, alpha=0.8,
                 label=f'UE{i+1} @ {ang}°')
    ax2.set_ylim([-40, 2]); ax2.set_xlabel('Angle (deg)', color='#8b949e')
    ax2.set_ylabel('Normalised Gain (dB)', color='#8b949e')
    ax2.set_title(f'Multi-User Beamforming\n{len(angles)} simultaneous beams (N_BS={N_bs})', color='#58a6ff', fontweight='bold')
    ax2.legend(fontsize=8, facecolor=BG, labelcolor='white')

    # ── 3. Precoder comparison (MRT/ZF/MMSE) BER ─────────────
    ax3 = dark_ax(gs[0, 2])
    N_bs_p, K_p = 64, 4
    snr_range = np.arange(-5, 25, 3)
    precoder_results = {}

    for prec_name, prec_fn in [('MRT', mrt_precoder), ('ZF', zf_precoder)]:
        bers = []
        for snr_db in snr_range:
            snr_lin = 10**(snr_db/10)
            H   = geometric_channel(N_bs_p, K_p, n_paths=15)
            W   = prec_fn(H)
            y   = H @ W   # K × K effective channel
            # SINR per user (diagonal vs interference)
            total_ber = 0
            for k in range(K_p):
                sig   = np.abs(y[k, k])**2 * snr_lin
                intf  = sum(np.abs(y[k, j])**2 * snr_lin for j in range(K_p) if j != k)
                sinr  = sig / (intf + 1)
                from scipy.special import erfc
                ber_k = 0.5 * erfc(np.sqrt(sinr))
                total_ber += ber_k / K_p
            bers.append(max(float(total_ber), 1e-5))
        precoder_results[prec_name] = bers

    colors_prec = {'MRT':'#58a6ff', 'ZF':'#3fb950', 'MMSE':'#e3b341'}
    for name, bers in precoder_results.items():
        ax3.semilogy(snr_range, bers, 'o-', color=colors_prec[name], lw=1.8, ms=4, label=name)
    ax3.set_xlabel('SNR (dB)', color='#8b949e'); ax3.set_ylabel('BER', color='#8b949e')
    ax3.set_title(f'Precoder Comparison — MRT vs ZF\n(N_BS={N_bs_p}, K={K_p} users)', color='#58a6ff', fontweight='bold')
    ax3.legend(fontsize=8, facecolor=BG, labelcolor='white')
    ax3.set_ylim([1e-4, 1])

    # ── 4. Capacity vs N_BS (massive MIMO gain) ───────────────
    ax4 = dark_ax(gs[1, 0])
    K   = 8
    N_range, caps = massive_mimo_capacity_vs_antennas(K=K, snr_db=10, N_max=200)
    # Theoretical: C ≈ K * log2(1 + SNR * N/K) for large N
    snr_lin = 10**(10/10)
    C_theory = K * np.log2(1 + snr_lin * N_range / K)
    ax4.plot(N_range, caps, color='#58a6ff', lw=2, label='Simulated')
    ax4.plot(N_range, C_theory, color='#f85149', lw=1.5, ls='--', label='Theory (large N)')
    ax4.axvline(K, color='white', lw=0.6, ls=':', alpha=0.5)
    ax4.text(K+2, 2, 'N=K', color='#8b949e', fontsize=8)
    ax4.set_xlabel('N_BS (BS antennas)', color='#8b949e')
    ax4.set_ylabel('Sum Capacity (bits/s/Hz)', color='#8b949e')
    ax4.set_title(f'Massive MIMO Capacity vs N_BS\n(K={K} UEs, SNR=10dB)', color='#58a6ff', fontweight='bold')
    ax4.legend(fontsize=8, facecolor=BG, labelcolor='white')

    # ── 5. Hybrid beamforming architecture ────────────────────
    ax5 = dark_ax(gs[1, 1])
    N_tx_hb = 64; N_rf_hb = 8; K_hb = 4
    H_hb   = geometric_channel(N_tx_hb, K_hb, n_paths=12)
    F_RF, F_BB = hybrid_precoder(H_hb, N_rf_hb, N_tx_hb)
    F_hybrid   = F_RF @ F_BB   # N_tx × K

    # Full digital for reference
    W_full, S_full = svd_precoder(H_hb, n_streams=K_hb)

    # Plot effective beampattern per stream
    for i in range(K_hb):
        P_h = array_pattern(N_tx_hb, theta_range, F_hybrid[:, i])
        P_f = array_pattern(N_tx_hb, theta_range, W_full[:, i])
        ax5.plot(theta_range, 10*np.log10(P_h+1e-10), lw=1, alpha=0.7, color=colors_u[i])
        ax5.plot(theta_range, 10*np.log10(P_f+1e-10), lw=0.7, ls='--', alpha=0.4, color=colors_u[i])

    ax5.set_ylim([-30, 25]); ax5.set_xlabel('Angle (deg)', color='#8b949e')
    ax5.set_ylabel('Array Gain (dB)', color='#8b949e')
    ax5.set_title(f'Hybrid Beamforming (N={N_tx_hb}, N_RF={N_rf_hb})\nSolid=Hybrid, Dashed=Full Digital', color='#58a6ff', fontweight='bold')

    # ── 6. Singular value / capacity distribution ─────────────
    ax6 = dark_ax(gs[1, 2])
    configs_c = [(4,4,'i.i.d'),(8,8,'i.i.d'),(64,4,'Geometric')]
    cap_colors = ['#58a6ff','#3fb950','#f85149']
    for (N_r, N_t, ch_type), col in zip(configs_c, cap_colors):
        caps_mc = []
        for _ in range(300):
            if ch_type == 'i.i.d':
                H_mc = iid_rayleigh_channel(N_r, N_t)
            else:
                H_mc = geometric_channel(N_t, N_r, n_paths=10)
            caps_mc.append(mimo_capacity(H_mc, snr_db=10))
        ax6.hist(caps_mc, bins=30, alpha=0.65, color=col,
                 label=f'{N_t}×{N_r} {ch_type}', density=True, edgecolor='none')
    ax6.set_xlabel('Capacity (bits/s/Hz)', color='#8b949e')
    ax6.set_ylabel('PDF', color='#8b949e')
    ax6.set_title('MIMO Capacity Distribution (MC)\nSNR=10dB, 300 realisations', color='#58a6ff', fontweight='bold')
    ax6.legend(fontsize=8, facecolor=BG, labelcolor='white')

    fig.text(0.5, 0.98, 'Massive MIMO & Beamforming — 5G NR Array Processing',
             ha='center', color='white', fontsize=14, fontweight='bold')
    plt.savefig('massive_mimo.png', dpi=130, bbox_inches='tight', facecolor=DARK)
    print("\n✓ Saved: massive_mimo.png")
    plt.close('all')
    print("\n✅  Massive MIMO & Beamforming demo complete.")
