"""
src/mimo/mimo_system.py
========================
MIMO (Multiple-Input Multiple-Output) communications:
  - Rayleigh flat-fading MIMO channel matrix H (Nr × Nt)
  - Spatial multiplexing (independent streams per TX antenna)
  - Zero-Forcing (ZF) equalizer: W = (HᴴH)⁻¹Hᴴ
  - MMSE equalizer: W = (HᴴH + σ²I)⁻¹Hᴴ
  - Alamouti 2×1 space-time block code (diversity)
  - MIMO capacity vs SISO Shannon capacity
  - BER comparison: SISO · ZF · MMSE · Alamouti
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc


# ─────────────────────────────────────────────
# 1. CHANNEL MODEL
# ─────────────────────────────────────────────

def rayleigh_channel(Nr, Nt, seed=None):
    """
    Generate a Rayleigh flat-fading MIMO channel matrix H (Nr × Nt).
    Each element ~ CN(0,1) (i.i.d. complex Gaussian).
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((Nr, Nt))
            + 1j * rng.standard_normal((Nr, Nt))) / np.sqrt(2)


def awgn_noise(Nr, n_syms, sigma2):
    """Generate complex AWGN noise of shape (Nr, n_syms)."""
    rng = np.random.default_rng()
    return np.sqrt(sigma2/2) * (rng.standard_normal((Nr, n_syms))
                                 + 1j*rng.standard_normal((Nr, n_syms)))


# ─────────────────────────────────────────────
# 2. SPATIAL MULTIPLEXING TX
# ─────────────────────────────────────────────

def bpsk_map(bits):
    return 1 - 2*bits.astype(float)    # 0→+1, 1→-1


def bpsk_demap(soft):
    return (soft < 0).astype(int)


def spatial_multiplex_tx(bits, Nt):
    """
    Split bit stream across Nt antennas (BPSK per antenna).
    Returns symbol matrix X of shape (Nt, n_syms).
    """
    n_per_ant = len(bits) // Nt
    bits = bits[:n_per_ant * Nt]
    X = bpsk_map(bits).reshape(Nt, n_per_ant)
    return X


# ─────────────────────────────────────────────
# 3. EQUALISERS
# ─────────────────────────────────────────────

def zf_equaliser(H):
    """
    Zero-Forcing equaliser: W_ZF = (HᴴH)⁻¹Hᴴ
    Inverts channel completely, but amplifies noise at nulls.
    """
    return np.linalg.pinv(H)     # Moore-Penrose pseudoinverse


def mmse_equaliser(H, sigma2, Nt):
    """
    MMSE equaliser: W_MMSE = (HᴴH + σ²·I)⁻¹Hᴴ
    Trades off ISI vs noise amplification optimally.
    sigma2: noise variance per receive antenna.
    """
    A = H.conj().T @ H + sigma2 * np.eye(Nt)
    return np.linalg.solve(A, H.conj().T)


def apply_equaliser(W, Y):
    """Apply equaliser W to received matrix Y → estimated symbols X̂."""
    return W @ Y


# ─────────────────────────────────────────────
# 4. ALAMOUTI 2×1 STBC (DIVERSITY)
# ─────────────────────────────────────────────

def alamouti_encode(s1, s2):
    """
    Alamouti 2×1 encoder for two consecutive symbols s1, s2.
    TX matrix (2 antennas × 2 time slots):
        [ s1  -s2* ]
        [ s2   s1* ]
    Returns (slot1_ant0, slot1_ant1, slot2_ant0, slot2_ant1)
    """
    return np.array([[s1, -s2.conj()],
                     [s2,  s1.conj()]])


def alamouti_decode(r1, r2, h1, h2):
    """
    Alamouti MRC combining.
    r1, r2: received samples at time slot 1 and 2.
    h1, h2: channel gains from TX antenna 0 and 1.
    Returns estimated (ŝ1, ŝ2).
    """
    s1_hat = np.conj(h1)*r1 + h2*np.conj(r2)
    s2_hat = np.conj(h2)*r1 - h1*np.conj(r2)
    return s1_hat, s2_hat


# ─────────────────────────────────────────────
# 5. MIMO CAPACITY
# ─────────────────────────────────────────────

def mimo_capacity(H, snr_linear):
    """
    MIMO ergodic capacity (bits/s/Hz):
    C = log2 det(I + (ρ/Nt) H Hᴴ)
    """
    Nr, Nt = H.shape
    A  = np.eye(Nr) + (snr_linear / Nt) * H @ H.conj().T
    sg = np.linalg.svd(A, compute_uv=False)
    return np.sum(np.log2(np.maximum(sg, 1e-10)))


def siso_capacity(snr_linear):
    return np.log2(1 + snr_linear)


# ─────────────────────────────────────────────
# 6. BER SIMULATION
# ─────────────────────────────────────────────

def simulate_mimo_ber(Nt, Nr, snr_db_range, equaliser='zf',
                      n_bits=4000, n_channels=20):
    """Monte-Carlo BER for Nt×Nr MIMO with chosen equaliser."""
    rng  = np.random.default_rng(42)
    bers = []

    for snr_db in snr_db_range:
        snr    = 10**(snr_db/10)
        sigma2 = 1.0 / snr           # noise variance (signal power = 1)
        errs   = 0; total = 0

        for _ in range(n_channels):
            H = rayleigh_channel(Nr, Nt, seed=None)
            bits = rng.integers(0, 2, n_bits)
            X    = spatial_multiplex_tx(bits, Nt)             # (Nt, n_syms)
            n_syms = X.shape[1]

            # Received: Y = HX + N
            N = awgn_noise(Nr, n_syms, sigma2)
            Y = H @ X + N                                     # (Nr, n_syms)

            if equaliser == 'zf':
                W = zf_equaliser(H)
            elif equaliser == 'mmse':
                W = mmse_equaliser(H, sigma2, Nt)

            X_hat = (W @ Y).real                              # (Nt, n_syms)
            bits_rx = bpsk_demap(X_hat.flatten())
            n = min(len(bits), len(bits_rx))
            errs  += np.sum(bits[:n] != bits_rx[:n])
            total += n

        bers.append(max(errs / total, 1e-6))

    return np.array(bers)


def simulate_alamouti_ber(snr_db_range, n_syms=2000):
    """BER for 2×1 Alamouti STBC over Rayleigh fading."""
    rng  = np.random.default_rng(7)
    bers = []

    for snr_db in snr_db_range:
        snr    = 10**(snr_db/10)
        sigma  = np.sqrt(0.5/snr)
        errs = 0

        for _ in range(n_syms // 2):
            b1, b2 = rng.integers(0, 2, 2)
            s1 = 1 - 2*b1 + 0j
            s2 = 1 - 2*b2 + 0j

            # Random 2-tap channel
            h1 = (rng.standard_normal() + 1j*rng.standard_normal()) / np.sqrt(2)
            h2 = (rng.standard_normal() + 1j*rng.standard_normal()) / np.sqrt(2)

            # TX
            tx = alamouti_encode(s1, s2)     # (2, 2)

            # RX (1 receive antenna)
            r1 = h1*tx[0,0] + h2*tx[1,0] + sigma*(rng.standard_normal()+1j*rng.standard_normal())
            r2 = h1*tx[0,1] + h2*tx[1,1] + sigma*(rng.standard_normal()+1j*rng.standard_normal())

            s1h, s2h = alamouti_decode(r1, r2, h1, h2)
            if (s1h.real < 0) != bool(b1): errs += 1
            if (s2h.real < 0) != bool(b2): errs += 1

    total = n_syms
    return max(errs / total, 1e-6)


def siso_rayleigh_ber(snr_db):
    """Analytical BER for BPSK over Rayleigh fading (SISO)."""
    snr = 10**(np.array(snr_db)/10)
    return 0.5 * (1 - np.sqrt(snr / (1 + snr)))


def bpsk_awgn_ber(snr_db):
    return 0.5 * erfc(np.sqrt(10**(np.array(snr_db)/10)))


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(0)

    print("=" * 58)
    print("  MIMO Demo — ZF · MMSE · Alamouti · Capacity")
    print("=" * 58)

    snr_db_range = list(range(0, 22, 2))
    snr_lin_arr  = 10**(np.array(snr_db_range)/10)

    # ── BER simulations ──
    print("SISO Rayleigh (analytical)...")
    ber_siso_rayleigh = siso_rayleigh_ber(snr_db_range)

    print("2×2 MIMO ZF...")
    ber_2x2_zf   = simulate_mimo_ber(2, 2, snr_db_range, 'zf',   n_bits=2000, n_channels=15)
    print("2×2 MIMO MMSE...")
    ber_2x2_mmse = simulate_mimo_ber(2, 2, snr_db_range, 'mmse', n_bits=2000, n_channels=15)
    print("4×4 MIMO ZF...")
    ber_4x4_zf   = simulate_mimo_ber(4, 4, snr_db_range, 'zf',   n_bits=2000, n_channels=10)
    print("Alamouti 2×1...")
    ber_alamouti = [simulate_alamouti_ber([s], n_syms=1000) for s in snr_db_range]

    ber_awgn = bpsk_awgn_ber(snr_db_range)

    # ── Capacity ──
    n_chan = 200
    cap_siso  = np.zeros(len(snr_db_range))
    cap_2x2   = np.zeros(len(snr_db_range))
    cap_4x4   = np.zeros(len(snr_db_range))
    cap_8x8   = np.zeros(len(snr_db_range))

    for i, snr in enumerate(snr_lin_arr):
        cap_siso[i] = siso_capacity(snr)
        for _ in range(n_chan):
            cap_2x2[i] += mimo_capacity(rayleigh_channel(2, 2), snr)
            cap_4x4[i] += mimo_capacity(rayleigh_channel(4, 4), snr)
            cap_8x8[i] += mimo_capacity(rayleigh_channel(8, 8), snr)
        cap_2x2[i] /= n_chan
        cap_4x4[i] /= n_chan
        cap_8x8[i] /= n_chan

    # ── MASTER FIGURE ──
    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("MIMO Systems — Spatial Multiplexing · ZF/MMSE · Alamouti · Capacity",
                 fontsize=13, fontweight='bold')

    # 1. BER curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.semilogy(snr_db_range, ber_awgn,         'k-',   lw=2,   label='BPSK AWGN (genie)')
    ax1.semilogy(snr_db_range, ber_siso_rayleigh, 'k--',  lw=1.5, label='SISO Rayleigh')
    ax1.semilogy(snr_db_range, ber_2x2_zf,        'b-o',  lw=1.5, ms=5, label='2×2 ZF')
    ax1.semilogy(snr_db_range, ber_2x2_mmse,      'b-s',  lw=1.5, ms=5, label='2×2 MMSE')
    ax1.semilogy(snr_db_range, ber_4x4_zf,        'r-o',  lw=1.5, ms=5, label='4×4 ZF')
    ax1.semilogy(snr_db_range, ber_alamouti,       'g-^',  lw=1.5, ms=5, label='Alamouti 2×1 (diversity)')
    ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("BER")
    ax1.set_title("BER Comparison: SISO vs MIMO Equalisers vs Alamouti Diversity")
    ax1.legend(fontsize=8); ax1.grid(True, which='both', alpha=0.3)
    ax1.set_ylim([1e-4, 1]); ax1.set_xlim([0, 20])

    # 2. Channel H singular values
    ax2 = fig.add_subplot(gs[0, 2])
    H44  = rayleigh_channel(4, 4, seed=1)
    svs  = np.linalg.svd(H44, compute_uv=False)
    ax2.bar(range(1, 5), svs**2, color=['steelblue','darkorange','tomato','mediumseagreen'])
    ax2.set_title("4×4 Channel Singular Values\n(= spatial multiplexing modes)")
    ax2.set_xlabel("Mode k"); ax2.set_ylabel("λₖ = σₖ²"); ax2.grid(alpha=0.3)

    # 3. Capacity curves
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(snr_db_range, cap_siso, 'k-',  lw=2,   label='SISO (1×1)')
    ax3.plot(snr_db_range, cap_2x2,  'b-',  lw=1.8, label='2×2 MIMO')
    ax3.plot(snr_db_range, cap_4x4,  'r-',  lw=1.8, label='4×4 MIMO')
    ax3.plot(snr_db_range, cap_8x8,  'g-',  lw=1.8, label='8×8 MIMO')
    ax3.set_xlabel("SNR (dB)"); ax3.set_ylabel("Capacity (bits/s/Hz)")
    ax3.set_title("MIMO Ergodic Capacity  C = log₂ det(I + ρ/Nₜ · HHᴴ)")
    ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    # 4. ZF vs MMSE noise amplification
    ax4 = fig.add_subplot(gs[1, 2])
    H22  = rayleigh_channel(2, 2, seed=3)
    snr_test = [5, 10, 20]
    zf_gain  = np.linalg.norm(zf_equaliser(H22), 'fro')
    mmse_gains = [np.linalg.norm(mmse_equaliser(H22, 1/10**(s/10), 2), 'fro') for s in snr_test]
    ax4.bar(['ZF', 'MMSE\n5dB', 'MMSE\n10dB', 'MMSE\n20dB'],
            [zf_gain] + mmse_gains,
            color=['tomato','steelblue','steelblue','steelblue'])
    ax4.set_title("Equaliser Frobenius Norm\n(noise amplification proxy)")
    ax4.set_ylabel("‖W‖_F"); ax4.grid(alpha=0.3, axis='y')

    # 5. Alamouti space-time diagram
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.axis('off')
    txt = (
        "           Alamouti 2×1 Space-Time Block Code\n\n"
        "           Antenna 1:   s₁       -s₂*\n"
        "           Antenna 2:   s₂        s₁*\n"
        "                      ───────────────────\n"
        "                        Time t      t+T\n\n"
        "  Combiner:  ŝ₁ = h₁* r₁ + h₂ r₂*\n"
        "             ŝ₂ = h₂* r₁ - h₁ r₂*\n\n"
        "  Diversity order = Nₜ × Nᵣ = 2 → steeper BER slope\n"
        "  No CSI required at transmitter\n"
        "  Rate = 1 (full-rate code for 2 TX antennas)"
    )
    ax5.text(0.05, 0.95, txt, transform=ax5.transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f9ff', alpha=0.9))
    ax5.set_title("Alamouti Code Structure & Combining", fontsize=11)

    # 6. Constellation: ZF vs MMSE
    ax6 = fig.add_subplot(gs[2, 2])
    H_demo = rayleigh_channel(2, 2, seed=5)
    rng = np.random.default_rng(0)
    bits_d = rng.integers(0, 2, 400)
    X_d = spatial_multiplex_tx(bits_d, 2)
    sigma2_d = 0.1
    Y_d = H_demo @ X_d + awgn_noise(2, X_d.shape[1], sigma2_d)
    W_zf   = zf_equaliser(H_demo)
    W_mmse = mmse_equaliser(H_demo, sigma2_d, 2)
    rx_zf   = (W_zf @ Y_d).real.flatten()
    rx_mmse = (W_mmse @ Y_d).real.flatten()
    ax6.scatter(rx_zf,   np.zeros(len(rx_zf))-0.05,  alpha=0.4, s=10, color='tomato',   label='ZF')
    ax6.scatter(rx_mmse, np.zeros(len(rx_mmse))+0.05, alpha=0.4, s=10, color='steelblue', label='MMSE')
    ax6.axvline(0, color='k', lw=1)
    ax6.set_title("ZF vs MMSE — Symbol Scatter\n(SNR=10dB, 2×2)")
    ax6.legend(fontsize=8); ax6.set_xlim([-3, 3])
    ax6.set_yticks([]); ax6.grid(alpha=0.3)

    fig.savefig("mimo_system.png", dpi=130, bbox_inches='tight')
    print("✓ Saved: mimo_system.png")
    print("\n✅ MIMO module complete.")
    plt.close('all')
