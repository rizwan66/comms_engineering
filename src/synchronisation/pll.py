"""
src/synchronisation/pll.py
===========================
Phase-Locked Loop (PLL) implementations:
  - Analog PLL model (2nd order)
  - Digital Costas loop (BPSK carrier recovery)
  - Mueller-Müller timing recovery
  - Full blind receiver demo (no assumed synchronisation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# 1. SECOND-ORDER ANALOG PLL MODEL
# ─────────────────────────────────────────────

class AnalogPLL:
    """
    Second-order analog PLL (discrete-time simulation).

    Parameters
    ----------
    kp        : phase detector gain
    k0        : VCO gain (rad/sample per unit input)
    loop_bw   : normalised loop bandwidth (fraction of sample rate)
    damping   : damping factor ζ (0.707 = critically damped)
    fs        : sample rate
    """

    def __init__(self, kp=1.0, k0=1.0, loop_bw=0.01, damping=0.707, fs=1.0):
        self.kp      = kp
        self.k0      = k0
        self.fs      = fs
        # Design loop filter coefficients from BW and damping
        wn           = loop_bw * 2 * np.pi          # natural frequency
        self.alpha   = 2 * damping * wn              # proportional
        self.beta    = wn**2                          # integral
        # State
        self.phase   = 0.0
        self.freq    = 0.0
        self.integr  = 0.0

    def step(self, input_phase_error):
        """Advance PLL by one sample. Returns updated VCO phase."""
        # Loop filter
        self.integr += self.beta * input_phase_error
        vco_freq     = self.alpha * input_phase_error + self.integr

        # VCO
        self.phase  += vco_freq * self.k0
        self.phase   = (self.phase + np.pi) % (2*np.pi) - np.pi
        return self.phase

    def run(self, phase_errors):
        """Run PLL over a sequence of phase errors."""
        phases = np.zeros(len(phase_errors))
        for i, e in enumerate(phase_errors):
            phases[i] = self.step(e)
        return phases


# ─────────────────────────────────────────────
# 2. COSTAS LOOP — BPSK CARRIER RECOVERY
# ─────────────────────────────────────────────

class CostasLoop:
    """
    Costas loop for BPSK carrier phase recovery.

    The Costas loop multiplies the incoming signal by a local I/Q pair,
    computes a phase error from the cross-product, and drives a VCO to
    lock onto the carrier — without needing to know the transmitted data.

    Parameters
    ----------
    alpha    : proportional gain
    beta     : integral gain
    """

    def __init__(self, alpha=0.05, beta=0.001):
        self.alpha  = alpha
        self.beta   = beta
        self.phase  = 0.0
        self.freq   = 0.0

    def process(self, rx_signal, fc, fs):
        """
        Run Costas loop on received RF signal.
        Returns: (I, Q, phase_track, freq_track)
        """
        N       = len(rx_signal)
        I_out   = np.zeros(N)
        Q_out   = np.zeros(N)
        ph_hist = np.zeros(N)
        fr_hist = np.zeros(N)

        for n in range(N):
            t    = n / fs
            # Mix with local oscillator
            I_lo = np.cos(2*np.pi*fc*t + self.phase)
            Q_lo = np.sin(2*np.pi*fc*t + self.phase)

            I    = rx_signal[n] * I_lo
            Q    = rx_signal[n] * Q_lo

            # Low-pass filter (simple integrator, 1-tap)
            I_out[n] = I
            Q_out[n] = Q

            # Phase error detector: e = sign(I)*Q - sign(Q)*I
            # For BPSK, only I matters: e = sign(I)*Q
            e    = np.sign(I) * Q

            # Loop filter update
            self.freq  += self.beta * e
            self.phase += self.alpha * e + self.freq

            ph_hist[n] = self.phase
            fr_hist[n] = self.freq

        return I_out, Q_out, ph_hist, fr_hist


# ─────────────────────────────────────────────
# 3. MUELLER-MÜLLER TIMING RECOVERY
# ─────────────────────────────────────────────

class MuellerMuller:
    """
    Mueller-Müller clock recovery (decision-directed symbol timing).

    Adjusts sampling phase to align with the optimal sampling instant.
    Works on a 2× oversampled signal.

    Parameters
    ----------
    mu       : fractional sample offset (0.0–1.0)
    alpha    : loop gain
    sps      : samples per symbol (must be ≥ 2)
    """

    def __init__(self, mu=0.0, alpha=0.01, sps=2):
        self.mu     = mu
        self.alpha  = alpha
        self.sps    = sps
        self._prev_samp = 0.0
        self._prev_dec  = 0.0

    def _lerp(self, buf, mu):
        """Linear interpolation between two samples."""
        i = int(mu)
        f = mu - i
        if i + 1 < len(buf):
            return buf[i] * (1 - f) + buf[i+1] * f
        return buf[i]

    def recover(self, signal):
        """
        Run M&M timing recovery.
        Returns (symbols, timing_error_history, mu_history)
        """
        out_syms = []
        te_hist  = []
        mu_hist  = []
        idx      = self.sps

        while idx + self.sps < len(signal):
            # Interpolate at current timing offset
            samp     = self._lerp(signal[idx:], self.mu)
            decision = np.sign(samp)

            # M&M error: e = d[n-1]*x[n] - d[n]*x[n-1]
            te = self._prev_dec * samp - decision * self._prev_samp

            # Update mu
            self.mu  = np.clip(self.mu + self.alpha * te, 0, self.sps - 0.001)

            out_syms.append(samp)
            te_hist.append(te)
            mu_hist.append(self.mu)

            self._prev_samp = samp
            self._prev_dec  = decision

            # Advance by (sps - correction)
            step  = int(self.sps - np.floor(self.mu))
            self.mu = self.mu - np.floor(self.mu)
            idx  += step

        return (np.array(out_syms),
                np.array(te_hist),
                np.array(mu_hist))


# ─────────────────────────────────────────────
# 4. FREQUENCY ERROR DETECTOR (for coarse AFC)
# ─────────────────────────────────────────────

def power_spectrum_freq_error(signal, fc_nominal, fs, search_range=50):
    """
    Estimate carrier frequency error by finding peak of |FFT(signal²)|.
    Squaring removes BPSK modulation (doubles frequency).
    Returns estimated frequency offset in Hz.
    """
    sq    = signal**2                                # remove BPSK data
    N     = len(sq)
    X     = np.abs(np.fft.fft(sq, n=N))
    freqs = np.fft.fftfreq(N, d=1/fs)

    # Search near 2*fc
    target = 2 * fc_nominal
    mask   = (np.abs(freqs - target) < search_range) | \
             (np.abs(freqs + target) < search_range)
    X_masked = X.copy()
    X_masked[~mask] = 0
    f_peak  = freqs[np.argmax(X_masked)]
    return f_peak / 2 - fc_nominal    # frequency error


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(0)
    fs  = 8000
    fc  = 1000
    sps = 8
    bit_rate = fs // sps

    print("=" * 58)
    print("  Synchronisation Demo — PLL · Costas · M&M")
    print("=" * 58)

    # ── Generate BPSK signal with frequency offset + phase offset ──
    n_bits  = 200
    bits    = np.random.randint(0, 2, n_bits)
    bpsk_sym = 1 - 2*bits.astype(float)

    # Upsample and shape (simple rectangular pulse)
    tx_up = np.repeat(bpsk_sym, sps)
    N     = len(tx_up)
    t     = np.arange(N) / fs

    freq_offset  = 15.0    # Hz offset (receiver doesn't know carrier exactly)
    phase_offset = 0.7     # rad offset

    # TX: modulate onto carrier with perfect fc
    tx_rf = tx_up * np.cos(2*np.pi*fc*t)

    # Channel: add freq offset + AWGN
    rx_rf = tx_up * np.cos(2*np.pi*(fc + freq_offset)*t + phase_offset)
    noise = 0.3 * np.random.randn(N)
    rx_noisy = rx_rf + noise

    # ── Costas Loop (carrier recovery) ──
    costas = CostasLoop(alpha=0.08, beta=0.002)
    I_cos, Q_cos, ph_hist, fr_hist = costas.process(rx_noisy, fc, fs)

    # ── M&M Timing Recovery (on I channel) ──
    mm = MuellerMuller(mu=0.0, alpha=0.02, sps=sps)
    syms_mm, te_mm, mu_mm = mm.recover(I_cos)

    # Detected bits
    bits_rx = (syms_mm[sps:] < 0).astype(int)    # skip transient
    n = min(len(bits), len(bits_rx))
    ber = np.sum(bits[:n] != bits_rx[:n]) / n

    print(f"  Freq offset injected : {freq_offset} Hz")
    print(f"  Phase offset injected: {phase_offset:.2f} rad")
    print(f"  BER after PLL+M&M   : {ber:.3f}")

    # ── Second-order PLL on phase-error sequence ──
    ramp_phase = np.linspace(0, 4*np.pi, 1000)   # linearly increasing phase (freq error)
    phase_err  = np.sin(ramp_phase)               # sinusoidal input
    pll2 = AnalogPLL(loop_bw=0.02, damping=0.707)
    pll_out = pll2.run(phase_err)

    # ── PLOTS ──
    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)
    fig.suptitle("Phase-Locked Loop — Carrier & Timing Recovery (Blind Receiver)",
                 fontsize=13, fontweight='bold')

    # 1. TX vs RX (frequency offset visible)
    ax1 = fig.add_subplot(gs[0, :])
    show = slice(0, int(0.02*fs))
    ax1.plot(t[show]*1000, tx_rf[show],    color='steelblue', lw=1.2, label=f'TX (fc={fc}Hz)')
    ax1.plot(t[show]*1000, rx_noisy[show], color='tomato',    lw=0.8, alpha=0.8, label=f'RX (fc+{freq_offset}Hz offset, noisy)')
    ax1.set_title("Received Signal with Carrier Frequency Offset")
    ax1.set_xlabel("ms"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # 2. Costas phase track
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(t*1000, ph_hist, color='darkorange', lw=1.2)
    ax2.axhline(-phase_offset, color='k', ls='--', lw=1, label=f'True offset ({phase_offset:.2f} rad)')
    ax2.set_title("Costas Loop — Phase Error Track"); ax2.set_xlabel("ms")
    ax2.set_ylabel("Phase (rad)"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # 3. Frequency error convergence
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(t*1000, fr_hist * fs / (2*np.pi), color='mediumseagreen', lw=1.2)
    ax3.axhline(-freq_offset, color='k', ls='--', lw=1, label=f'True -{freq_offset}Hz')
    ax3.set_title("Costas — Freq. Error"); ax3.set_xlabel("ms")
    ax3.set_ylabel("Hz"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # 4. M&M timing error
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.plot(te_mm, color='steelblue', lw=0.9)
    ax4.axhline(0, color='k', ls='--', lw=0.8)
    ax4.set_title("Mueller-Müller — Timing Error"); ax4.set_xlabel("Symbol"); ax4.grid(alpha=0.3)

    # 5. Recovered symbols scatter
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.scatter(syms_mm.real, np.zeros(len(syms_mm)),
                c=np.arange(len(syms_mm)), cmap='coolwarm', s=15, alpha=0.6)
    ax5.set_title(f"Recovered Symbols (BER={ber:.3f})")
    ax5.set_xlabel("I"); ax5.axvline(0, color='k', lw=0.5); ax5.grid(alpha=0.3)

    # 6. 2nd-order PLL tracking
    ax6 = fig.add_subplot(gs[3, :2])
    ax6.plot(phase_err, color='tomato',   lw=1.5, label='Input phase error')
    ax6.plot(pll_out,   color='steelblue', lw=1.5, label='PLL output')
    ax6.set_title("2nd-Order PLL Tracking a Sinusoidal Phase Input (loop_bw=0.02, ζ=0.707)")
    ax6.set_xlabel("Sample"); ax6.legend(fontsize=9); ax6.grid(alpha=0.3)

    # 7. PLL block diagram text
    ax7 = fig.add_subplot(gs[3, 2])
    ax7.axis('off')
    diagram = (
        "       PLL Block Diagram\n"
        "  ┌──────────────────────┐\n"
        "  │  Phase   Loop    VCO │\n"
        "  │  Detect──Filter──>───┤\n"
        "  │    ↑              │  │\n"
        "  │    └──────────────┘  │\n"
        "  └──────────────────────┘\n\n"
        " Costas (BPSK carrier):\n"
        "  e[n] = sign(I)·Q\n\n"
        " M&M (timing):\n"
        "  e[n] = d̂[n-1]·x[n]\n"
        "       - d̂[n]·x[n-1]"
    )
    ax7.text(0.05, 0.95, diagram, transform=ax7.transAxes,
             fontsize=9, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))

    fig.savefig("pll_synchronisation.png", dpi=130, bbox_inches='tight')
    print("✓ Saved: pll_synchronisation.png")
    print("\n✅ Synchronisation module complete.")
    plt.close('all')
