"""
simulations/full_chain_demo.py
==============================
Complete DSP system demonstration.
Runs all modules in sequence and produces a combined summary figure.

Usage:
    python simulations/full_chain_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp
from scipy.special import erfc

from signals.generator    import sine_wave, awgn, chirp, compute_fft
from filters.design       import butterworth_lpf, apply_iir, notch_filter, lms_filter
from modulation.schemes   import (fm_modulate, fm_demodulate,
                                   bpsk_modulate, bpsk_demodulate,
                                   qam_modulate, ber_bpsk_theory, ber_bpsk_simulation)
from noise_cancellation.canceller import spectral_subtraction, lms_anc, compute_snr
from transceivers.chain   import (transmitter, receiver, awgn_channel,
                                   raised_cosine_filter, root_raised_cosine_filter)


def run_full_demo():
    rng = np.random.default_rng(42)
    fs  = 8000
    fc  = 1000

    print("=" * 60)
    print("  DSP PROJECT — Full System Demonstration")
    print("  Rizwan66 | github.com/rizwan66/dsp")
    print("=" * 60)

    # ════════════════════════════════════════════════
    # MASTER SUMMARY FIGURE  (4×3 grid)
    # ════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)

    LABEL_KW  = dict(color='white', fontsize=10, fontweight='bold')
    TITLE_KW  = dict(color='#58a6ff', fontsize=11, fontweight='bold', pad=6)
    GRID_KW   = dict(alpha=0.15, color='white')

    def styled_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.grid(**GRID_KW)
        return ax

    # ── Row 0: Signals ──────────────────────────────
    t1, clean = sine_wave(200, 1.0, duration=0.05, fs=fs)
    noisy_sig, _  = awgn(clean, snr_db=10)
    freqs, mag   = compute_fft(clean, fs)

    ax00 = styled_ax(gs[0, 0])
    ax00.plot(t1*1000, clean,     color='#58a6ff', lw=1.5, label='clean')
    ax00.plot(t1*1000, noisy_sig, color='#f78166', lw=0.8, alpha=0.7, label='noisy')
    ax00.set_title("Signals & Noise (200 Hz)", **TITLE_KW)
    ax00.set_xlabel("ms", **LABEL_KW); ax00.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    ax01 = styled_ax(gs[0, 1])
    t_c, ch = chirp(50, 1200, duration=0.5, fs=fs)
    ax01.specgram(ch, NFFT=128, Fs=fs, noverlap=64, cmap='plasma')
    ax01.set_title("Chirp Spectrogram (50→1200 Hz)", **TITLE_KW)
    ax01.set_xlabel("Time (s)", **LABEL_KW); ax01.set_ylabel("Hz", **LABEL_KW)
    ax01.tick_params(colors='#8b949e')

    ax02 = styled_ax(gs[0, 2])
    ax02.plot(freqs, mag, color='#3fb950', lw=1.5)
    ax02.set_title("FFT Spectrum", **TITLE_KW)
    ax02.set_xlabel("Hz", **LABEL_KW); ax02.set_ylabel("|X(f)|", **LABEL_KW)

    # ── Row 1: Filters ──────────────────────────────
    t2, s_low = sine_wave(100, 1.0, duration=0.05, fs=fs)
    t2, s_high= sine_wave(900, 0.8, duration=0.05, fs=fs)
    noisy_f   = s_low + s_high
    b, a      = butterworth_lpf(300, fs, order=5)
    filtered  = apply_iir(b, a, noisy_f)
    w, H      = sp.freqz(b, a, worN=2048, fs=fs)

    ax10 = styled_ax(gs[1, 0])
    ax10.plot(t2*1000, noisy_f,  color='#f78166', lw=1,   label='noisy')
    ax10.plot(t2*1000, filtered, color='#58a6ff', lw=1.5, label='filtered')
    ax10.set_title("Butterworth LPF (cutoff=300 Hz)", **TITLE_KW)
    ax10.set_xlabel("ms", **LABEL_KW); ax10.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    ax11 = styled_ax(gs[1, 1])
    ax11.plot(w, 20*np.log10(np.abs(H)+1e-12), color='#d2a8ff', lw=2)
    ax11.axhline(-3, color='#f78166', ls='--', lw=0.8)
    ax11.set_ylim([-80, 5]); ax11.set_title("Butterworth Freq. Response", **TITLE_KW)
    ax11.set_xlabel("Hz", **LABEL_KW); ax11.set_ylabel("dB", **LABEL_KW)

    # LMS adaptive filter
    N_lms = 2000
    t_l   = np.arange(N_lms) / fs
    desired_lms = np.sin(2*np.pi*150*t_l)
    ref_noise   = np.sin(2*np.pi*50*t_l)
    corrupt_lms = desired_lms + ref_noise
    _, e_lms, _ = lms_filter(ref_noise, corrupt_lms, mu=0.004, num_taps=64)

    ax12 = styled_ax(gs[1, 2])
    ax12.plot(t_l[:500], corrupt_lms[:500], color='#f78166', lw=0.9, alpha=0.8, label='noisy')
    ax12.plot(t_l[:500], e_lms[:500],       color='#3fb950', lw=1.5, label='LMS out')
    ax12.set_title("LMS Adaptive Filter", **TITLE_KW)
    ax12.set_xlabel("Time (s)", **LABEL_KW); ax12.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    # ── Row 2: Modulation ───────────────────────────
    t_msg = np.linspace(0, 0.05, int(fs*0.05))
    msg   = np.sin(2*np.pi*200*t_msg)
    s_fm, _ = fm_modulate(msg, fc=fc, fs=fs, kf=50)
    dem_fm  = fm_demodulate(s_fm, fs)
    dem_fm /= np.max(np.abs(dem_fm) + 1e-8)

    ax20 = styled_ax(gs[2, 0])
    ax20.plot(t_msg*1000, s_fm, color='#e3b341', lw=1)
    ax20.set_title("FM Modulated Signal", **TITLE_KW)
    ax20.set_xlabel("ms", **LABEL_KW)

    # BPSK
    bits_demo = np.array([1,0,1,1,0,1,0,0,1,1])
    s_bpsk, t_bpsk = bpsk_modulate(bits_demo, fc=2000, fs=fs, bit_rate=200)

    ax21 = styled_ax(gs[2, 1])
    ax21.plot(t_bpsk[:len(t_bpsk)//2]*1000, s_bpsk[:len(t_bpsk)//2], color='#79c0ff', lw=1)
    ax21.set_title("BPSK Modulated Signal", **TITLE_KW)
    ax21.set_xlabel("ms", **LABEL_KW)

    # QAM constellation
    bits_q = rng.integers(0, 2, 320)
    _, qam_syms = qam_modulate(bits_q, M=16, fc=fc, fs=fs, bit_rate=100)
    noise_q = rng.normal(0, 0.25, len(qam_syms)) + 1j*rng.normal(0, 0.25, len(qam_syms))
    qam_rx = qam_syms + noise_q

    ax22 = styled_ax(gs[2, 2])
    ax22.scatter(qam_rx.real, qam_rx.imag, alpha=0.5, s=15, color='#d2a8ff')
    ax22.axhline(0, color='white', lw=0.4); ax22.axvline(0, color='white', lw=0.4)
    ax22.set_title("16-QAM Constellation (with noise)", **TITLE_KW)
    ax22.set_xlabel("I", **LABEL_KW); ax22.set_ylabel("Q", **LABEL_KW)
    ax22.set_aspect('equal')

    # ── Row 3: Noise Cancellation + BER ─────────────
    N_nc   = 3000
    t_nc   = np.arange(N_nc) / fs
    sig_nc = np.sin(2*np.pi*300*t_nc)
    ref_nc = np.sin(2*np.pi*50*t_nc)
    cor_nc = sig_nc + 0.8*ref_nc + 0.1*rng.standard_normal(N_nc)
    e_anc, _ = lms_anc(ref_nc, cor_nc, mu=0.002, num_taps=64)

    ax30 = styled_ax(gs[3, 0])
    ax30.plot(t_nc[:600], cor_nc[:600], color='#f78166', lw=1,   label='corrupted')
    ax30.plot(t_nc[:600], e_anc[:600],  color='#3fb950', lw=1.5, label='ANC output')
    ax30.set_title("LMS Active Noise Cancellation", **TITLE_KW)
    ax30.set_xlabel("Time (s)", **LABEL_KW); ax30.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    # Spectral subtraction
    noise_burst = 0.5 * rng.standard_normal(N_nc)
    noisy_speech = sig_nc + noise_burst
    ss_out = spectral_subtraction(noisy_speech, fs, noise_frames=8, alpha=1.5)
    snr_in  = compute_snr(sig_nc, noisy_speech)
    snr_out = compute_snr(sig_nc[:len(ss_out)], ss_out)

    ax31 = styled_ax(gs[3, 1])
    ax31.plot(t_nc[:600], noisy_speech[:600], color='#f78166', lw=1, label=f'noisy {snr_in:.0f}dB')
    ax31.plot(t_nc[:600], ss_out[:600],        color='#3fb950', lw=1.5, label=f'clean {snr_out:.0f}dB')
    ax31.set_title("Spectral Subtraction", **TITLE_KW)
    ax31.set_xlabel("Time (s)", **LABEL_KW); ax31.legend(fontsize=8, facecolor='#161b22', labelcolor='white')

    # BER curve
    snr_arr   = np.arange(0, 14)
    ber_th    = ber_bpsk_theory(snr_arr)
    ber_sim   = ber_bpsk_simulation(snr_arr, num_bits=30000)

    ax32 = styled_ax(gs[3, 2])
    ax32.semilogy(snr_arr, ber_th,  color='#58a6ff', lw=2,   label='Theory')
    ax32.semilogy(snr_arr, ber_sim, color='#f78166', lw=1.5, ls='--', marker='o', ms=4, label='Simulation')
    ax32.set_title("BER vs Eb/N₀ — BPSK AWGN", **TITLE_KW)
    ax32.set_xlabel("Eb/N₀ (dB)", **LABEL_KW); ax32.set_ylabel("BER", **LABEL_KW)
    ax32.legend(fontsize=8, facecolor='#161b22', labelcolor='white')
    ax32.set_ylim([1e-5, 1])
    ax32.grid(which='both', **GRID_KW)

    # ── Master title ────────────────────────────────
    fig.text(0.5, 0.98, "📡  Digital Signal Processing — Complete System Overview",
             ha='center', va='top', fontsize=16, fontweight='bold', color='white')
    fig.text(0.5, 0.96, "Signals · Filters · Modulation · Noise Cancellation · Transceivers",
             ha='center', va='top', fontsize=11, color='#8b949e')
    fig.text(0.5, 0.005, "github.com/rizwan66/dsp", ha='center', fontsize=9, color='#30363d')

    out = "full_system_demo.png"
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d1117')
    print(f"\n✅  Master summary saved → {out}")
    plt.close('all')

    print("\n── Module outputs ──────────────────────────────")
    print("  Run each module individually for detailed plots:")
    print("  python src/signals/generator.py")
    print("  python src/filters/design.py")
    print("  python src/modulation/schemes.py")
    print("  python src/noise_cancellation/canceller.py")
    print("  python src/transceivers/chain.py")
    print("────────────────────────────────────────────────")


if __name__ == "__main__":
    run_full_demo()
