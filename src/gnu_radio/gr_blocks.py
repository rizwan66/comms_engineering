"""
src/gnu_radio/gr_blocks.py
===========================
GNU Radio integration layer.

This module provides:
  1. Pure-Python DSP blocks that mirror GNU Radio's block API —
     run standalone (no GR install needed) for simulation.
  2. Drop-in GR block wrappers (activated when GNU Radio is installed).
  3. A flowgraph builder that works both in simulation and with real SDR
     hardware (RTL-SDR via osmosdr, USRP via UHD).

Architecture
------------
  [Source]  →  [LPF]  →  [AGC]  →  [Costas PLL]  →  [M&M timing]  →  [Slicer]
   RTL-SDR                                                               Decoded bits
   USRP
   File
   Simulation

GNU Radio installation (Ubuntu/Debian):
  sudo apt install gnuradio python3-osmosdr
  pip install gnuradio-companion

RTL-SDR driver:
  sudo apt install rtl-sdr librtlsdr-dev
  pip install pyrtlsdr

USRP (UHD):
  sudo apt install uhd-host python3-uhd
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Try importing GNU Radio — fall back to simulation mode if not installed
try:
    import gnuradio
    from gnuradio import gr, blocks, analog, filter as gr_filter, digital
    import osmosdr
    GR_AVAILABLE = True
    print("[gr_blocks] GNU Radio detected — hardware mode available")
except ImportError:
    GR_AVAILABLE = False
    print("[gr_blocks] GNU Radio not installed — running in simulation mode")


# ─────────────────────────────────────────────
# 1. STANDALONE BLOCK API (mirrors GR interface)
# ─────────────────────────────────────────────

class Block:
    """Base class for all DSP blocks."""
    def process(self, samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SourceBlock(Block):
    """Signal source: file, sine wave, or random BPSK."""

    def __init__(self, mode='bpsk', fs=2e6, fc=100e3,
                 filename=None, n_samples=None):
        self.mode      = mode
        self.fs        = fs
        self.fc        = fc
        self.filename  = filename
        self.n_samples = n_samples or 65536
        self._offset   = 0

    def process(self, n=None):
        n = n or self.n_samples
        t = (np.arange(n) + self._offset) / self.fs
        self._offset += n

        if self.mode == 'sine':
            return np.exp(1j * 2 * np.pi * self.fc * t).astype(np.complex64)

        elif self.mode == 'bpsk':
            sps    = 8
            n_syms = n // sps
            bits   = np.random.randint(0, 2, n_syms)
            syms   = 1 - 2*bits.astype(float)
            bb     = np.repeat(syms, sps)[:n]
            carrier = np.exp(1j * 2*np.pi*self.fc*t)
            return (bb * carrier).astype(np.complex64)

        elif self.mode == 'file':
            assert self.filename, "filename required for file mode"
            data = np.fromfile(self.filename, dtype=np.complex64)
            chunk = data[self._offset - n : self._offset]
            return chunk if len(chunk) == n else np.zeros(n, dtype=np.complex64)

        else:
            return np.zeros(n, dtype=np.complex64)


class LowPassFilterBlock(Block):
    """FIR low-pass filter."""

    def __init__(self, cutoff_hz, fs, num_taps=101):
        from scipy.signal import firwin
        nyq     = fs / 2
        self.h  = firwin(num_taps, cutoff_hz / nyq, window='hamming')
        self._zi = np.zeros(len(self.h) - 1, dtype=np.complex64)

    def process(self, samples):
        from scipy.signal import lfilter
        out, self._zi = lfilter(self.h, [1.0], samples, zi=self._zi)
        return out.astype(np.complex64)


class DecimatorBlock(Block):
    """Decimate by integer factor."""

    def __init__(self, factor):
        self.factor = factor

    def process(self, samples):
        return samples[::self.factor]


class AGCBlock(Block):
    """Automatic Gain Control — normalise signal power."""

    def __init__(self, target_power=1.0, attack=0.001, decay=0.0001):
        self.target = target_power
        self.attack = attack
        self.decay  = decay
        self.gain   = 1.0

    def process(self, samples):
        out = np.zeros_like(samples)
        for i, s in enumerate(samples):
            pwr       = abs(s)**2 * self.gain**2
            err       = self.target - pwr
            rate      = self.attack if err > 0 else self.decay
            self.gain = max(0.01, self.gain + rate * err)
            out[i]    = s * self.gain
        return out.astype(np.complex64)


class CostasLoopBlock(Block):
    """BPSK Costas loop for carrier phase recovery."""

    def __init__(self, alpha=0.05, beta=0.001):
        self.alpha = alpha
        self.beta  = beta
        self.phase = 0.0
        self.freq  = 0.0

    def process(self, samples):
        out = np.zeros_like(samples)
        for i, s in enumerate(samples):
            corrected = s * np.exp(-1j * self.phase)
            I, Q = corrected.real, corrected.imag
            e    = np.sign(I) * Q          # BPSK phase error
            self.freq  += self.beta * e
            self.phase += self.alpha * e + self.freq
            out[i] = corrected
        return out


class SlicerBlock(Block):
    """Hard-decision slicer for BPSK."""

    def process(self, samples):
        return (samples.real < 0).astype(np.uint8)


class SinkBlock(Block):
    """Collect output samples into a buffer."""

    def __init__(self, maxlen=1_000_000):
        self.buffer = deque(maxlen=maxlen)

    def process(self, samples):
        self.buffer.extend(samples)
        return samples

    def read(self, n=None):
        buf = np.array(self.buffer)
        return buf if n is None else buf[:n]


# ─────────────────────────────────────────────
# 2. FLOWGRAPH RUNNER
# ─────────────────────────────────────────────

class Flowgraph:
    """
    Chain of DSP blocks. Call .run() to push samples through the pipeline.

    Usage
    -----
    fg = Flowgraph()
    fg.add(SourceBlock('bpsk', fs=2e6))
    fg.add(LowPassFilterBlock(200e3, 2e6))
    fg.add(AGCBlock())
    fg.add(CostasLoopBlock())
    fg.add(SlicerBlock())
    sink = SinkBlock()
    fg.add(sink)
    fg.run(n_samples=65536)
    bits = sink.read()
    """

    def __init__(self):
        self.blocks = []

    def add(self, block):
        self.blocks.append(block)
        return self

    def run(self, n_samples=65536, chunk_size=4096):
        """Process n_samples in chunks through the pipeline."""
        source = self.blocks[0]
        rest   = self.blocks[1:]
        processed = 0

        while processed < n_samples:
            n    = min(chunk_size, n_samples - processed)
            data = source.process(n)
            for block in rest:
                data = block.process(data)
            processed += n

        return self


# ─────────────────────────────────────────────
# 3. GNU RADIO FLOWGRAPH (real hardware)
# ─────────────────────────────────────────────

def build_rtlsdr_flowgraph(center_freq=433.92e6, samp_rate=2e6,
                            gain=40, output_file=None):
    """
    Build a GNU Radio flowgraph to receive from RTL-SDR.
    Requires: gnuradio + gr-osmosdr + rtl-sdr dongle.

    Returns the gr.top_block or None if GR not available.
    """
    if not GR_AVAILABLE:
        print("GNU Radio not available. Install with:")
        print("  sudo apt install gnuradio python3-osmosdr")
        print("  sudo apt install rtl-sdr")
        return None

    class RTLSDRFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)

            # RTL-SDR source
            self.src = osmosdr.source(args="numchan=1 rtl=0")
            self.src.set_sample_rate(samp_rate)
            self.src.set_center_freq(center_freq)
            self.src.set_gain(gain)

            # Low-pass filter (channel filter)
            taps = gr_filter.firdes.low_pass(
                gain=1, sampling_freq=samp_rate,
                cutoff_freq=100e3, transition_width=10e3
            )
            self.lpf = gr_filter.fir_filter_ccf(
                decimation=4, taps=taps
            )

            # AGC
            self.agc = analog.agc_cc(
                rate=1e-4, reference=1.0, gain=1.0
            )

            # File sink (IQ data)
            if output_file:
                self.sink = blocks.file_sink(
                    gr.sizeof_gr_complex, output_file
                )
            else:
                self.sink = blocks.null_sink(gr.sizeof_gr_complex)

            # Connect
            self.connect(self.src, self.lpf, self.agc, self.sink)

    return RTLSDRFlowgraph()


def build_bpsk_receiver_flowgraph(samp_rate=2e6, center_freq=433.92e6,
                                   baud_rate=9600):
    """
    Complete BPSK receiver flowgraph for GNU Radio.
    RTL-SDR → filter → BPSK demod → output bits.
    """
    if not GR_AVAILABLE:
        print("GNU Radio not available.")
        return None

    class BPSKReceiverFG(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            sps = int(samp_rate / baud_rate)

            self.src = osmosdr.source(args="numchan=1 rtl=0")
            self.src.set_sample_rate(samp_rate)
            self.src.set_center_freq(center_freq)
            self.src.set_gain(40)

            # Channel filter
            taps = gr_filter.firdes.root_raised_cosine(
                gain=1, sampling_freq=samp_rate,
                symbol_rate=baud_rate, alpha=0.35,
                ntaps=11*sps+1
            )
            self.rrc = gr_filter.fir_filter_ccf(1, taps)

            # BPSK demod (Costas + timing)
            self.bpsk_demod = digital.bpsk_demod(
                samples_per_symbol=sps,
                excess_bw=0.35,
                phase_bw=2*np.pi/100,
                timing_bw=2*np.pi/100,
                mod_code="gray",
                verbose=False,
                log=False
            )

            self.sink = blocks.vector_sink_b()
            self.connect(self.src, self.rrc, self.bpsk_demod, self.sink)

    return BPSKReceiverFG()


# ─────────────────────────────────────────────
# 4. SIMULATION DEMO (no hardware needed)
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 58)
    print("  GNU Radio Integration — Simulation Demo")
    print("  (Pure Python flowgraph, no hardware required)")
    print("=" * 58)

    fs  = 2e6
    fc  = 200e3
    sps = 8

    # Build flowgraph
    fg     = Flowgraph()
    source = SourceBlock('bpsk', fs=fs, fc=fc)
    lpf    = LowPassFilterBlock(cutoff_hz=300e3, fs=fs, num_taps=101)
    agc    = AGCBlock(target_power=1.0)
    costas = CostasLoopBlock(alpha=0.05, beta=0.001)
    slicer = SlicerBlock()
    sink   = SinkBlock(maxlen=500_000)

    fg.add(source).add(lpf).add(agc).add(costas).add(slicer).add(sink)
    fg.run(n_samples=65536, chunk_size=8192)

    bits = sink.read()
    print(f"  Processed samples : 65536")
    print(f"  Output bits       : {len(bits)}")
    print(f"  Ones ratio        : {bits.mean():.3f} (expect ≈0.5)")

    # ── Visualise intermediate stages ──
    n_plot = 4096
    raw    = source.process(n_plot)

    # Stage signals
    lpf2   = LowPassFilterBlock(300e3, fs, 101)
    agc2   = AGCBlock()
    costas2= CostasLoopBlock()

    filtered = lpf2.process(raw.copy())
    gained   = agc2.process(filtered.copy())
    locked   = costas2.process(gained.copy())

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle("GNU Radio-Style Flowgraph — Simulation Pipeline\n"
                 "RTL-SDR → LPF → AGC → Costas PLL → Slicer",
                 fontsize=13, fontweight='bold')

    stages = [
        (raw,      "1. Raw IQ (RTL-SDR input)"),
        (filtered, "2. After Low-Pass Filter"),
        (gained,   "3. After AGC"),
        (locked,   "4. After Costas Loop (carrier locked)"),
    ]

    for i, (sig, title) in enumerate(stages):
        t = np.arange(n_plot) / fs * 1e3
        axes[i, 0].plot(t[:200], sig.real[:200], color='steelblue', lw=0.9, label='I')
        axes[i, 0].plot(t[:200], sig.imag[:200], color='darkorange', lw=0.9, alpha=0.7, label='Q')
        axes[i, 0].set_title(title); axes[i, 0].grid(alpha=0.3)
        if i == 0: axes[i, 0].legend(fontsize=8)

        # IQ constellation
        axes[i, 1].scatter(sig.real[500:1500:2], sig.imag[500:1500:2],
                           alpha=0.4, s=8, color='steelblue')
        axes[i, 1].set_title(f"IQ Constellation — {title.split('.')[1].strip()}")
        axes[i, 1].set_aspect('equal'); axes[i, 1].grid(alpha=0.3)
        axes[i, 1].axhline(0, color='k', lw=0.4); axes[i, 1].axvline(0, color='k', lw=0.4)

    plt.tight_layout()
    fig.savefig("gr_flowgraph_sim.png", dpi=130, bbox_inches='tight')
    print("✓ Saved: gr_flowgraph_sim.png")

    # ── Spectrum at each stage ──
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    fig2.suptitle("Spectrum at Each Pipeline Stage", fontsize=13, fontweight='bold')

    for ax, (sig, title) in zip(axes2.flat, stages):
        freqs = np.fft.fftshift(np.fft.fftfreq(n_plot, 1/fs)) / 1e3
        psd   = np.fft.fftshift(np.abs(np.fft.fft(sig))**2)
        psd_db = 10*np.log10(psd + 1e-10)
        ax.plot(freqs, psd_db, color='steelblue', lw=0.8)
        ax.set_title(title); ax.set_xlabel("kHz"); ax.set_ylabel("dBW")
        ax.grid(alpha=0.3); ax.set_xlim([-fs/2e3, fs/2e3])

    plt.tight_layout()
    fig2.savefig("gr_spectrum_stages.png", dpi=130, bbox_inches='tight')
    print("✓ Saved: gr_spectrum_stages.png")

    # ── GNU Radio connection diagram ──
    fig3, ax3 = plt.subplots(figsize=(14, 4))
    ax3.axis('off')
    ax3.set_title("GNU Radio Flowgraph — RTL-SDR BPSK Receiver", fontsize=12, fontweight='bold')
    blocks_list = [
        ("RTL-SDR\nSource\n2 MHz", '#4c72b0'),
        ("Low-Pass\nFilter\n300 kHz BW", '#dd8452'),
        ("AGC\nPower\nnorm.", '#55a868'),
        ("Costas\nLoop\nBPSK PLL", '#c44e52'),
        ("M&M\nTiming\nRecovery", '#8172b2'),
        ("Hard\nDecision\nSlicer", '#937860'),
        ("Output\nBits", '#da8bc3'),
    ]
    x = 0
    for i, (label, color) in enumerate(blocks_list):
        rect = plt.Rectangle((x, 0.3), 1.6, 0.4, color=color, alpha=0.85)
        ax3.add_patch(rect)
        ax3.text(x+0.8, 0.5, label, ha='center', va='center',
                 fontsize=8, color='white', fontweight='bold')
        if i < len(blocks_list)-1:
            ax3.annotate("", xy=(x+1.75, 0.5), xytext=(x+1.6, 0.5),
                         arrowprops=dict(arrowstyle='->', color='black', lw=2))
        x += 2.0

    ax3.set_xlim(-0.2, x); ax3.set_ylim(0, 1)
    fig3.savefig("gr_block_diagram.png", dpi=130, bbox_inches='tight')
    print("✓ Saved: gr_block_diagram.png")

    print("\n✅ GNU Radio integration module complete.")
    print("\nTo use with real RTL-SDR hardware:")
    print("  fg = build_rtlsdr_flowgraph(center_freq=433.92e6)")
    print("  fg.start(); fg.wait()")
    plt.close('all')
