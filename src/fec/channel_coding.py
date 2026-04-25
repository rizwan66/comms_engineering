"""
src/fec/channel_coding.py
==========================
Modern Forward Error Correction (FEC):
  - LDPC (Low-Density Parity-Check) with belief propagation decoding
  - Turbo code (parallel concatenated convolutional) with iterative decoding
  - Rate-1/2 convolutional code with Viterbi decoder
  - BER vs Eb/N0 curves showing approach to Shannon limit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


# ─────────────────────────────────────────────
# 1. LDPC — BELIEF PROPAGATION
# ─────────────────────────────────────────────

class LDPC:
    """
    Simple irregular LDPC code using belief propagation (sum-product).
    Generates a random sparse parity-check matrix H of size (M x N).

    Parameters
    ----------
    N      : codeword length (number of variable nodes)
    M      : number of check nodes  (rate = 1 - M/N)
    d_v    : variable node degree
    d_c    : check node degree
    seed   : RNG seed for reproducibility
    """

    def __init__(self, N=256, M=128, d_v=3, d_c=6, seed=0):
        self.N, self.M = N, M
        self.rate = 1 - M / N
        self.H    = self._make_H(N, M, d_v, d_c, seed)
        # Precompute neighbour lists
        self.cn_nbrs = [np.where(self.H[m, :] == 1)[0] for m in range(M)]  # check→var
        self.vn_nbrs = [np.where(self.H[:, n] == 1)[0] for n in range(N)]  # var→check

    def _make_H(self, N, M, d_v, d_c, seed):
        """Construct approximate regular LDPC H via random permutation."""
        rng = np.random.default_rng(seed)
        H   = np.zeros((M, N), dtype=np.int8)
        # Each variable node has degree d_v
        for n in range(N):
            rows = rng.choice(M, size=d_v, replace=False)
            H[rows, n] = 1
        return H

    def encode(self, info_bits):
        """
        Systematic encoding (approximate): pad info_bits with zeros as parity.
        For a proper code you'd use Gaussian elimination; here we demonstrate
        the decoding pipeline with a known codeword = [info | 0...0] and
        correct via BP.
        """
        k = self.N - self.M
        assert len(info_bits) == k, f"Need {k} info bits, got {len(info_bits)}"
        cw = np.zeros(self.N, dtype=int)
        cw[:k] = info_bits
        # Simple XOR parity fill (not a true systematic encoder — demo only)
        for m in range(self.M):
            nbrs = self.cn_nbrs[m]
            data_nbrs = nbrs[nbrs < k]
            if len(data_nbrs):
                cw[k + m % self.M] ^= int(np.sum(cw[data_nbrs]) % 2)
        return cw

    def decode(self, llr, max_iter=50):
        """
        Sum-product (belief propagation) LDPC decoder.

        llr : log-likelihood ratios, shape (N,)
              LLR = log P(c=0|y) / P(c=1|y)
              For BPSK/AWGN: LLR = 2*y/σ²
        Returns hard-decision bits.
        """
        N, M = self.N, self.M
        # Message arrays
        msg_v2c = np.zeros((M, N))   # variable → check messages
        msg_c2v = np.zeros((M, N))   # check    → variable messages

        # Initialise variable→check messages with channel LLRs
        for n in range(N):
            for m in self.vn_nbrs[n]:
                msg_v2c[m, n] = llr[n]

        for _ in range(max_iter):
            # Check node update (tanh rule)
            for m in range(M):
                nbrs = self.cn_nbrs[m]
                for n in nbrs:
                    others = nbrs[nbrs != n]
                    if len(others) == 0:
                        msg_c2v[m, n] = 0.0
                        continue
                    # product of tanh(v/2) for all other neighbours
                    prod = np.prod(np.tanh(np.clip(msg_v2c[m, others] / 2,
                                                    -10, 10)))
                    msg_c2v[m, n] = 2 * np.arctanh(np.clip(prod, -0.9999, 0.9999))

            # Variable node update
            for n in range(N):
                nbrs = self.vn_nbrs[n]
                total = llr[n] + np.sum(msg_c2v[nbrs, n])
                for m in nbrs:
                    msg_v2c[m, n] = total - msg_c2v[m, n]

            # Marginals and hard decision
            marginals = llr.copy()
            for n in range(N):
                marginals[n] += np.sum(msg_c2v[self.vn_nbrs[n], n])

            bits = (marginals < 0).astype(int)

            # Check syndrome
            syndrome = self.H @ bits % 2
            if np.all(syndrome == 0):
                break

        return bits


# ─────────────────────────────────────────────
# 2. CONVOLUTIONAL CODE + VITERBI
# ─────────────────────────────────────────────

class ConvolutionalCode:
    """
    Rate-1/2 convolutional code, constraint length K=3.
    Generator polynomials: g1=0b111 (7), g2=0b101 (5)
    (Standard NASA code used in early satellites)
    """

    def __init__(self):
        self.K  = 3          # constraint length
        self.g1 = 0b111      # generator 1
        self.g2 = 0b101      # generator 2
        self.n_states = 2 ** (self.K - 1)

    def _output(self, state, bit):
        """Compute coded output bits for (state, input bit)."""
        reg = (bit << (self.K-1)) | state
        o1  = bin(reg & self.g1).count('1') % 2
        o2  = bin(reg & self.g2).count('1') % 2
        return o1, o2

    def _next_state(self, state, bit):
        return ((state >> 1) | (bit << (self.K-2))) & (self.n_states - 1)

    def encode(self, bits):
        """Encode bits → coded bits (rate 1/2, length 2*(N+K-1))."""
        state  = 0
        coded  = []
        for b in bits:
            o1, o2 = self._output(state, b)
            coded += [o1, o2]
            state  = self._next_state(state, b)
        # Flush encoder (tail bits)
        for _ in range(self.K - 1):
            o1, o2 = self._output(state, 0)
            coded += [o1, o2]
            state  = self._next_state(state, 0)
        return np.array(coded, dtype=int)

    def viterbi_decode(self, soft_rx, sigma2=1.0):
        """
        Viterbi decoder with soft-input metrics (AWGN branch metric).
        soft_rx : received soft values (±1 BPSK + noise, interleaved I/Q)
        """
        n_sym   = len(soft_rx) // 2
        S       = self.n_states
        INF     = 1e9

        path_metric  = np.full(S, INF)
        path_metric[0] = 0.0
        survivors    = np.zeros((n_sym, S), dtype=int)
        prev_states  = np.zeros((n_sym, S), dtype=int)

        for t in range(n_sym):
            r1, r2   = soft_rx[2*t], soft_rx[2*t+1]
            new_pm   = np.full(S, INF)
            new_prev = np.zeros(S, dtype=int)

            for s in range(S):
                for bit in [0, 1]:
                    o1, o2   = self._output(s, bit)
                    ns       = self._next_state(s, bit)
                    # AWGN branch metric: squared Euclidean distance
                    c1, c2   = 1 - 2*o1, 1 - 2*o2   # BPSK: 0→+1, 1→-1
                    metric   = (r1 - c1)**2 + (r2 - c2)**2
                    candidate = path_metric[s] + metric
                    if candidate < new_pm[ns]:
                        new_pm[ns]   = candidate
                        new_prev[ns] = s
                        survivors[t, ns] = bit

            path_metric = new_pm
            prev_states[t] = new_prev

        # Traceback
        state = int(np.argmin(path_metric))
        bits  = np.zeros(n_sym, dtype=int)
        for t in range(n_sym - 1, -1, -1):
            bits[t] = survivors[t, state]
            state   = prev_states[t, state]

        return bits[:n_sym - (self.K - 1)]


# ─────────────────────────────────────────────
# 3. TURBO CODE (simplified PCCC)
# ─────────────────────────────────────────────

class TurboCode:
    """
    Simplified Parallel Concatenated Convolutional Code (PCCC) turbo code.
    Uses two rate-1/2 RSC encoders and iterative SISO decoding (BCJR-lite).
    For clarity, we use the convolutional code with a random interleaver and
    implement log-MAP decoding as LLR combining (simplified turbo iterations).
    """

    def __init__(self, n_iter=6):
        self.cc     = ConvolutionalCode()
        self.n_iter = n_iter

    def _interleave(self, bits, perm):
        return bits[perm]

    def encode(self, bits):
        """
        Turbo encode: systematic + parity1 (original) + parity2 (interleaved).
        Rate ≈ 1/3.
        """
        N    = len(bits)
        perm = np.random.permutation(N)
        self.perm    = perm
        self.inv_perm = np.argsort(perm)

        p1 = self.cc.encode(bits)[:2*N:2]       # parity bits from encoder 1
        p2 = self.cc.encode(bits[perm])[:2*N:2] # parity bits from encoder 2

        # Transmit: [sys, p1, p2] interleaved
        out = np.zeros(3*N, dtype=int)
        out[0::3] = bits
        out[1::3] = p1[:N]
        out[2::3] = p2[:N]
        return out

    def decode(self, rx_llr3, noise_var=1.0):
        """Simplified turbo decoding: iterative LLR exchange."""
        N = len(rx_llr3) // 3
        L_sys  = rx_llr3[0::3]      # systematic LLRs
        L_p1   = rx_llr3[1::3]      # parity 1 LLRs
        L_p2   = rx_llr3[2::3]      # parity 2 LLRs

        Le1 = np.zeros(N)           # extrinsic from decoder 1
        Le2 = np.zeros(N)           # extrinsic from decoder 2

        for _ in range(self.n_iter):
            # Decoder 1: combine sys + Le2 + p1
            in1  = L_sys + Le2 + L_p1
            out1 = np.tanh(np.clip(in1 / 2, -8, 8))
            Le1  = np.clip(2 * np.arctanh(np.clip(out1, -0.9999, 0.9999)),
                           -10, 10) - Le2

            # Decoder 2: interleaved sys + Le1 + p2
            in2_sys = (L_sys + Le1)[self.perm]
            in2  = in2_sys + L_p2
            out2 = np.tanh(np.clip(in2 / 2, -8, 8))
            Le2_int = np.clip(2 * np.arctanh(np.clip(out2, -0.9999, 0.9999)),
                              -10, 10) - in2_sys
            Le2  = Le2_int[self.inv_perm]

        # Final decision
        L_total = L_sys + Le1 + Le2
        return (L_total < 0).astype(int)


# ─────────────────────────────────────────────
# 4. SHANNON LIMIT
# ─────────────────────────────────────────────

def shannon_limit_ebn0(rate):
    """
    Shannon limit: minimum Eb/N0 for reliable communication at given rate.
    C/B = rate = log2(1 + SNR) → SNR = 2^rate - 1
    Eb/N0 = SNR / rate  (in linear)
    """
    snr  = 2**rate - 1
    ebn0 = snr / rate
    return 10 * np.log10(ebn0)


def ber_uncoded_bpsk(ebn0_db):
    return 0.5 * erfc(np.sqrt(10**(np.array(ebn0_db)/10)))


def ber_conv_simulation(cc, ebn0_db_range, n_bits=500):
    """Simulate BER for convolutional code with Viterbi."""
    rng  = np.random.default_rng(1)
    bers = []
    for ebn0_db in ebn0_db_range:
        bits    = rng.integers(0, 2, n_bits)
        coded   = cc.encode(bits)
        bpsk    = 1 - 2 * coded.astype(float)     # 0→+1, 1→-1
        # Rate 1/2 code: Eb/N0 = Es/N0 / 2
        sigma   = np.sqrt(1 / (2 * 10**(ebn0_db/10)))
        rx      = bpsk + rng.normal(0, sigma, len(bpsk))
        decoded = cc.viterbi_decode(rx)
        n = min(len(bits), len(decoded))
        bers.append(max(np.sum(bits[:n] != decoded[:n]) / n, 1e-6))
    return np.array(bers)


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 58)
    print("  FEC Demo — Convolutional · LDPC · Turbo")
    print("=" * 58)

    ebn0_range = np.arange(0, 10, 1.0)

    # ── Uncoded BPSK ──
    ber_uncoded = ber_uncoded_bpsk(ebn0_range)

    # ── Convolutional + Viterbi ──
    print("Running convolutional code simulation...")
    cc      = ConvolutionalCode()
    ber_conv = ber_conv_simulation(cc, ebn0_range, n_bits=800)

    # ── LDPC (demonstration: encode/decode one block) ──
    print("Running LDPC demo block...")
    ldpc   = LDPC(N=128, M=64, d_v=3, d_c=6, seed=0)
    k      = ldpc.N - ldpc.M
    rng    = np.random.default_rng(5)
    ber_ldpc = []
    for ebn0_db in ebn0_range:
        errs = 0; total = 0
        for _ in range(8):
            info = rng.integers(0, 2, k)
            cw   = ldpc.encode(info)
            bpsk = 1 - 2*cw.astype(float)
            sigma = np.sqrt(1 / (2 * 10**(ebn0_db/10) * ldpc.rate))
            rx   = bpsk + rng.normal(0, sigma, ldpc.N)
            llr  = 2 * rx / sigma**2
            dec  = ldpc.decode(llr, max_iter=30)
            errs  += np.sum(info != dec[:k])
            total += k
        ber_ldpc.append(max(errs / total, 1e-6))
    ber_ldpc = np.array(ber_ldpc)

    # ── Shannon limit ──
    shannon_r12 = shannon_limit_ebn0(0.5)   # rate-1/2
    shannon_r13 = shannon_limit_ebn0(1/3)   # rate-1/3
    shannon_r1  = shannon_limit_ebn0(1.0)   # rate-1 (uncoded)

    print(f"  Shannon limit (rate 1/2) : {shannon_r12:.2f} dB")
    print(f"  Shannon limit (rate 1/3) : {shannon_r13:.2f} dB")
    print(f"  Shannon limit (rate 1)   : {shannon_r1:.2f} dB")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.semilogy(ebn0_range, ber_uncoded, 'k-',   lw=2,   label='Uncoded BPSK')
    ax.semilogy(ebn0_range, ber_conv,    'b--o',  lw=1.8, ms=6, label='Conv. Code K=3, r=1/2 + Viterbi')
    ax.semilogy(ebn0_range, ber_ldpc,    'g--s',  lw=1.8, ms=6, label=f'LDPC N={ldpc.N}, r={ldpc.rate:.2f} + BP')

    # Shannon limits
    ax.axvline(shannon_r1,  color='k',  ls=':', lw=1.5, label=f'Shannon r=1 ({shannon_r1:.1f} dB)')
    ax.axvline(shannon_r12, color='b',  ls=':', lw=1.5, label=f'Shannon r=1/2 ({shannon_r12:.1f} dB)')
    ax.axvline(shannon_r13, color='g',  ls=':', lw=1.5, label=f'Shannon r=1/3 ({shannon_r13:.1f} dB)')

    ax.fill_betweenx([1e-5, 1], -2, shannon_r13, alpha=0.05, color='red',
                     label='Shannon limit region (impossible)')

    ax.set_xlabel("Eb/N₀ (dB)", fontsize=12)
    ax.set_ylabel("Bit Error Rate", fontsize=12)
    ax.set_title("FEC Performance vs Shannon Capacity Limit\n"
                 "Convolutional + Viterbi · LDPC + Belief Propagation", fontsize=12)
    ax.set_xlim([-1, 9]); ax.set_ylim([1e-4, 1])
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    fig.savefig("fec_ber_shannon.png", dpi=130, bbox_inches='tight')
    print("✓ Saved: fec_ber_shannon.png")

    # ── Tanner graph illustration ──
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.set_title(f"LDPC Tanner Graph (N={ldpc.N}, M={ldpc.M}, partial view — first 12 nodes)",
                  fontsize=12)
    ax2.set_xlim(-1, 13); ax2.set_ylim(-1, 3)
    ax2.axis('off')

    show_v = 12; show_c = 6
    for n in range(show_v):
        ax2.plot(n, 2, 'o', color='steelblue', ms=18, zorder=3)
        ax2.text(n, 2, str(n), ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    for m in range(show_c):
        ax2.plot(m*2 + 0.5, 0, 's', color='tomato', ms=18, zorder=3)
        ax2.text(m*2 + 0.5, 0, f'c{m}', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    for m in range(show_c):
        for n in ldpc.cn_nbrs[m]:
            if n < show_v:
                ax2.plot([n, m*2+0.5], [2, 0], 'gray', lw=0.8, alpha=0.5, zorder=1)

    ax2.text(6, 2.6, "Variable Nodes (bits)", ha='center', fontsize=11, color='steelblue', fontweight='bold')
    ax2.text(6, -0.6, "Check Nodes (parity)", ha='center', fontsize=11, color='tomato', fontweight='bold')
    fig2.savefig("fec_tanner_graph.png", dpi=130, bbox_inches='tight')
    print("✓ Saved: fec_tanner_graph.png")

    print("\n✅ FEC module complete.")
    plt.close('all')
