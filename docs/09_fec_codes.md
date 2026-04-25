# Forward Error Correction (FEC) Codes

## 9.1 Why FEC?

Without coding, bit errors caused by noise are accepted. FEC adds **structured redundancy** so the receiver can **detect and correct** errors — effectively improving the link's SNR by the **coding gain**.

```
Channel capacity (Shannon limit):
  C = B · log₂(1 + SNR)  [bps]

Coding theorem: If R_b < C, there exists a code achieving
arbitrarily low BER. FEC gets us close to this limit.

    BER
  10⁻¹ ┤
  10⁻² ┤           uncoded
  10⁻³ ┤          ╱
  10⁻⁴ ┤        ╱
  10⁻⁵ ┤      ╱      coded (with FEC)
  10⁻⁶ ┤    ╱      ╱
       └──────────────────► Eb/N0 [dB]
             ↑        ↑
          waterfall  waterfall
          uncoded    coded
                     ←────→
                    coding gain
                    (~4–8 dB)
```

---

## 9.2 Hamming(7,4) Code

The simplest single-error-correcting code. Adds 3 parity bits to 4 data bits.

```
Codeword structure (7 bits):
  [d₁ d₂ d₃ p₁ d₄ p₂ p₃]

Parity equations:
  p₁ = d₁ ⊕ d₂ ⊕ d₄
  p₂ = d₁ ⊕ d₃ ⊕ d₄
  p₃ = d₂ ⊕ d₃ ⊕ d₄

Generator matrix G (4×7):
  [1 0 0 0 1 1 0]
  [0 1 0 0 1 0 1]
  [0 0 1 0 0 1 1]
  [0 0 0 1 1 1 1]

Parity-check matrix H (3×7):
  [1 1 0 1 1 0 0]
  [1 0 1 1 0 1 0]
  [0 1 1 1 0 0 1]
```

**Syndrome decoding:**

```
s = H · r^T (mod 2)     (r = received codeword)

s = 000: no error
s = 001: error at bit 1
s = 010: error at bit 2
s = 011: error at bit 3
  ...etc  (syndrome directly gives error position)
```

- Code rate: R = 4/7 ≈ 0.571
- Minimum Hamming distance: d_min = 3 → corrects 1 error, detects 2 errors

---

## 9.3 Convolutional Code (Rate 1/2, K=7)

The NASA standard code used in Voyager, CDMA, LTE, DVB.

```
Generators (octal): G1 = 0o171 = 1111001₂
                    G2 = 0o133 = 1011011₂

Trellis state: last K−1 = 6 bits of shift register → 2⁶ = 64 states

For each input bit, output 2 bits (one from each generator).
```

**Trellis diagram (2-state simplified view):**

```
State 00 ──┬──(00)──► 00 ──┬──(00)──►
           │               │
           └──(11)──► 10   └──(11)──►
                      │
State 10 ──┬──(10)──► 01   ...
           │
           └──(01)──► 11
```

Each edge labeled with input/output pair. Viterbi finds the **most likely path** through this trellis.

---

## 9.4 Viterbi Algorithm

Hard-decision Viterbi on a 4-state trellis:

```
Received: 11 10 00 11 (4 pairs)

Time:     0   1   2   3   4

State 00: 0 ─── 2 ─── 2 ─── ?
              ↗     ↘
State 01: ∞ ─── 1 ─── ?
              ↗
State 10: ∞ ─── ? ...
State 11: ∞ ─── ?

Path metric = accumulated Hamming distance between received and expected
Survivor path stored at each state
Traceback from final state → decoded bits
```

**Soft Viterbi (LLR-based):**

```
Branch metric = Σ r_i · c_i   (correlation of received LLR with code bit)

LLR = log( P(bit=0|y) / P(bit=1|y) ) = 2y/σ²  for AWGN

Soft decoding gives ~2 dB gain over hard decoding
```

---

## 9.5 LDPC Codes

Low-Density Parity-Check codes approach the Shannon limit within 0.04 dB (turbo codes: ~0.5 dB, Hamming: ~4 dB away).

**Tanner graph representation:**

```
Variable nodes (bits): v₁  v₂  v₃  v₄  v₅  v₆
                        │╲  │  ╱│  │   │╲  │
                        │ ╲ │ ╱ │  │   │ ╲ │
Check nodes (parity): c₁   c₂   c₃  c₄   c₅

H matrix is SPARSE: ~3–6 ones per column (variable degree)
                         ~6–10 ones per row (check degree)
```

**Belief Propagation (Sum-Product) Algorithm:**

```
Initialize:  L_v = LLR from channel  (prior)

Repeat for T iterations:
  Check → Variable message:
    L_c→v = 2 tanh⁻¹( ∏_{v' ∈ N(c)\v} tanh(L_{v'→c}/2) )

  Variable → Check message:
    L_v→c = L_channel + Σ_{c' ∈ N(v)\c} L_{c'→v}

  Hard decision:
    b̂_v = 0 if L_v > 0, else 1

Stop if H·b̂ = 0 (valid codeword) or T iterations reached
```

**Rate matching (LTE style):**

```
Coded bits ──[sub-block interleaver]──[circular buffer]──► systematic + parity bits

Puncturing: skip some parity bits to achieve higher code rate (e.g. 1/3 → 1/2)
Repetition: repeat bits to achieve lower code rate (for poor channel)
```

---

## 9.6 BER Theory Curves

These are plotted in `examples/ex2_ber_curves.py`.

```
BPSK:    BER = Q(√(2·Eb/N0))

QPSK:    BER = Q(√(2·Eb/N0))    (same as BPSK — 2 bits/symbol but 2× energy)

M-QAM:   BER ≈ (4/k)(1 − 1/√M) · Q(√(6k·Eb / ((M−1)·N0)))
         where k = log₂(M) bits/symbol

Q-function: Q(x) = (1/2)·erfc(x/√2) = P(N(0,1) > x)
```

```
BER vs Eb/N0 [dB]:

10⁻¹ ─────────────────────────────────────
10⁻² ─────────────── 64QAM ──────────────
10⁻³ ──────────── 16QAM ─────────────────   each ~6 dB apart
10⁻⁴ ─────────── QPSK ───────────────────
10⁻⁵ ──────────── BPSK ──────────────────
10⁻⁶ ─────────────────────────────────────
     0   2   4   6   8  10  12  14  16  Eb/N0 (dB)
```

---

## 9.7 Coding Gain Summary

| Code | Rate | d_min | Coding Gain vs uncoded BPSK @ 10⁻⁵ BER |
|------|------|-------|---------------------------------------|
| Uncoded BPSK | 1 | 1 | 0 dB (reference) |
| Hamming(7,4) | 4/7 | 3 | ~1.5 dB |
| Convolutional R=1/2, K=7 | 1/2 | 10 | ~5 dB (soft) |
| Turbo R=1/2 | 1/2 | — | ~7 dB |
| LDPC R=1/2 | 1/2 | — | ~8 dB (within 0.1 dB of Shannon) |

---

## 9.8 Code Usage

```python
from src.fec.channel_coding import LDPC, ConvolutionalCode
from src.coding import hamming74_encode, hamming74_decode, bit_error_rate

# Hamming(7,4)
data_bits = np.array([1, 0, 1, 1])
codeword = hamming74_encode(data_bits)          # 7-bit codeword
# introduce 1 error:
codeword[3] ^= 1
decoded = hamming74_decode(codeword)            # corrects the error
assert np.all(decoded == data_bits)

# Convolutional code + Viterbi
cc = ConvolutionalCode()
info = np.random.randint(0, 2, 100)
encoded = cc.encode(info)                       # 200 bits (rate 1/2)
# pass through AWGN ...
decoded = cc.viterbi_decode(llr_values)         # soft Viterbi

# LDPC
ldpc = LDPC(n=128, k=64)                        # code rate 1/2
tx = ldpc.encode(np.random.randint(0, 2, 64))
# introduce errors ...
rx = ldpc.decode(noisy_llr, iterations=50)      # belief propagation
```
