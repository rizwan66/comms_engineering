# Block Diagrams & Signal Flow Diagrams

Signal processing "circuit diagrams" for all major subsystems. Each block represents a mathematical operation on the signal.

---

## 1. IQ (In-Phase / Quadrature) Modulator

The IQ architecture is the foundation of every modern radio transmitter. Baseband I and Q channels modulate two 90°-shifted carriers.

```
Baseband I(t) ──────────────[× cos(2πfct)]──────────────┐
                                                          [+]──► RF s(t)
Baseband Q(t) ──────────────[× sin(2πfct)]──────────────┘
                                    ↑
                            90° phase shift
                            from same oscillator
```

**IQ Demodulator (coherent receiver):**

```
RF r(t) ──┬──[× 2cos(2πfct)]──[LPF]──► I(t)  (in-phase)
          │
          └──[× 2sin(2πfct)]──[LPF]──► Q(t)  (quadrature)
```

**Complex representation:**  `s(t) = I(t)cos(2πfct) − Q(t)sin(2πfct)`

---

## 2. AM / DSB-SC / SSB / FM Modulators

### AM Modulator
```
m(t) ──[+ Ac]──[× cos(2πfct)]──► s_AM(t) = Ac[1 + μ·mn(t)]cos(2πfct)
                   ↑
              carrier osc.
                                  μ = modulation index ∈ (0, 1]
```

### DSB-SC Modulator (suppressed carrier)
```
m(t) ──[× cos(2πfct)]──► s_DSB(t) = m(t)·cos(2πfct)
            ↑
       carrier osc.                  (no DC term — more power-efficient)
```

### SSB Modulator (Hilbert method)
```
m(t) ──────────────────────────────────────────┐
      └──[Hilbert Transform]──[×±sin(2πfct)]──[+]──► s_SSB(t)
                                   ↑
                              carrier osc.     (+ → USB, − → LSB)
```

### FM Modulator
```
m(t) ──[∫ dt  (cumulative sum)]──[× 2πkf]──[exp(j·φ(t))]──► s_FM(t)
                                                  ↑
                              φ(t) = 2πfc·t + 2πkf∫m(τ)dτ
```

---

## 3. Digital Modulation Constellations

### BPSK (Binary PSK)
```
        Q
        │
  −1 ───┼─── +1    I-axis only
        │           BER = Q(√(2Eb/N0))
```

### QPSK (Quadrature PSK — Gray coded)
```
        Q
    01  │  00
  ──────┼──────  I
    11  │  10
        │
       45° rotation from axes
       BER = Q(√(2Eb/N0))  (same as BPSK per bit)
```

### 16-QAM
```
        Q
  ● ● ● ●
  ● ● ● ●   4×4 square grid
  ● ● ● ●   Gray coded along rows and columns
  ● ● ● ●   BER ≈ (3/8)·Q(√(4Eb/(5N0)))
        I
```

### 64-QAM
```
  8×8 grid — 6 bits per symbol
  Requires higher SNR (~26 dB for 10⁻⁶ BER)
```

---

## 4. Root Raised Cosine Pulse Shaping Chain

The RRC split between TX and RX eliminates inter-symbol interference (ISI).

```
Symbols ──[Upsample Lsps]──[RRC filter h_T(t)]──► shaped signal
                                 TX                     │
                                                   Channel + noise
                                                         │
Symbols ◄──[Downsample Lsps]──[RRC filter h_R(t)]──────┘
                                 RX

h_T(t) * h_R(t) = RC(t)  → zero ISI at sampling instants
```

**RRC impulse response:**
```
       1 + (4β/π)cos((1+β)πt/T) + sin((1−β)πt/T) / (4βt/T)
h(t) = ──────────────────────────────────────────────────────
                    π√T · [1 − (4βt/T)²]

β = roll-off factor (0 → sinc, 1 → widest)
```

---

## 5. OFDM Transceiver Block Diagram

```
                    ┌─────────────── OFDM TRANSMITTER ────────────────┐

Bits ──[Serial to Parallel]──► N_data symbols per OFDM symbol
                                      │
                              [QAM Mapper: QPSK/16-QAM/64-QAM]
                                      │
                              [Insert Pilots at every P-th subcarrier]
                                      │
                              [Zero-pad unused subcarriers]
                                      │
                              [N_fft-point IFFT]
                                      │
                              [Prepend Cyclic Prefix (CP length = N_cp)]
                                      │
                              [Parallel to Serial] ──► x[n] (time domain)
                                                            │
                                                     [Channel h[n] + w[n]]
                                                            │
                    ┌─────────────── OFDM RECEIVER ─────────────────┐
                                                            │
                              [Serial to Parallel]  ◄───────┘
                                      │
                              [Remove Cyclic Prefix]
                                      │
                              [N_fft-point FFT]  ──► Y[k] frequency domain
                                      │
                              [LS Channel Estimation at pilot positions]
                              H_est[k] = Y[k] / P[k]
                                      │
                              [Interpolate H across all subcarriers]
                                      │
                              [MMSE Equalization]
                              X̂[k] = H*(k)·Y[k] / (|H(k)|² + σ²)
                                      │
                              [QAM DeMapper → soft / hard bits]
                                      │
                              [Parallel to Serial] ──► Bits out
```

---

## 6. MIMO 2×2 System (Spatial Multiplexing)

```
         Transmitter                    Channel                  Receiver
         ──────────                  ──────────                 ────────

Bits ─[S/P]─ s₁ ──────────────── h₁₁ ────────────────────┐
              │  ╲              ╱                           ├──► r₁ ──┐
              │   ╲ h₂₁     h₁₂╱                           │         │
              │    ╲         ╱                              │         [W]──► ŝ₁, ŝ₂
              │     ╲       ╱                               │         │
             s₂ ──── h₂₂ ──────────────────────────────────┘──► r₂ ──┘

Matrix form: r = H·s + n      [r₁]   [h₁₁ h₁₂][s₁]   [n₁]
                               [r₂] = [h₂₁ h₂₂][s₂] + [n₂]

ZF equalizer:   W_ZF  = (H^H H)^-1 H^H
MMSE equalizer: W_MMSE = (H^H H + σ²I)^-1 H^H
```

---

## 7. Alamouti Space-Time Block Code (2×1)

Encodes two symbols s₁, s₂ across two antennas and two time slots — achieving full diversity at full rate.

```
Time slot:        t           t+T
Antenna 1:        s₁         −s₂*
Antenna 2:        s₂          s₁*

          ┌──────────────────────────────────────────┐
          │  Tx Ant 1: [s₁]──────────────[−s₂*]     │
Encoder:  │                                           │ ──► h₁, h₂ → single Rx
          │  Tx Ant 2: [s₂]──────────────[ s₁*]     │
          └──────────────────────────────────────────┘
```

**Combined received signal at two time instants:**
```
r₁ = h₁s₁ + h₂s₂ + n₁
r₂ = −h₁s₂* + h₂s₁* + n₂

After MRC combining → separates s₁ and s₂ with full 2nd-order diversity
```

---

## 8. PLL (Phase-Locked Loop) Block Diagram

```
                    ┌──────────────────────────────────────────────┐
                    │                                              │
Input x(t)         ↓                                              │
──────────[Phase Detector]──[Loop Filter F(s)]──[VCO]──► Output  │
          (PD: multiply,                          ↑               │
           then LPF)                   v_tune = kv∫e(t)dt         │
                                                  │               │
                                                  └───────────────┘
                                                 feedback phase θ_out

2nd-order PLL loop filter: F(s) = (1 + s·τ₂) / (s·τ₁)

Natural frequency: ωn = √(Kd·Ko / τ₁)
Damping factor:    ζ  = (ωn·τ₂) / 2
```

### Costas Loop (Carrier Recovery for BPSK)

```
r(t)·cos(θ̂) ──[LPF]──► I(t) ──[sgn(·)]──────────[×]──► e(t) = sgn(I)·Q
     │                                               ↑
     └──[×−sin(θ̂)]──[LPF]──► Q(t) ─────────────────┘

e(t) → [Loop Filter] → [NCO] → θ̂(t)  (phase estimate)
```

---

## 9. Mueller-Müller Timing Recovery

Operates on 2× oversampled signal — extracts symbol clock without explicit pilot tones.

```
Input y[n] (2× sps) ──[Slicer D[n]]──────────────────────────────┐
         │                                                         │
         │         e[n] = y[n-1]·D[n] − y[n]·D[n-1]             │
         │         (error proportional to timing offset)          │
         │                                                         │
         └──[Loop Filter μ_loop]──[Interpolator]──► Symbol out    │
                                         ↑                        │
                                   timing adjust ◄────────────────┘
```

---

## 10. Spectral Subtraction Noise Canceller

```
Noisy signal y[n] ──[Frame + Window]──[FFT]──► Y(ω)
                                                 │
                                          |Ŷ(ω)| = max(|Y(ω)| − α·√N̂(ω), β·|Y(ω)|)
                                                 ↑
Noise-only frames ──[FFT + avg]──► N̂(ω)  (noise PSD estimate)

Cleaned spectrum ──[IFFT + Overlap-Add]──► x̂[n]

α = over-subtraction factor (>1 more aggressive)
β = spectral floor (prevents musical noise)
```

---

## 11. Wiener Filter (Optimal Linear Filter)

```
Observation:  y[n] = x[n] + v[n]    (desired x, noise v)

Wiener solution in frequency domain:
        Sxx(ω)
H(ω) = ─────────────    where Sxx = signal PSD, Svv = noise PSD
        Sxx(ω) + Svv(ω)

H(ω) → 1 when Sxx >> Svv  (pass signal)
H(ω) → 0 when Svv >> Sxx  (suppress noise)
```

---

## 12. Kalman Filter Predict-Update Cycle

```
┌─────────────────── PREDICT ───────────────────┐
│                                               │
│  x̂⁻[k] = F·x̂[k−1]           (state predict) │
│  P⁻[k]  = F·P[k−1]·Fᵀ + Q   (cov. predict)  │
│                                               │
└──────────────────────────────── ↓ ────────────┘
                                  │
                           measurement z[k]
                                  │
┌─────────────────── UPDATE ────────────────────┐
│                                               │
│  K[k] = P⁻·Hᵀ / (H·P⁻·Hᵀ + R)   (gain)     │
│  x̂[k] = x̂⁻ + K·(z[k] − H·x̂⁻)  (update)    │
│  P[k]  = (I − K·H)·P⁻            (cov. upd.) │
│                                               │
└───────────────────────────────────────────────┘
         ↓
  x̂[k], P[k]  →  feed back to PREDICT at k+1
```

---

## 13. Convolutional Encoder (Rate 1/2, K=7)

NASA standard (Voyager): generators G1 = 1111001 (0o171), G2 = 1011011 (0o133)

```
Input bit ──►[D]──►[D]──►[D]──►[D]──►[D]──►[D]──┐  shift register
              │     │     │     │     │     │     │  (K−1 = 6 stages)
              ▼     ▼     ▼     ▼     ▼     ▼     ▼
G1: ─────────[⊕]──────────────[⊕]──────────────[⊕]──► output bit 0
                                                        (G1 taps)
G2: ─────────[⊕]─────────────[⊕]──[⊕]──────────────── output bit 1
                                                        (G2 taps)

Each input bit → 2 output bits → rate 1/2
Constraint length K=7 → 64 states in Viterbi trellis
```

---

## 14. LDPC Tanner Graph

Bipartite graph connecting variable nodes (bits) to check nodes (parity equations).

```
Variable nodes (codeword bits):
  v₁  v₂  v₃  v₄  v₅  v₆  v₇  ...  vN

   │╲  │  ╱│  │╲  │
   │  ╲│╱  │  │  ╲│
  c₁   c₂   c₃   c₄  ...  cM   ← Check nodes (parity equations)

Each check cᵢ: XOR of its connected variable nodes = 0
Belief propagation passes LLR messages along edges iteratively.
```

---

## 15. OTFS Delay-Doppler Domain Processing

OTFS maps information symbols onto the **delay-Doppler (DD) grid** — a 2D domain that is sparse and stable for fast-moving channels.

```
          ┌──── OTFS TRANSMITTER ────┐

X_DD[l,k] ──[ISFFT]──► X_TF[n,m]  (Time-Frequency domain)
                              │
                    [Heisenberg Transform]
                         (IFFT per slot)
                              │
                           s(t)  (transmitted time signal)
                              │
                    ┌── Doubly-dispersive channel ──┐
                    │  h(τ, ν) = Σ hᵢ δ(τ−τᵢ) δ(ν−νᵢ) │
                    └──────────────────────────────┘
                              │
                           r(t)  (received time signal)
                              │
          ┌──── OTFS RECEIVER ────────┐
                              │
                    [Wigner Transform]
                         (FFT per slot)
                              │
                        Y_TF[n,m]
                              │
                    [SFFT]──► Y_DD[l,k]
                              │
                    [MMSE Equalization in DD domain]
                              │
                        X̂_DD[l,k]  → decoded bits

ISFFT: Inverse Symplectic Finite Fourier Transform
SFFT:  Symplectic Finite Fourier Transform
```

**Why DD domain beats OFDM at high mobility:**
- Channel appears as a sparse 2D impulse in DD domain: easy to estimate
- OFDM sees a dense, time-varying frequency response: hard to track
