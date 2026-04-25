# OTFS: Orthogonal Time-Frequency Space Modulation

## 12.1 Motivation: The High-Mobility Problem

**OFDM** dominates 4G/5G because it handles frequency-selective channels efficiently. But at **high velocity** (fast trains, aircraft, LEO satellites), the channel becomes **doubly dispersive** — varying in both delay AND Doppler frequency.

```
                        Delay spread τ_max (channel memory)
                              ↕
OFDM symbol duration T_sym must be << coherence time T_c = 1/(4·f_d)

High velocity examples:
  Train 300 km/h at 5 GHz:  f_d = 1389 Hz → T_c = 0.18 ms
  OFDM subcarrier spacing Δf = 15 kHz → T_sym = 67 μs  (LTE)
  Ratio T_sym / T_c ≈ 0.37  ← Doppler causes significant ICI!

ICI (Inter-Carrier Interference): channel changes within one OFDM symbol
→ orthogonality between subcarriers is destroyed
→ severe performance degradation at high speed
```

---

## 12.2 The Delay-Doppler Domain

OTFS maps symbols to the **delay-Doppler (DD) grid** — a 2D domain that is:

1. **Sparse**: a channel with P paths has only P non-zero elements
2. **Stable**: path delay τ and Doppler shift ν are constant over many OFDM frames
3. **Physically meaningful**: each tap corresponds to a real scatterer with specific range and velocity

```
DD Grid X_DD[l, k]:
  l = delay bin  (0, 1, ..., M−1)   resolution: Δτ = 1/(M·Δf)
  k = Doppler bin (0, 1, ..., N−1)  resolution: Δν = Δf/N

One symbol X_DD[l,k] represents energy at:
  delay   τ = l·Δτ
  Doppler ν = k·Δν

For a channel with paths {(hᵢ, τᵢ, νᵢ)}:
  Y_DD[l,k] = Σᵢ hᵢ · X_DD[(l − lᵢ) mod M, (k − kᵢ) mod N]

The channel appears as a 2D circular convolution — easy to equalize!
```

---

## 12.3 OTFS vs OFDM: Domain Comparison

```
OFDM View:
  ┌─────────────────────────────────────────────────────┐
  │  Time-Frequency Grid                                 │
  │  Subcarrier f₁: H₁(t) [time-varying per symbol]    │
  │  Subcarrier f₂: H₂(t) [different, also varying]    │
  │  ...        Hard to estimate, pilot overhead large   │
  └─────────────────────────────────────────────────────┘

OTFS View:
  ┌─────────────────────────────────────────────────────┐
  │  Delay-Doppler Grid                                  │
  │  h(τ₁, ν₁): one tap = one scatterer                │
  │  h(τ₂, ν₂): sparse — only P non-zero values        │
  │  ...        Stable over entire frame, easy to       │
  │             estimate with few pilots                 │
  └─────────────────────────────────────────────────────┘
```

---

## 12.4 OTFS Modulation Mathematics

**Transmit signal generation:**

```
Step 1: Place QAM symbols on DD grid
  X_DD[l, k]  for l=0..M−1, k=0..N−1

Step 2: ISFFT (Inverse Symplectic Finite Fourier Transform)
         1   N−1
  X_TF[n,m] = ─ Σ  X_DD[l, k] · exp(j2πnl/M − j2πkm/N)
               N l,k
  (this maps DD → Time-Frequency grid)

Step 3: Heisenberg Transform (TF → time)
  s(t) = Σₙ Σₘ X_TF[n,m] · g_tx(t − nT) · exp(j2πmΔf·t)
  (IFFT across Doppler bins per time slot)
```

**Receive signal processing:**

```
Step 4: Wigner Transform (time → TF)
  Y_TF[n,m] = ∫ r(t) · g_rx*(t−nT) · exp(−j2πmΔf·t) dt
  (FFT across Doppler bins per time slot)

Step 5: SFFT (Symplectic Finite Fourier Transform)
  Y_DD[l,k] = Σₙₘ Y_TF[n,m] · exp(−j2πnl/M + j2πkm/N)
```

**Channel input-output in DD domain:**

```
Y_DD[l, k] = Σᵢ hᵢ · X_DD[(l − lᵢ) mod M, (k − kᵢ) mod N] + W_DD[l,k]

This is a 2D circular convolution — known structure → simple equalization
```

---

## 12.5 Channel Estimation in OTFS

Since the DD domain channel is sparse, a **single pilot** in DD domain can estimate all paths:

```
DD grid with pilot (P) and guard region (G):
  G G G G G G G G G G
  G G G G G G G G G G
  G G G G P G G G G G  ← pilot at (l_p, k_p)
  G G G G G G G G G G
  G G G G G G G G G G
  D D D D D D D D D D  ← data symbols (safe from pilot leakage)
  D D D D D D D D D D

Guard region: l_guard = l_max, k_guard = k_max
(isolates pilot from data, allows detecting delayed/Doppler-shifted echo)

Estimation: look for peaks in received Y_DD around pilot location
  h_est(lᵢ, kᵢ) = Y_DD[l_p + lᵢ, k_p + kᵢ] / X_DD[l_p, k_p]
```

---

## 12.6 OTFS Equalization

The DD domain input-output relationship is a 2D convolution:

```
Approximate MMSE in TF domain:
  X̂_TF[n,m] = H_TF*(n,m) · Y_TF[n,m] / (|H_TF(n,m)|² + σ²_n)

Then SFFT → X̂_DD

Full DD-domain MMSE (message passing):
  More complex but can exploit sparsity of h(τ,ν)
  Approaches MF bound for sparse channels
```

---

## 12.7 Performance Comparison: OFDM vs OTFS

| Metric | OFDM | OTFS |
|--------|------|------|
| Frequency-selective channels | Excellent | Excellent |
| High-velocity channels | Degrades (ICI) | Robust |
| Channel estimation overhead | High (many pilots) | Low (one pilot + guard) |
| Channel coherence | Per-subcarrier per-symbol | Over entire frame |
| Equalization complexity | Low (per-subcarrier scalar) | Higher (2D) |
| Applicable to | 4G, 5G NR (low mobility) | V2X, LEO, HSR |
| Standard support | LTE, NR | Proposed for 5G-Advanced |

---

## 12.8 Physical Intuition

```
Imagine a radar:
  - Target at range r → round-trip delay τ = 2r/c
  - Target moving at velocity v → Doppler shift ν = 2v·fc/c

OTFS symbols are the "pixels" in a delay-Doppler radar image.
The channel's tap structure == the scatterer map.

High-speed scenario (v2x at 120 km/h, fc=5.9 GHz):
  f_d = 2 × 120/3.6 × 5.9e9 / 3e8 = 1311 Hz
  τ_max = 1 μs (urban)

  OTFS: frame of N=14, M=512 → channel looks static per frame
  OFDM: ICI visible within each 66.7 μs symbol
```

---

## 12.9 Code Usage

```python
from src.otfs.otfs_system import OTFSSystem, isfft, sfft

otfs = OTFSSystem(N=14, M=64, df=15e3, modulation='QPSK')

# Transmit
bits = np.random.randint(0, 2, 14*64*2)
tx_signal = otfs.otfs_tx(bits)

# Channel: path at (delay=3, Doppler=2), gain=0.8+0.3j
channel_taps = [(0, 0, 1.0+0j), (3, 2, 0.8+0.3j), (7, -1, 0.3-0.1j)]
rx_signal = otfs.apply_channel(tx_signal, channel_taps)

# Add noise
rx_signal += np.random.randn(*rx_signal.shape) * 0.1

# Receive
Y_DD = otfs.otfs_rx(rx_signal)

# Equalize
H_DD = otfs.estimate_channel(Y_DD)   # from pilot region
X_hat = otfs.mmse_eq_tf(Y_DD, H_DD, snr_db=20)

# Demodulate
bits_rx = otfs.demodulate(X_hat)
```
