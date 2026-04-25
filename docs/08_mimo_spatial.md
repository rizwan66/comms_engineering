# MIMO Systems: Spatial Multiplexing, Diversity & Capacity

## 8.1 What is MIMO?

**MIMO** (Multiple-Input Multiple-Output) uses Nt transmit and Nr receive antennas simultaneously. Unlike beamforming (which focuses power), MIMO can transmit **independent data streams** (spatial multiplexing) or create **diversity** (more reliable link).

```
Nt=2 TX, Nr=2 RX example:

  Tx1 ─── h₁₁ ──── Rx1
   │  ╲         ╱  │
   │   ╲       ╱   │
   │    h₂₁  h₁₂   │
   │     ╲   ╱     │
   │      ╲ ╱      │
  Tx2 ─── h₂₂ ──── Rx2

H = [h₁₁  h₁₂]    (Nr×Nt channel matrix)
    [h₂₁  h₂₂]

y = H·s + n
```

---

## 8.2 Channel Matrix Properties

For i.i.d. Rayleigh fading (rich scattering, no LoS):

```
H[i,j] ~ CN(0, 1)    complex Gaussian, independent entries

|H[i,j]|²  ~ Exp(1)   (Rayleigh envelope)
```

**Singular Value Decomposition (SVD):**

```
H = U · Σ · V^H

U: Nr×Nr unitary (receive directions)
Σ: Nr×Nt diagonal (singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_min)
V: Nt×Nt unitary (transmit directions)

min(Nr, Nt) independent parallel channels ("eigenmodes")
each with gain σᵢ²
```

---

## 8.3 Zero-Forcing (ZF) Equalizer

ZF inverts the channel matrix to separate streams:

```
W_ZF = (H^H H)^{-1} H^H    (Moore-Penrose pseudoinverse)

ŝ = W_ZF · y = W_ZF · (H·s + n) = s + W_ZF·n

Noise after equalization: ñ = W_ZF·n

Noise amplification: diagonal of (H^H H)^{-1}  ← large if H is ill-conditioned
```

**Problem:** If H has small singular values (poorly conditioned), (H^H H)^{-1} blows up — ZF **amplifies noise** in weak eigenmodes.

---

## 8.4 MMSE Equalizer

MMSE adds a regularization term (σ²·I) that prevents noise amplification:

```
W_MMSE = (H^H H + σ²_n · I)^{-1} H^H

Derivation: minimizes E[||ŝ − s||²]

Bias-variance tradeoff:
  σ²_n → 0:  W_MMSE → W_ZF    (high SNR, ZF optimal)
  σ²_n → ∞:  W_MMSE → 0      (very low SNR, don't trust receive)
```

**SNR per stream after MMSE equalization:**

```
SINR_k = 1 / [(H^H H + σ²I)^{-1}]_{kk}  −  1
```

---

## 8.5 Alamouti Space-Time Block Code

Achieves **2nd-order diversity** (2 transmit, 1 receive) at **full rate** (1 symbol per channel use).

```
Encoding matrix:
       Time t         Time t+T
Ant 1: s₁             −s₂*
Ant 2: s₂              s₁*

Received signals:
r₁ = h₁·s₁ + h₂·s₂ + n₁        (time t)
r₂ = −h₁·s₂* + h₂·s₁* + n₂    (time t+T)

Combining:
ŝ₁ = h₁*·r₁ + h₂·r₂*  =  (|h₁|² + |h₂|²)·s₁ + noise term
ŝ₂ = h₂*·r₁ − h₁·r₂*  =  (|h₁|² + |h₂|²)·s₂ + noise term

Effective SNR: ρ_eff = (|h₁|² + |h₂|²)·SNR   (MRC-like diversity gain)
```

---

## 8.6 MIMO Capacity

**Shannon capacity of MIMO channel:**

```
C = log₂ det( I_Nr + (ρ/Nt) · H·H^H )   [bits/s/Hz]

where ρ = total transmit SNR

Using SVD decomposition:
C = Σᵢ log₂(1 + (ρ/Nt)·σᵢ²)     (sum of parallel SISO capacities)
```

**Water-filling power allocation** (optimal):

```
Pᵢ* = (μ − σ²_n/σᵢ²)⁺       where μ is chosen so Σ Pᵢ = P_total

More power to strong eigenmodes (large σᵢ)
Zero power to weak eigenmodes below "water level" μ
```

**Capacity scaling:**

```
SISO:   C = log₂(1 + SNR)                  (grows slowly with SNR)
MIMO:   C ≈ min(Nr,Nt) · log₂(1 + SNR)    (scales linearly with min(Nr,Nt))

At 20 dB SNR:
  1×1:  C ≈ 6.6 bps/Hz
  2×2:  C ≈ 13.3 bps/Hz
  4×4:  C ≈ 26.6 bps/Hz
```

---

## 8.7 Diversity vs Multiplexing Tradeoff

```
Diversity gain d vs Multiplexing gain r (Zheng-Tse tradeoff):

d*(r) = (Nr − r)(Nt − r)   for 0 ≤ r ≤ min(Nr, Nt)

        ▲ diversity gain d
   Nr·Nt│●
        │ ╲
        │  ╲   (Nr−1)(Nt−1) ●
        │   ╲
        │    ●──────────────► multiplexing gain r
        0    1       min(Nr,Nt)

●  Pure diversity (r=0): maximum d = Nr·Nt
●  Pure multiplexing (d=1): maximum r = min(Nr,Nt)−1
```

---

## 8.8 Code: MIMO Usage

```python
from src.mimo.mimo_system import (
    rayleigh_channel, spatial_multiplex_tx, awgn_noise,
    zf_equaliser, mmse_equaliser, apply_equaliser,
    alamouti_encode, alamouti_decode, mimo_capacity
)

Nt, Nr = 2, 2
n_bits = 1000
snr_db = 15

# Generate channel matrix
H = rayleigh_channel(Nr, Nt)

# Spatial multiplexing TX
bits = np.random.randint(0, 2, n_bits)
tx_streams = spatial_multiplex_tx(bits, Nt)  # shape: (Nt, n_symbols)

# Add noise
noise = awgn_noise(Nr, tx_streams.shape[1], snr_db)
rx = H @ tx_streams + noise

# ZF equalization
W_zf = zf_equaliser(H)
rx_zf = apply_equaliser(W_zf, rx)

# MMSE equalization (better at low SNR)
sigma2 = 10**(-snr_db/10)
W_mmse = mmse_equaliser(H, sigma2)
rx_mmse = apply_equaliser(W_mmse, rx)

# Capacity
C = mimo_capacity(H, snr_db)
print(f"MIMO capacity: {C:.2f} bps/Hz")
```

---

## 8.9 Key Performance Metrics

| Metric | Formula | Typical value (2×2, 20dB SNR) |
|--------|---------|-------------------------------|
| Capacity | log₂ det(I + ρ/Nt · HH^H) | ~13 bps/Hz |
| ZF SNR per stream | 1/[(H^H H)^{-1}]_{kk} | ~17 dB |
| MMSE SNR per stream | 1/[(H^H H + σ²I)^{-1}]_{kk} − 1 | ~19 dB |
| Alamouti BER (Rayleigh) | ≈ (1/(4SNR))² · C(4,2) | much better than SISO |
| Condition number | σ_max / σ_min of H | 1 (ideal) to ∞ (rank deficient) |
