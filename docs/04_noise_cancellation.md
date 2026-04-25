# 04 — Noise Cancellation

## 4.1 Types of Noise

| Noise Type | Description | Source |
|-----------|-------------|--------|
| **AWGN** | White Gaussian — flat PSD | Thermal / receiver electronics |
| **Thermal** | kTB, unavoidable | Resistors, amplifiers |
| **Shot** | Discrete electron arrivals | Photodetectors, PN junctions |
| **Flicker (1/f)** | Power ∝ 1/f | Semiconductors at low freq |
| **Impulsive** | Short high-amplitude bursts | Lightning, switching |
| **Narrowband** | Single-freq interference | 50/60 Hz power line hum |
| **Multipath/ISI** | Delayed reflections | Indoor/mobile propagation |

---

## 4.2 SNR — Signal-to-Noise Ratio

```
SNR = P_signal / P_noise

SNR_dB = 10 · log₁₀(SNR)
```

### Practical thresholds
- Voice intelligibility : SNR > 10 dB
- Good audio quality   : SNR > 40 dB
- Professional audio   : SNR > 60 dB
- 5G NR cell edge      : SNR ~ 0–5 dB

---

## 4.3 Wiener Filter (Optimal Linear Filter)

Minimises Mean Square Error for **stationary** signals:

```
H_opt(ω) = S_xd(ω) / S_xx(ω)
```

Discrete **Wiener-Hopf** equations:
```
R_xx · w = r_xd
```
- R_xx = autocorrelation matrix of input
- r_xd = cross-correlation of input and desired output

Limitation: requires stationary statistics — poor for time-varying noise.

---

## 4.4 LMS — Least Mean Squares

Real-time adaptive filter; weights update every sample:

```
y[n]   = wᵀ[n] · x[n]          (filter output)
e[n]   = d[n] − y[n]           (error)
w[n+1] = w[n] + 2μ · e[n] · x[n]  (weight update)
```

**Step size μ**:
- Too large → unstable
- Too small → slow convergence
- Stable if: `0 < μ < 1 / (N · Pₓ)`

Converges on average to the Wiener solution.

---

## 4.5 RLS — Recursive Least Squares

Faster convergence than LMS; minimises exponentially weighted LS cost:

```
K[n] = P[n−1]·x[n] / (λ + xᵀ[n]·P[n−1]·x[n])
e[n] = d[n] − wᵀ[n−1]·x[n]
w[n] = w[n−1] + K[n]·e[n]
P[n] = (1/λ)(P[n−1] − K[n]·xᵀ[n]·P[n−1])
```

- λ = forgetting factor (0.95–0.999)
- Complexity: O(N²) vs O(N) for LMS
- Better tracking of non-stationary processes

---

## 4.6 Active Noise Cancellation (ANC)

Generates **anti-phase sound** to cancel unwanted noise physically:

```
Reference mic → [Adaptive Filter] → Anti-noise speaker
                        ↑
                Error mic (residual noise)
```

### Topologies
| Type | Mics | Best for |
|------|------|---------|
| Feed-forward | Reference + Error | Predictable/tonal noise |
| Feedback | Error only | Broadband noise |
| Hybrid | Both | Best overall (ANC headphones) |

---

## 4.7 Spectral Subtraction

Simple frequency-domain approach for speech enhancement:

```
|Ŝ(ω)|² = max(|Y(ω)|² − α·|N̂(ω)|², β·|Y(ω)|²)
```
- α = over-subtraction factor (reduces residual noise, risk of musical noise)
- β = spectral floor (prevents negative power)
- N̂(ω) estimated from silence/noise-only frames

---

## 4.8 Noise Figure and Cascaded Systems

```
F_total = F₁ + (F₂−1)/G₁ + (F₃−1)/(G₁G₂) + ...
NF = 10·log₁₀(F)   [dB]
```

First stage dominates → LNA must have the lowest noise figure.

---

## 4.9 Key Equations Summary

| Method | Update Rule | Complexity | Best For |
|--------|------------|-----------|---------|
| Wiener | R·w = r (batch) | O(N³) | Stationary, offline |
| LMS | w += 2μ·e·x | O(N) | Real-time, simple |
| RLS | Kalman-like | O(N²) | Fast convergence |
| Spectral Sub | PSD subtraction | O(N log N) | Speech, offline |
| ANC (LMS) | Physical anti-noise | O(N) | Headphones, HVAC |
