# Wavelets & Kalman Filtering

---

# Part A: Wavelet Analysis

## 11.1 Wavelet vs Fourier

| Property | Fourier Transform | Wavelet Transform |
|----------|-------------------|-------------------|
| Basis functions | Infinite sinusoids | Short, oscillating "wavelets" |
| Time localization | None | Yes — knows when features occur |
| Frequency localization | Perfect | Uncertainty principle applies |
| Resolution | Fixed | Multi-resolution (coarse + fine) |
| Best for | Stationary signals | Transients, edges, EEG, seismic |

```
Fourier:  x(t) → decomposes by frequency (no time info)
          |X(f)|² — what frequencies exist? (but not when)

Wavelet:  x(t) → decomposes by scale AND position
          |W(a,b)|² — what frequency range, at what time?
```

---

## 11.2 Continuous Wavelet Transform (CWT)

```
          1    ∞          t − b
W(a,b) = ─── ∫  x(t) · ψ*(─────) dt
         √|a| -∞           a

a = scale (large a → low frequency, stretched wavelet)
b = translation (position in time)
ψ(t) = mother wavelet (must satisfy admissibility condition)
```

### Morlet Wavelet

Complex wavelet — provides both amplitude and phase information:

```
ψ(t) = exp(j2πf₀t) · exp(−t²/2σ²)

      = (oscillation) × (Gaussian window)

f₀ = center frequency of the wavelet (typically 1 Hz at unit scale)
σ  = controls time-frequency resolution tradeoff

Scalogram: |W(a,b)|² shown as 2D image → time-frequency energy map
```

### Mexican Hat Wavelet

Second derivative of Gaussian — real-valued, good for edge detection:

```
ψ(t) = (1 − t²) · exp(−t²/2)

Zero mean: ∫ψ(t)dt = 0  (wavelet admissibility)
Used for: ridge detection, image edge analysis
```

---

## 11.3 Discrete Wavelet Transform (DWT)

DWT uses dyadic (power-of-2) scales: a = 2^j, b = 2^j · k

This creates an orthogonal, critically sampled decomposition via **filter banks**.

```
Mallat filter bank (single-level DWT):

x[n] ──┬──[h[n] HPF]──[↓2]──► detail coefficients d[n]     (high freq)
        │
        └──[g[n] LPF]──[↓2]──► approximation coefficients a[n] (low freq)

h[n] = high-pass filter (wavelet filter)
g[n] = low-pass filter  (scaling filter)
[↓2] = downsample by 2  (keep every other sample)

Perfect reconstruction (PR) condition:
G(z)·G̃(z) + H(z)·H̃(z) = 2  (analysis + synthesis filters matched)
```

**Multi-level DWT (3 levels):**

```
x[n] ──[DWT-1]──► d₁[n]  (level 1 details — highest freq band)
                └──[DWT-1]──► d₂[n]  (level 2 details)
                            └──[DWT-1]──► d₃[n]  (level 3 details)
                                        └──► a₃[n]  (approximation)

Frequency bands:
  a₃: [0,  fs/16]    coarsest
  d₃: [fs/16, fs/8]
  d₂: [fs/8, fs/4]
  d₁: [fs/4, fs/2]   finest (Nyquist)
```

---

## 11.4 Wavelet Denoising

Uses the fact that signal energy concentrates in **few large coefficients** while noise spreads across all coefficients.

```
Procedure:
  1. DWT:  coefficients {cⱼ[k]} = DWT(noisy signal)

  2. Threshold:
     Soft:  c̃ = sign(c) · max(|c| − λ, 0)   (shrinkage toward zero)
     Hard:  c̃ = c if |c| > λ, else 0          (zeroing)

  3. IDWT: x̂[n] = IDWT({c̃ⱼ[k]})

Threshold selection (universal threshold — Donoho-Johnstone):
  λ = σ_n · √(2 ln N)    where σ_n = noise std (estimated from finest level)
```

```
Hard vs Soft thresholding:

   c̃
   │  ╱╱╱              Hard: sharp cutoff (may introduce ringing)
───┼──────────── c      
   │    ╱╱╱

   c̃
   │     ╱              Soft: smooth shrinkage (preferred for denoising)
   │   ╱              
───┼──────────── c      
   │ ╱              
```

---

# Part B: Kalman Filter

## 11.5 The State-Space Model

The Kalman filter operates on a **linear dynamical system** described by:

```
State equation:       x[k] = F·x[k−1] + B·u[k−1] + w[k−1]
Measurement equation: z[k] = H·x[k] + v[k]

x[k] ∈ ℝⁿ   state vector          (position, velocity, phase, ...)
z[k] ∈ ℝᵐ   measurement vector    (what we observe)
F            state transition matrix
H            measurement matrix
w[k] ~ N(0, Q)  process noise
v[k] ~ N(0, R)  measurement noise
```

---

## 11.6 Kalman Filter Algorithm

```
Initialization:
  x̂[0|0] = E[x₀]     (initial state estimate)
  P[0|0]  = P₀         (initial error covariance)

═══════════ PREDICT ═══════════

  x̂[k|k−1] = F · x̂[k−1|k−1]                     (prior state)
  P[k|k−1]  = F · P[k−1|k−1] · Fᵀ + Q             (prior covariance)

═══════════ UPDATE ═════════════

  ỹ[k]  = z[k] − H · x̂[k|k−1]                    (innovation)
  S[k]  = H · P[k|k−1] · Hᵀ + R                   (innovation covariance)
  K[k]  = P[k|k−1] · Hᵀ · S[k]⁻¹                  (Kalman gain)
  x̂[k|k] = x̂[k|k−1] + K[k] · ỹ[k]               (posterior state)
  P[k|k]  = (I − K[k]·H) · P[k|k−1]               (posterior covariance)
```

**Interpretation of Kalman gain K:**

```
K → large  (R small, sensor precise):  trust measurement more
K → small  (R large, sensor noisy):    trust prediction more

K = P·Hᵀ / (H·P·Hᵀ + R)

If H=1, P=prediction uncertainty, R=measurement noise:
  K = P/(P+R)   (weighted average: P/(P+R) × measurement + R/(P+R) × prediction)
```

---

## 11.7 Constant-Velocity Tracker

Tracks a target moving with unknown acceleration.

```
State: x = [position, velocity]ᵀ

F = [1  Δt]    (position = old_pos + velocity×Δt)
    [0   1]    (velocity stays constant, noise drives changes)

H = [1  0]     (we only measure position)

Q = q·[Δt³/3  Δt²/2]   (process noise — models acceleration uncertainty)
      [Δt²/2  Δt   ]

q = acceleration spectral density (tune: large q → more responsive, more noise)
```

---

## 11.8 Extended Kalman Filter (EKF)

For **nonlinear** systems: linearize around the current estimate using Jacobians.

```
Nonlinear model:
  x[k] = f(x[k−1]) + w[k]
  z[k] = h(x[k]) + v[k]

EKF Jacobians:
  F_k = ∂f/∂x |_{x̂[k−1|k−1]}     (linearized state transition)
  H_k = ∂h/∂x |_{x̂[k|k−1]}       (linearized measurement)

EKF algorithm: same as KF but use F_k, H_k, and nonlinear propagation f(), h()
```

**Application: carrier phase tracking**

```
State:  x = [phase, freq_offset]ᵀ
f(x) = [phase + freq_offset·Ts,  freq_offset]ᵀ    (nonlinear: phase wraps)
h(x) = exp(j·phase)                                 (nonlinear measurement)

EKF tracks slowly-varying carrier phase under noise.
```

---

## 11.9 RTS Smoother (Offline Kalman)

The standard Kalman filter is a **causal estimator** (uses past only). The **RTS smoother** runs a backward pass to incorporate future measurements — optimal for offline processing.

```
Forward pass:  compute x̂[k|k], P[k|k] for k=1..N  (standard Kalman)
Backward pass: starting from k=N down to k=1:

  G[k] = P[k|k] · Fᵀ · P[k+1|k]⁻¹                  (smoother gain)
  x̂s[k] = x̂[k|k] + G[k]·(x̂s[k+1] − x̂[k+1|k])    (smoothed state)
  Ps[k] = P[k|k] + G[k]·(Ps[k+1] − P[k+1|k])·G[k]ᵀ (smoothed cov)
```

---

## 11.10 Code Usage

```python
from src.kalman.kalman_filter import KalmanFilter, EKF, build_constant_velocity_kf
from src.wavelets.wavelet_transform import dwt_multilevel, idwt_multilevel

# Kalman tracking
kf = build_constant_velocity_kf(dt=0.01, sigma_a=1.0, sigma_z=0.5)
estimates = []
for measurement in noisy_positions:
    kf.predict()
    kf.update(np.array([measurement]))
    estimates.append(kf.x[0])  # filtered position

# Wavelet denoising
from src.wavelets.wavelet_transform import db4_filters, soft_threshold

h, g = db4_filters()
coeffs = dwt_multilevel(noisy_signal, h, g, levels=4)
# threshold detail coefficients (not approximation)
threshold = 0.04 * np.sqrt(2 * np.log(len(noisy_signal)))
coeffs_thresh = [soft_threshold(c, threshold) if i > 0 else c
                 for i, c in enumerate(coeffs)]
denoised = idwt_multilevel(coeffs_thresh, h, g)
```
