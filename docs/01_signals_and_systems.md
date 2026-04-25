# 01 — Signals and Systems

## 1.1 What is a Signal?

A **signal** is any physical quantity that varies with time, space, or another independent variable and carries information.

- **Analog (Continuous-Time)**: x(t) — exists for all t ∈ ℝ (e.g., microphone output)
- **Digital (Discrete-Time)**: x[n] — defined only at integer n (e.g., MP3 audio sample)

---

## 1.2 Signal Classification

| Type | Description | Example |
|------|-------------|---------|
| Deterministic | Fully predictable by a formula | Sine wave |
| Random/Stochastic | Described statistically | Thermal noise |
| Periodic | x(t) = x(t + T) | AC power signal |
| Aperiodic | No repeating pattern | Speech |
| Even | x(-t) = x(t) | Cosine |
| Odd | x(-t) = -x(t) | Sine |
| Energy Signal | Finite total energy | Pulse |
| Power Signal | Finite average power | Sine wave |

---

## 1.3 Fundamental Signals

### Unit Step: u(t)
```
u(t) = { 1,  t ≥ 0
        { 0,  t < 0
```

### Unit Impulse (Dirac Delta): δ(t)
```
∫ δ(t) dt = 1,   δ(t) = 0 for t ≠ 0
```
**Sifting property**: ∫ x(t)·δ(t−t₀) dt = x(t₀)

### Complex Exponential: e^(jωt)
The most fundamental signal in DSP — basis of Fourier analysis.
```
e^(jωt) = cos(ωt) + j·sin(ωt)   [Euler's Formula]
```

---

## 1.4 Systems

A **system** transforms an input signal x into output y:
```
x(t) → [ System H ] → y(t)
```

### Key System Properties

| Property | Condition | Significance |
|----------|-----------|--------------|
| **Linearity** | H{ax₁+bx₂} = aH{x₁}+bH{x₂} | Superposition holds |
| **Time-Invariance** | x(t−t₀) → y(t−t₀) | Behavior doesn't change with time |
| **Causality** | y(t) depends only on x(τ), τ≤t | Realizable in real-time |
| **Stability (BIBO)** | Bounded input → bounded output | System won't blow up |
| **Memory** | y(t) depends on past/future x | FIR vs IIR distinction |

**LTI Systems** (Linear + Time-Invariant) are the core of classical DSP — fully described by their impulse response h(t).

---

## 1.5 Convolution

The output of any LTI system:
```
y(t) = x(t) * h(t) = ∫ x(τ)·h(t−τ) dτ

y[n] = x[n] * h[n] = Σ x[k]·h[n−k]
```
**Key insight**: Convolution in time = Multiplication in frequency domain.

---

## 1.6 Fourier Series

For periodic signal x(t) with period T:
```
x(t) = Σ cₙ · e^(jnω₀t)

cₙ = (1/T) ∫ x(t) · e^(−jnω₀t) dt
```
- ω₀ = 2π/T (fundamental frequency)
- Decomposes any periodic signal into harmonics

---

## 1.7 Fourier Transform (CTFT)

For aperiodic signals:
```
X(jω) = ∫ x(t) · e^(−jωt) dt        [Forward]
x(t)  = (1/2π) ∫ X(jω) · e^(jωt) dω  [Inverse]
```

### Key CTFT Properties
| Property | Time Domain | Frequency Domain |
|----------|-------------|-----------------|
| Linearity | ax(t)+by(t) | aX(ω)+bY(ω) |
| Time Shift | x(t−t₀) | e^(−jωt₀)·X(ω) |
| Frequency Shift | e^(jω₀t)·x(t) | X(ω−ω₀) |
| Convolution | x(t)*h(t) | X(ω)·H(ω) |
| Duality | X(t) | 2π·x(−ω) |
| Differentiation | dx/dt | jω·X(ω) |

---

## 1.8 Discrete-Time Fourier Transform (DTFT)

```
X(e^jω) = Σ x[n] · e^(−jωn)
x[n] = (1/2π) ∫ X(e^jω) · e^(jωn) dω
```

---

## 1.9 Discrete Fourier Transform (DFT) & FFT

The **DFT** is the computable, finite version:
```
X[k] = Σ x[n] · e^(−j2πkn/N),  k = 0,1,...,N−1
x[n] = (1/N) Σ X[k] · e^(j2πkn/N)
```

**FFT** (Fast Fourier Transform): Cooley-Tukey algorithm reduces DFT from O(N²) to **O(N log N)**.

---

## 1.10 Laplace Transform

Generalization of CTFT for system analysis:
```
X(s) = ∫ x(t) · e^(−st) dt,   s = σ + jω
```
Used to find **poles and zeros** of a system → stability analysis.

**Region of Convergence (ROC)**: The set of s values for which X(s) converges.

---

## 1.11 Z-Transform

Discrete counterpart of Laplace:
```
X(z) = Σ x[n] · z^(−n),   z = re^(jω)
```

**Relationship to DTFT**: Evaluate on unit circle (|z|=1): X(e^jω) = X(z)|_{z=e^jω}

### Poles & Zeros
- **Zeros** (numerator roots): X(z) = 0
- **Poles** (denominator roots): X(z) = ∞
- System stable ↔ all poles inside unit circle

---

## 1.12 Sampling Theorem

**Nyquist-Shannon Theorem**:
> A bandlimited signal with max frequency fₘ can be perfectly reconstructed if sampled at:
> **fₛ ≥ 2·fₘ**  (Nyquist rate)

### Aliasing
When fₛ < 2fₘ, high-frequency components "fold back" and corrupt lower frequencies.

**Anti-aliasing filter**: Low-pass filter applied BEFORE sampling to remove frequencies > fₛ/2.

---

## 1.13 Key Formulas Summary

| Transform | Forward | Inverse |
|-----------|---------|---------|
| Fourier (CT) | ∫x(t)e^(−jωt)dt | (1/2π)∫X(ω)e^(jωt)dω |
| Laplace | ∫x(t)e^(−st)dt | Partial fractions + tables |
| DTFT | Σx[n]e^(−jωn) | (1/2π)∫X(e^jω)e^(jωn)dω |
| Z-Transform | Σx[n]z^(−n) | Contour integral / tables |
| DFT | Σx[n]e^(−j2πkn/N) | (1/N)ΣX[k]e^(j2πkn/N) |
