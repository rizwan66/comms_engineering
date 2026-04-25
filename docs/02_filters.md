# 02 — Filters

## 2.1 What is a Filter?

A filter is an LTI system that **selectively passes or rejects frequency components** of a signal.

```
x[n] → [ Filter H(z) ] → y[n]
```

Characterized by its **frequency response** H(e^jω) = |H(e^jω)|·e^(j∠H(e^jω))

---

## 2.2 Filter Types by Frequency Response

| Filter | Passes | Rejects | |
|--------|--------|---------|--|
| **Low-Pass (LPF)** | Low frequencies | High frequencies | ω < ωc |
| **High-Pass (HPF)** | High frequencies | Low frequencies | ω > ωc |
| **Band-Pass (BPF)** | Band [ωL, ωH] | Outside band | ωL < ω < ωH |
| **Band-Stop/Notch** | Outside band | Band [ωL, ωH] | Blocks specific freq |
| **All-Pass** | All frequencies | None (phase only) | Phase correction |

---

## 2.3 FIR vs IIR Filters

### FIR (Finite Impulse Response)
```
y[n] = Σ bₖ · x[n−k]   (k = 0 to M)
```
- **Always stable** (no feedback)
- Linear phase possible → no phase distortion
- Requires more coefficients for sharp cutoffs
- Transfer function: H(z) = Σ bₖ z^(−k)

### IIR (Infinite Impulse Response)
```
y[n] = Σ bₖx[n−k] − Σ aₖy[n−k]
```
- Uses feedback (recursive)
- **Fewer coefficients** for same sharpness
- Can be unstable if poles outside unit circle
- Non-linear phase (usually)
- Transfer function: H(z) = B(z)/A(z)

---

## 2.4 Classic IIR Filter Designs

### Butterworth Filter
- **Maximally flat** magnitude response in passband (no ripple)
- -3dB at cutoff ωc, rolls off at -20N dB/decade
- Order N determines roll-off steepness

```
|H(jω)|² = 1 / (1 + (ω/ωc)^(2N))
```

### Chebyshev Type I
- **Equiripple in passband**, monotone in stopband
- Sharper roll-off than Butterworth for same order
- Ripple factor ε controls passband variation

### Chebyshev Type II
- Monotone passband, **equiripple in stopband**
- Flat passband like Butterworth, better stopband attenuation

### Elliptic (Cauer) Filter
- **Equiripple in BOTH passband and stopband**
- Sharpest possible transition for given order
- Most efficient, but complex design

### Bessel Filter
- **Maximally flat group delay** (linear phase)
- Used when pulse shape preservation matters more than sharpness

---

## 2.5 Filter Design Specifications

```
Passband:  |H(jω)| ≥ 1−δp   for  |ω| ≤ ωp
Stopband:  |H(jω)| ≤ δs     for  |ω| ≥ ωs
```

- ωp = passband edge frequency
- ωs = stopband edge frequency  
- δp = passband ripple
- δs = stopband attenuation
- Transition band: ωp to ωs

---

## 2.6 FIR Filter Design Methods

### Window Method
1. Compute ideal impulse response hd[n]
2. Multiply by window w[n]: h[n] = hd[n] · w[n]

| Window | Sidelobe (dB) | Transition Width | Stopband (dB) |
|--------|---------------|-----------------|---------------|
| Rectangular | -13 | 4π/N | -21 |
| Hann | -31 | 8π/N | -44 |
| Hamming | -41 | 8π/N | -53 |
| Blackman | -57 | 12π/N | -74 |
| Kaiser | Adjustable | Adjustable | Adjustable |

### Parks-McClellan (Equiripple)
- Uses Remez exchange algorithm
- Optimal equiripple design — minimizes maximum error

---

## 2.7 Adaptive Filters

Adaptive filters **adjust their own coefficients** to minimize an error signal, making them ideal for time-varying environments.

```
d[n] → [+] → e[n] → [Adaptation Algorithm]
         ↑                    ↓
x[n] → [W(z)] → y[n]     update W
```

### LMS (Least Mean Squares)
```
w[n+1] = w[n] + 2μ·e[n]·x[n]
```
- μ = step size (controls convergence speed vs stability)
- Simple, low compute, robust

### RLS (Recursive Least Squares)
```
w[n] = w[n−1] + K[n]·e[n]
```
- Faster convergence than LMS
- Higher complexity O(N²)

---

## 2.8 Notch Filter

Removes a **specific frequency** (e.g., 50/60 Hz power line hum):
```
H(z) = (1 − 2cos(ω₀)z⁻¹ + z⁻²) / (1 − 2r·cos(ω₀)z⁻¹ + r²z⁻²)
```
- r close to 1 → very narrow notch
- Place zeros exactly on unit circle at ω = ω₀

---

## 2.9 Digital Filter Implementation

### Direct Form I
```
y[n] = b₀x[n] + b₁x[n-1] + ... − a₁y[n-1] − a₂y[n-2] − ...
```

### Direct Form II (Transposed)
- Minimizes memory (uses shared delay line)
- Preferred for fixed-point implementations

### Cascade (Second-Order Sections — SOS)
- Factor H(z) into 2nd order sections
- **Numerically more stable** for high-order filters
```
H(z) = H₁(z) · H₂(z) · ... · HK(z)
```

---

## 2.10 Key Filter Equations Summary

| Design | Transfer Function | Key Feature |
|--------|-----------------|-------------|
| Butterworth | 1/(1+(s/ωc)^2N) | Maximally flat |
| Chebyshev I | 1/(1+ε²Tₙ²(ω/ωc)) | Equiripple passband |
| Elliptic | Complex rational | Equiripple both |
| FIR (window) | B(z) (no poles) | Linear phase |
| Notch | Zeros on unit circle | Remove single freq |
