# OFDM, Channel Estimation & Equalization

## 7.1 Why OFDM?

Traditional single-carrier systems suffer **inter-symbol interference (ISI)** on frequency-selective channels: delayed multipath copies smear adjacent symbols together. OFDM defeats ISI by splitting the wideband channel into hundreds of **flat, narrowband subcarriers** — each experiences only a single complex gain.

```
Wideband channel (frequency-selective):
  |H(f)|
    ▲
    │  ╭──╮      ╭─╮
    │ ╱    ╲    ╱   ╲
    │╱      ╲──╱     ╲──
    └─────────────────────► f
    
OFDM divides this into N flat sub-channels:
  |H[k]|  ← scalar per subcarrier k
    ▲
    │ ● ● ● ● ● ● ● ● ● ●   (each ● is one subcarrier gain)
    └─────────────────────► subcarrier index k
```

---

## 7.2 OFDM Signal Mathematics

An OFDM symbol carrying N complex data symbols {X[k]} is:

```
         N−1
x[n] = Σ   X[k] · exp(j2πkn/N),    n = 0, 1, ..., N−1
         k=0
```

This is exactly the **Inverse DFT (IDFT)**. At the receiver, the FFT recovers each subcarrier:

```
         N−1
Y[k] = Σ   y[n] · exp(−j2πkn/N) = H[k]·X[k] + W[k]
         n=0
```

where H[k] is the channel gain at subcarrier k and W[k] is noise.

---

## 7.3 Cyclic Prefix

The cyclic prefix (CP) converts linear convolution (causing ISI) into **circular convolution** (which maps to multiplication in the DFT domain).

```
OFDM symbol:  [x[N−Ncp] ... x[N−1] | x[0] x[1] ... x[N−1]]
               ◄──── CP (copy of tail) ────►◄──── data ────►

CP length Ncp must satisfy: Ncp ≥ maximum channel delay spread (in samples)

Effect on received signal:
  y[n] = h[n] ⊛ x[n]  (circular conv.)  →  Y[k] = H[k]·X[k]  (DFT domain)
```

CP overhead: `η = N / (N + Ncp)` efficiency factor (LTE: N=1024, Ncp=72 → η ≈ 93%)

---

## 7.4 Pilot-Based Channel Estimation

Channel coefficients H[k] are unknown at the receiver. **Pilot subcarriers** — known transmitted symbols — allow estimation.

```
Frequency
  ▲
  │  [P][D][D][D][D][P][D][D][D][D][P]  ← Pilot spacing = 5
  │  [D][P][D][D][D][D][P][D][D][D][D]  ← Pilots staggered per symbol
  │  ...
  └───────────────────────────────────► OFDM symbol index

P = pilot subcarrier (known)
D = data subcarrier (unknown)
```

### Least-Squares (LS) Estimation

At pilot positions k_p:

```
H_LS[k_p] = Y[k_p] / P[k_p]

Error:  H_LS[k_p] = H[k_p] + W[k_p] / P[k_p]

MSE_LS = σ²_n / |P|²    (amplifies noise — no regularization)
```

### MMSE Estimation

MMSE exploits the covariance structure of the channel:

```
H_MMSE = R_HH · (R_HH + σ²_n I)^{-1} · H_LS

where R_HH = channel covariance matrix (prior knowledge of delay spread)

MSE_MMSE < MSE_LS  (always, by ~3 dB in typical channels)
```

### Interpolation to Data Subcarriers

```
Pilots at positions: k₁, k₆, k₁₁, k₁₆, ...
                       ↑         ↑
                H_est[k₁]   H_est[k₆]
                       └─────────┘
                         linear / spline interpolation
                         to all N_data positions
```

---

## 7.5 Equalization

After FFT, the received data subcarrier is:

```
Y[k] = H[k] · X[k] + W[k]
```

**Zero-Forcing (ZF):**
```
X̂_ZF[k] = Y[k] / H̃[k]   (divide by estimated channel)

Risk: if |H̃[k]| ≈ 0 (deep fade), noise is amplified enormously
```

**MMSE Equalizer:**
```
          H̃*[k]
X̂_MMSE = ────────────── · Y[k]
          |H̃[k]|² + σ²_n

→ 1/H̃[k]  when |H̃[k]| >> σ_n  (approaches ZF)
→ 0        when |H̃[k]| << σ_n  (suppresses noise in deep fades)
```

---

## 7.6 PAPR: Peak-to-Average Power Ratio

A key weakness of OFDM: many subcarriers can add coherently, creating large peaks.

```
        ▲  voltage
        │        ╷               PAPR = max|x[n]|² / mean|x[n]|²
        │   ╷    │   ╷
        │───│────│───│──── peak power
        │   │    │   │
     ───────────────────── average power
        │
        └──────────────────────────────► n

Typical OFDM PAPR: 8–12 dB
LTE/5G mitigation: clipping + filtering, tone reservation, SLM
```

**CCDF curve** (from `src/ofdm/ofdm_system.py`):

```
P(PAPR > x₀) vs x₀ [dB]
  1.0 ┤
      │╲
  0.1 ┤ ╲
      │  ╲
 0.01 ┤   ╲
      │    ╲____
      └──────────────► x₀
```

---

## 7.7 OFDM vs Single-Carrier: When to Use Which

| Property | OFDM | Single-Carrier |
|----------|------|----------------|
| Frequency-selective channels | Excellent (per-SC equalization) | Requires long equalizer |
| High mobility (Doppler) | Vulnerable (ICI) | More robust |
| PAPR | High (~10 dB) | Low (~3 dB) |
| Spectral efficiency | High with QAM + water-fill | Moderate |
| Complexity | Moderate (FFT) | Lower |
| Standards | LTE, WiFi, DVB-T, 5G NR | EDGE, Bluetooth |

---

## 7.8 Code: OFDMSystem Usage

```python
from src.ofdm.ofdm_system import OFDMSystem

ofdm = OFDMSystem(
    N_fft=64,
    N_data=48,          # data subcarriers per symbol
    cp_len=16,          # cyclic prefix length
    modulation='QPSK',  # or '16QAM'
    pilot_spacing=6     # 1 pilot per 6 subcarriers
)

# Transmit
bits = np.random.randint(0, 2, 192)     # 48 subcarriers × 2 bits × 2 symbols
tx_signal = ofdm.transmit(bits)

# Channel + AWGN
rx_signal = tx_signal + noise

# Receive (LS channel estimation + MMSE equalization built-in)
bits_rx, H_est = ofdm.receive(rx_signal)

# Measure BER over SNR sweep
ber_list = ofdm.compute_ber(snr_range_db=np.arange(0, 20))
```
