# 05 — Transceivers (TX/RX Systems)

## 5.1 Transceiver Architecture Overview

A **transceiver** is a device that both transmits and receives signals. Modern digital communication transceivers follow this general chain:

```
TRANSMITTER
─────────────────────────────────────────────────────────────────
Bits → [Symbol Map] → [Pulse Shape] → [Up-Convert] → [PA] → Antenna
                                                        ↑
                                              RF Front-End

RECEIVER
─────────────────────────────────────────────────────────────────
Antenna → [LNA] → [Down-Convert] → [Matched Filter] → [ADC] → [Equalizer] → Bits
            ↑
    RF Front-End
```

---

## 5.2 Transmitter Chain (TX)

### 1. Source Coding
- Compress the source (speech → codec, video → H.264/H.265)
- Reduces redundancy before transmission

### 2. Channel Coding (FEC)
- Add controlled redundancy to correct bit errors
- Examples: Turbo codes, LDPC, Polar codes (5G NR)
- Rate = k/n (k info bits, n coded bits)

### 3. Symbol Mapping / Modulation
- Map bits to constellation points (BPSK, QPSK, QAM)
- Higher order = more bits/symbol but needs better SNR

### 4. Pulse Shaping
- Shape bits into band-limited pulses to avoid ISI
- **Root Raised Cosine (RRC)** filter at TX
- Nyquist criterion: zero ISI at sampling instants

### 5. Up-Conversion (RF)
```
s(t) = I(t)·cos(2πfct) − Q(t)·sin(2πfct)
```
- Mixes baseband signal to carrier frequency fc
- IQ modulator architecture

### 6. Power Amplifier (PA)
- Amplifies to transmission power
- Non-linearity causes spectral regrowth (PAPR problem with OFDM)

---

## 5.3 Receiver Chain (RX)

### 1. Low Noise Amplifier (LNA)
- First stage — amplifies tiny received signal
- Dominates noise figure of entire receiver
- Noise Figure (NF): NF = 10·log₁₀(F), F = SNRin/SNRout

### 2. Down-Conversion
```
I(t) = r(t)·cos(2πfct)  → LPF → I baseband
Q(t) = r(t)·sin(2πfct)  → LPF → Q baseband
```
- Superheterodyne: IF stage then baseband
- Direct conversion (zero-IF): straight to baseband

### 3. ADC (Analog-to-Digital Converter)
- Sample at ≥ 2× signal bandwidth (Nyquist)
- Quantization noise: SNR_quant ≈ 6.02·N + 1.76 dB (N bits)
- Typical: 10–16 bit ADC in modern receivers

### 4. Matched Filter
- RRC filter at RX → combined with TX RRC = full RC
- Maximizes SNR at sampling instant
- Eliminates ISI (Nyquist zero-ISI condition)

### 5. Synchronization
- **Timing recovery**: find optimal sampling instant
- **Carrier recovery**: synchronize local oscillator phase
- **Frame sync**: detect packet/frame boundaries

### 6. Equalization
- Compensates for multipath distortion (ISI)
- **MMSE equalizer**: minimize mean square error
- **Zero-forcing (ZF)**: invert channel response
- **DFE** (Decision Feedback): use past decisions

### 7. Channel Decoding
- FEC decoder corrects errors introduced by channel
- Viterbi, BCJR, belief propagation algorithms

---

## 5.4 RF Front-End

### Key Parameters
| Parameter | Definition |
|-----------|-----------|
| Noise Figure (NF) | Degradation in SNR through device |
| Gain (G) | Signal amplification in dB |
| IIP3 | 3rd order intercept (linearity measure) |
| P1dB | 1dB compression point |
| VSWR | Voltage Standing Wave Ratio (matching) |

### Friis Formula (Cascaded Noise Figure)
```
F_total = F₁ + (F₂−1)/G₁ + (F₃−1)/(G₁G₂) + ...
```
First stage dominates → LNA must have lowest NF.

---

## 5.5 ADC/DAC

### ADC — Analog to Digital
```
Quantization step: Δ = Vref / 2^N
Quantization noise power: σ²_q = Δ²/12
SNR_max = 6.02N + 1.76 dB
```

### Key ADC Specs
- **ENOB** (Effective Number of Bits): accounts for noise floor
- **SFDR** (Spurious-Free Dynamic Range)
- **SNDR** (Signal-to-Noise-and-Distortion Ratio)
- Sample rate vs resolution trade-off

---

## 5.6 Pulse Shaping Details

### ISI (Inter-Symbol Interference)
When symbol rate > channel bandwidth, symbols overlap → ISI.

**Nyquist ISI criterion**: h(nT) = δ[n] for zero ISI.

### Raised Cosine Spectrum
```
H(f) = T,                           |f| ≤ (1−β)/(2T)
H(f) = T/2·{1+cos[πT/β(|f|−(1−β)/(2T))]},  (1−β)/(2T) < |f| ≤ (1+β)/(2T)
H(f) = 0,                           |f| > (1+β)/(2T)
```
- β = 0: sinc pulse, minimum bandwidth, sensitive to timing
- β = 1: maximum bandwidth, robust to timing errors

---

## 5.7 Eye Diagram

Visual tool to assess signal quality:
- **Eye opening** = noise margin
- **Jitter** = horizontal eye closure (timing uncertainty)
- **ISI** = vertical eye closure

A wide, tall, clear eye = good system performance.

---

## 5.8 Link Budget

End-to-end power analysis:
```
Received Power (dBm) = EIRP − Path Loss + Rx Gain − Rx Losses
```

### Free-Space Path Loss (Friis)
```
FSPL (dB) = 20·log₁₀(4πd/λ) = 20·log₁₀(d) + 20·log₁₀(f) − 147.55
```

### Link Margin
```
Link Margin = Received Power − Receiver Sensitivity
```
Must be > 0 dB (with fade margin of 10–20 dB for reliability).
