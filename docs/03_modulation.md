# 03 — Modulation

## 3.1 Why Modulate?

Modulation moves information from a **baseband signal** onto a **carrier wave** for efficient transmission.

**Reasons:**
- Antennas need to be ~λ/4 long — at 1 kHz that's 75 km! At 1 GHz: 7.5 cm ✓
- Multiplexing: many signals share the same channel at different frequencies
- Noise immunity: some modulation types resist interference better

---

## 3.2 Analog Modulation

### AM — Amplitude Modulation
```
s(t) = Ac[1 + ka·m(t)]·cos(2πfct)
```
- m(t) = message, Ac = carrier amplitude, ka = modulation index
- **Bandwidth**: BAM = 2W  (W = message bandwidth)
- Spectral efficiency: poor (carrier + 2 sidebands, carrier carries no info)

**DSB-SC (Double Sideband Suppressed Carrier)**:
```
s(t) = Ac·m(t)·cos(2πfct)
```
- No carrier → more power efficient
- Requires coherent demodulation

**SSB (Single Sideband)**:
```
s(t) = (Ac/2)·m(t)·cos(2πfct) ∓ (Ac/2)·m̂(t)·sin(2πfct)
```
- Only one sideband transmitted → **half the bandwidth of AM**
- Best spectral efficiency for analog

---

### FM — Frequency Modulation
```
s(t) = Ac·cos[2πfct + 2πkf∫m(τ)dτ]
```
- kf = frequency sensitivity (Hz/V)
- **Modulation index**: β = Δf/W = kf·Aₘ/W
- **Carson's rule** bandwidth: BFM ≈ 2(Δf + W) = 2W(β + 1)

**Advantages**:
- Much better noise immunity than AM (FM captures effect)
- Constant amplitude → not affected by amplitude variations

**Wideband FM**: β >> 1 → large bandwidth but excellent SNR  
**Narrowband FM**: β << 1 → similar bandwidth to AM

---

### PM — Phase Modulation
```
s(t) = Ac·cos[2πfct + kp·m(t)]
```
- kp = phase sensitivity (rad/V)
- Closely related to FM (integral/derivative relationship)
- Used in digital phase-based schemes

---

## 3.3 Digital Modulation

### ASK — Amplitude Shift Keying
Binary ASK (OOK):
```
s(t) = { Ac·cos(2πfct)  for bit 1
        { 0              for bit 0
```

### FSK — Frequency Shift Keying
```
s(t) = { Ac·cos(2πf₁t)  for bit 1
        { Ac·cos(2πf₂t)  for bit 0
```
- Minimum Shift Keying (MSK): special FSK where Δf = 1/(2Tb)

### PSK — Phase Shift Keying

**BPSK (Binary PSK)**:
```
s(t) = Ac·cos(2πfct + πb[n])   b[n] ∈ {0,1}
```
- BER = Q(√(2Eb/N₀)) — excellent performance

**QPSK (Quadrature PSK)**:
- 4 phases: 45°, 135°, 225°, 315°
- 2 bits per symbol
- Same bandwidth as BPSK, double data rate

**8-PSK**: 3 bits/symbol, 8 phases

---

### QAM — Quadrature Amplitude Modulation
Combines amplitude AND phase modulation:
```
s(t) = I(t)·cos(2πfct) − Q(t)·sin(2πfct)
```
- I = in-phase component, Q = quadrature component
- **16-QAM**: 4 bits/symbol (4x4 constellation)
- **64-QAM**: 6 bits/symbol
- **256-QAM**: 8 bits/symbol (used in 5G, cable)

**Spectral efficiency**: M-QAM = log₂(M) bits/s/Hz

---

### OFDM — Orthogonal Frequency Division Multiplexing

Split data across **many parallel subcarriers**:
```
s(t) = Re{ Σ Xₖ · e^(j2πfₖt) }
```
- Subcarriers are orthogonal: fₖ = f₀ + k/T
- Each subcarrier can carry QAM symbols
- **Cyclic prefix** (CP): adds guard interval to eliminate ISI

**Advantages**:
- Excellent multipath resistance
- Efficient use of spectrum
- Used in: WiFi (802.11), LTE, 5G NR, DVB-T, ADSL

**IFFT/FFT implementation**:
- TX: Use IFFT to convert freq-domain symbols → time domain
- RX: Use FFT to convert back

---

## 3.4 Spread Spectrum

### DSSS (Direct Sequence Spread Spectrum)
```
s(t) = m(t)·c(t)·cos(2πfct)
```
- c(t) = PN (pseudo-noise) spreading code at chip rate >> bit rate
- Spreading gain = chip rate / bit rate
- Resistant to narrowband interference, enables CDMA

### FHSS (Frequency Hopping Spread Spectrum)
- Carrier hops between frequencies using PN sequence
- Avoids narrowband jamming

---

## 3.5 Modulation Comparison Table

| Scheme | Bits/Symbol | BW Efficiency | Noise Immunity | Use Case |
|--------|-------------|--------------|----------------|----------|
| BPSK | 1 | Low | Best | Deep space |
| QPSK | 2 | Medium | Good | Satellite, GPS |
| 16-QAM | 4 | High | Moderate | LTE, WiFi |
| 64-QAM | 6 | Very High | Lower | 5G, Cable |
| 256-QAM | 8 | Excellent | Lowest | 5G, DOCSIS |
| OFDM | Varies | Excellent | Excellent (multipath) | WiFi, 5G, DVB |
| FM | - | Wide | Excellent | Broadcasting |

---

## 3.6 Demodulation

### Coherent Detection
- Requires phase-synchronized local oscillator
- Better performance
- Used for: PSK, QAM, DSB-SC

### Non-Coherent Detection
- No phase synchronization needed
- Slightly worse performance
- Used for: FSK (envelope detector), DPSK

### Matched Filter
The **optimal receiver** for an AWGN channel:
```
h(t) = s(T−t)   [matched to transmitted pulse s(t)]
```
Maximizes SNR at the sampling instant.

---

## 3.7 BER Performance in AWGN

| Modulation | BER Formula |
|-----------|-------------|
| BPSK | Q(√(2Eb/N₀)) |
| QPSK | Q(√(2Eb/N₀)) |
| M-PSK | (2/log₂M)·Q(√(2log₂M·sin²(π/M)·Eb/N₀)) |
| M-QAM | ≈ (4/log₂M)·Q(√(3log₂M/(M−1)·Eb/N₀)) |
| FSK (non-coh) | (1/2)·e^(−Eb/(2N₀)) |
