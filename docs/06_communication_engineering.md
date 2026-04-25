# 06 — Communication Engineering

## 6.1 Shannon's Fundamental Theorem

The **Shannon-Hartley theorem** gives the theoretical maximum data rate (channel capacity):

```
C = B · log₂(1 + SNR)   [bits/second]
```
- C = channel capacity (bps)
- B = bandwidth (Hz)
- SNR = signal-to-noise ratio (linear)

**Implications:**
- You can never transmit error-free faster than C
- To double capacity: either double B or square the SNR
- Below capacity, arbitrarily low BER is achievable (with coding)

**Spectral efficiency**: η = C/B = log₂(1 + SNR) [bits/s/Hz]

---

## 6.2 Channel Models

### AWGN Channel
```
r(t) = s(t) + n(t),    n(t) ~ N(0, N₀/2)
```
- Simplest model, baseline for BER analysis
- N₀/2 = noise power spectral density (two-sided)

### Rayleigh Fading
- Envelope of received signal follows Rayleigh distribution
- Models urban/indoor environments with many scatterers
- No line-of-sight (NLOS)
- PDF: p(r) = (r/σ²)·e^(−r²/2σ²)

### Rician Fading
- Strong LOS component + scattered paths
- K-factor = LOS power / scattered power
- K=0 → Rayleigh, K→∞ → AWGN

### Path Loss Models
```
Urban macro:  PL = 128.1 + 37.6·log₁₀(d)  [dB, d in km]
Free space:   PL = 20·log₁₀(4πd/λ)
```

---

## 6.3 Noise in Communication Systems

### Thermal Noise
```
P_noise = kTB
```
- k = 1.38 × 10⁻²³ J/K (Boltzmann constant)
- T = temperature in Kelvin (room temp: 290 K)
- B = bandwidth (Hz)
- At 290 K, 1 Hz BW: P_noise = −174 dBm/Hz

### Eb/N₀ (Energy per bit to noise PSD ratio)
```
Eb/N₀ = (C/B) · (B/R) · SNR = SNR · (B/R)
```
- C = signal power, R = bit rate, B = bandwidth
- Key figure of merit for comparing modulation schemes

---

## 6.4 Multiple Access Techniques

| Technique | How it works | Used in |
|-----------|-------------|---------|
| **FDMA** | Different freq bands per user | Analog cellular, ADSL |
| **TDMA** | Time slots per user | GSM, GPS |
| **CDMA** | Different PN codes per user | CDMA2000, 3G WCDMA |
| **OFDMA** | Subcarrier groups per user | LTE, 5G NR, WiFi 6 |
| **SDMA** | Spatial beams per user (MIMO) | 5G Massive MIMO |

---

## 6.5 MIMO (Multiple Input Multiple Output)

Multiple antennas at TX and RX:

### Capacity
```
C = B · log₂ det(I + (ρ/Nₜ)·H·Hᴴ)
```
- H = Nᵣ × Nₜ channel matrix
- ρ = average SNR
- Nₜ, Nᵣ = number of TX, RX antennas

### Spatial Multiplexing
- Transmit independent streams on each antenna
- Capacity scales linearly with min(Nₜ, Nᵣ)
- Requires rich scattering environment

### Diversity (Alamouti code)
- Same data on multiple antennas with coding
- Improves reliability (diversity gain = Nₜ × Nᵣ)

---

## 6.6 Error Correction (FEC)

### Hamming Code
- (n, k) code: n-bit codeword, k info bits
- Can correct 1 error, detect 2
- Rate = k/n

### Convolutional Codes
- Encoder has memory (shift register)
- Viterbi algorithm for decoding (MLSE)
- Used in: WiFi (802.11a/b/g), satellite

### Turbo Codes
- Parallel concatenated convolutional codes
- Iterative (turbo) decoding
- Near Shannon capacity performance
- Used in: 3G/4G LTE

### LDPC (Low Density Parity Check)
- Sparse parity check matrix
- Belief propagation decoding
- Used in: 5G NR, DVB-S2, WiFi 802.11n+

### Polar Codes
- First capacity-achieving codes with explicit construction
- Used in: 5G NR control channels

---

## 6.7 Wireless Standards Summary

| Standard | Generation | Modulation | Peak Rate | Key Feature |
|----------|-----------|-----------|-----------|-------------|
| GSM | 2G | GMSK | 9.6 kbps | Voice, TDMA |
| CDMA2000 | 3G | QPSK | 3.1 Mbps | CDMA |
| LTE | 4G | OFDMA/64-QAM | 100 Mbps | OFDM, MIMO |
| LTE-A | 4.5G | OFDMA/256-QAM | 1 Gbps | Carrier aggregation |
| 5G NR | 5G | OFDMA/256-QAM | 20 Gbps | mmWave, Massive MIMO |
| WiFi 4 | 802.11n | OFDM/64-QAM | 600 Mbps | MIMO |
| WiFi 6 | 802.11ax | OFDMA/1024-QAM | 9.6 Gbps | OFDMA, MU-MIMO |

---

## 6.8 Nyquist & Shannon — Key Limits

| Theorem | Statement | Formula |
|---------|-----------|---------|
| Nyquist Sampling | Sample at 2× max freq | fₛ ≥ 2W |
| Nyquist Bandwidth | Max symbol rate in B Hz | Rs ≤ 2B symbols/s |
| Shannon Capacity | Max error-free rate | C = B·log₂(1+SNR) |

---

## 6.9 Signal Flow in a Real 5G System

```
User Data
   │
[App Layer] → IP packets
   │
[PDCP] → Header compression, encryption
   │
[RLC]  → Segmentation, ARQ
   │
[MAC]  → Scheduling, HARQ
   │
[PHY]  → Polar/LDPC coding → OFDM modulation → MIMO precoding
   │
[RF]   → DAC → PA → Antenna Array (Massive MIMO)
   │
[Air Interface] → Propagation channel (fading, multipath, Doppler)
   │
[RX RF] → LNA → ADC
   │
[PHY RX] → OFDM demod → Equalization → LDPC/Polar decoding
   │
[MAC/RLC/PDCP] → Deliver to app
```
