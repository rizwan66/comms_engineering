# 📡 Communications Engineering & DSP

> A comprehensive, hands-on Python project covering Digital Signal Processing, Communication Engineering, Transceivers, Filters, Noise Cancellation, Modulation, OFDM, MIMO, FEC, and Synchronisation.

**GitHub:** [@rizwan66](https://github.com/rizwan66) | **Repo:** `comms_engineering`

---

## 🗂️ Project Structure

```
comms_engineering/
├── src/
│   ├── signals/generator.py          # Sine, chirp, FFT, STFT, convolution
│   ├── filters/design.py             # Butterworth, Chebyshev, Elliptic, FIR, LMS, RLS, Notch
│   ├── modulation/schemes.py         # AM, FM, BPSK, QPSK, QAM, OFDM, BER
│   ├── noise_cancellation/canceller.py  # Spectral sub, Wiener, LMS/RLS ANC
│   ├── transceivers/chain.py         # Pulse shaping, matched filter, eye diagram
│   ├── ofdm/ofdm_system.py           # Pilots, channel estimation, PAPR
│   ├── fec/channel_coding.py         # Convolutional, LDPC, Turbo, Shannon limit
│   ├── mimo/mimo_system.py           # ZF, MMSE, Alamouti, capacity
│   └── synchronisation/pll.py        # PLL, Costas loop, M&M timing recovery
├── simulations/full_chain_demo.py    # Master demo
├── notebooks/DSP_Interactive_Notebook.ipynb
├── docs/                             # 6 theory markdown files
├── assets/plots/                     # 34 generated figures
└── requirements.txt
```

---

## 🧠 Topics Covered

| Module | Key Topics |
|--------|-----------|
| **Signals & Systems** | CTFT, DTFT, DFT/FFT, Z-transform, Laplace, convolution, sampling |
| **Filters** | Butterworth, Chebyshev I/II, Elliptic, FIR (window, Parks-McClellan), LMS, RLS, Notch |
| **Modulation** | AM, DSB-SC, FM, PM, BPSK, QPSK, M-QAM, OFDM, spread spectrum, BER |
| **Noise Cancellation** | Wiener filter, LMS, RLS, ANC, spectral subtraction, SNR analysis |
| **Transceivers** | RF front-end, pulse shaping (RC/RRC), matched filter, eye diagram, link budget |
| **OFDM** | IFFT/FFT, cyclic prefix, pilot insertion, LS channel estimation, PAPR, CCDF |
| **FEC** | Convolutional, Viterbi, LDPC, Turbo, Shannon capacity, coding gain |
| **MIMO** | Spatial multiplexing, ZF/MMSE, Alamouti diversity, SVD beamforming, capacity |
| **Synchronisation** | PLL, Costas loop, frequency offset correction, Mueller & Muller timing |
| **Comm. Engineering** | Shannon-Hartley, Rayleigh/Rician fading, FDMA/TDMA/CDMA/OFDMA, 5G NR |

---

## Quick Start

```bash
git clone https://github.com/rizwan66/comms_engineering.git
cd comms_engineering
pip install -r requirements.txt

# Master demo
python simulations/full_chain_demo.py

# Individual modules
python src/signals/generator.py
python src/filters/design.py
python src/modulation/schemes.py
python src/ofdm/ofdm_system.py
python src/fec/channel_coding.py
python src/mimo/mimo_system.py
python src/synchronisation/pll.py

# Interactive notebook
jupyter notebook notebooks/DSP_Interactive_Notebook.ipynb
```

---

## Author

**Rizwan** — Cloud & Communications Engineering Student, Munich
GitHub: [@rizwan66](https://github.com/rizwan66)
