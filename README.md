# 📡 Communications Engineering & DSP

> A comprehensive, hands-on Python project covering **15 modules** of Digital Signal Processing and Communications Engineering — from fundamentals to advanced 5G topics.

**GitHub:** [@rizwan66](https://github.com/rizwan66) | **Repo:** `comms_engineering`

---

## 🗂️ Project Structure

```
comms_engineering/
│
├── src/                                    # 15 Python modules
│   ├── signals/generator.py                # Signal gen, FFT, STFT, convolution
│   ├── filters/design.py                   # FIR/IIR, Butterworth, Chebyshev, Elliptic, LMS, Notch
│   ├── modulation/schemes.py               # AM, FM, BPSK, QPSK, QAM, OFDM, BER
│   ├── noise_cancellation/canceller.py     # Wiener, LMS/RLS ANC, spectral subtraction
│   ├── transceivers/chain.py               # Full TX→RX chain, pulse shaping, eye diagram
│   ├── ofdm/ofdm_system.py                 # Pilots, LS channel estimation, PAPR, CCDF
│   ├── fec/channel_coding.py               # Convolutional, LDPC, Turbo, Shannon limit
│   ├── mimo/mimo_system.py                 # ZF/MMSE, Alamouti, capacity, SVD
│   ├── synchronisation/pll.py             # PLL, Costas loop, M&M timing recovery
│   ├── wavelets/wavelet_transform.py       # CWT (Morlet), DWT (db4/Haar), denoising
│   ├── kalman/kalman_filter.py             # KF, RTS smoother, EKF, KF denoising
│   ├── ml_classifier/modulation_classifier.py  # CNN + RF on IQ data, 8 mod classes
│   ├── gnu_radio/gr_blocks.py              # GNU Radio block sim, RTL-SDR flowgraph
│   ├── lte_simulator/lte_downlink.py       # LTE PDSCH: Viterbi, OFDM, MMSE, EPA
│   └── otfs/otfs_system.py                 # OTFS: ISFFT/SFFT, Heisenberg/Wigner, Doppler
│
├── simulations/full_chain_demo.py          # Master 4×3 dashboard (all modules)
├── notebooks/DSP_Interactive_Notebook.ipynb
├── docs/                                   # 6 detailed theory docs
├── assets/plots/                           # 48 pre-generated figures
└── requirements.txt
```

---

## 🧠 Topics Covered (15 Modules)

| # | Module | Key Topics |
|---|--------|-----------|
| 1 | **Signals & Systems** | CTFT, DTFT, DFT/FFT, Z-transform, Laplace, sampling theorem |
| 2 | **Filters** | Butterworth, Chebyshev I/II, Elliptic, FIR (window, Parks-McClellan), LMS, RLS, Notch |
| 3 | **Modulation** | AM, DSB-SC, FM, PM, BPSK, QPSK, M-QAM, OFDM, BER curves |
| 4 | **Noise Cancellation** | Wiener filter, LMS/RLS, ANC, spectral subtraction, SNR analysis |
| 5 | **Transceivers** | RF front-end, pulse shaping (RC/RRC), matched filter, eye diagram, link budget |
| 6 | **OFDM** | IFFT/FFT, cyclic prefix, pilot insertion, LS channel estimation, PAPR CCDF |
| 7 | **FEC** | Convolutional codes, Viterbi, LDPC Tanner graphs, Turbo, Shannon capacity |
| 8 | **MIMO** | Spatial multiplexing, ZF/MMSE, Alamouti diversity, SVD beamforming, capacity |
| 9 | **Synchronisation** | PLL, Costas loop, frequency correction, Mueller & Müller timing recovery |
| 10 | **Wavelets** | CWT (Morlet), DWT (Haar, db4), MRA, soft/hard denoising, STFT vs CWT |
| 11 | **Kalman Filter** | Standard KF, RTS smoother, EKF for phase tracking, signal denoising |
| 12 | **ML Classifier** | CNN + Random Forest on raw IQ data, 8-class modulation recognition |
| 13 | **GNU Radio** | Software-defined radio blocks, RTL-SDR flowgraph, spectrum pipeline |
| 14 | **LTE Simulator** | PDSCH: K=7 Viterbi, 16QAM/64QAM Gray coding, MMSE equaliser, EPA channel |
| 15 | **OTFS** | Delay-Doppler domain, ISFFT/SFFT, Heisenberg/Wigner, Doppler resilience vs OFDM |

---

## ✅ Module Status & Results

| Module | Status | Key Result |
|--------|--------|-----------|
| Signals | ✅ | Multi-tone FFT, chirp spectrogram, convolution demo |
| Filters | ✅ | FIR/IIR freq response, LMS adaptive, notch filter |
| Modulation | ✅ | FM demod, 16-QAM constellation, BER theory vs sim |
| Noise Cancellation | ✅ | LMS ANC, RLS ANC, spectral subtraction SNR gains |
| Transceivers | ✅ | Full TX→RX BER, eye diagram, EPA channel |
| OFDM | ✅ | Pilot channel estimation, PAPR CCDF |
| FEC | ✅ | BER vs Shannon limit, LDPC Tanner graph |
| MIMO | ✅ | Capacity curves 1×1 to 8×8 |
| Synchronisation | ✅ | PLL lock, Costas carrier recovery |
| **Wavelets** | ✅ | CWT scalogram, DWT MRA, denoising SNR +20dB |
| **Kalman** | ✅ | KF vs RTS smoother RMSE, EKF phase lock |
| **ML Classifier** | ✅ | 82% val accuracy on 8 modulation classes |
| **GNU Radio** | ✅ | RTL-SDR flowgraph sim, spectrum stages |
| **LTE Simulator** | ✅ | QPSK BER=0 @ 3dB, 16QAM @ 12dB, 64QAM @ 18dB |
| **OTFS** | ✅ | SER=0.000 vs OFDM SER=0.742 @ 20dB Doppler channel |

---

## 🚀 Quick Start

```bash
git clone https://github.com/rizwan66/comms_engineering.git
cd comms_engineering
pip install -r requirements.txt

# Master demo — all modules in one figure
python simulations/full_chain_demo.py

# Run any individual module
python src/signals/generator.py
python src/filters/design.py
python src/modulation/schemes.py
python src/noise_cancellation/canceller.py
python src/transceivers/chain.py
python src/ofdm/ofdm_system.py
python src/fec/channel_coding.py
python src/mimo/mimo_system.py
python src/synchronisation/pll.py
python src/wavelets/wavelet_transform.py
python src/kalman/kalman_filter.py
python src/ml_classifier/modulation_classifier.py
python src/gnu_radio/gr_blocks.py
python src/lte_simulator/lte_downlink.py
python src/otfs/otfs_system.py

# Interactive notebook
jupyter notebook notebooks/DSP_Interactive_Notebook.ipynb
```

---

## 📦 Requirements

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
jupyter >= 1.0
```

```bash
pip install -r requirements.txt
```

---

## 📈 Pre-Generated Plots (48 figures)

All figures are in `assets/plots/`. Key ones:

| Plot | Description |
|------|-------------|
| `full_system_demo.png` | Master 4×3 overview — all modules |
| `otfs_system.png` | OTFS vs OFDM in Doppler channel |
| `lte_ber_throughput.png` | LTE PDSCH BER + throughput vs SNR |
| `lte_resource_grid.png` | LTE resource grid with CRS pilots |
| `ml_classifier_results.png` | 8-class modulation recognition dashboard |
| `wavelet_stft_vs_cwt.png` | STFT vs CWT time-frequency resolution |
| `kalman_tracking.png` | KF vs RTS smoother tracking |
| `ofdm_papr_ccdf.png` | OFDM PAPR CCDF vs single carrier |
| `mimo_system.png` | MIMO capacity 1×1 to 8×8 |
| `filters_iir_comparison.png` | Butterworth vs Chebyshev vs Elliptic |
| `mod_ber_bpsk.png` | BER theory vs simulation |

---

## 📚 Learning Path

1. `docs/01_signals_and_systems.md` — transforms, sampling, convolution
2. `src/signals/generator.py` — generate and visualise signals
3. `docs/02_filters.md` + `src/filters/design.py` — filter design
4. `docs/03_modulation.md` + `src/modulation/schemes.py` — AM/FM/QAM
5. `src/ofdm/ofdm_system.py` — OFDM with pilots
6. `src/fec/channel_coding.py` — error correction
7. `src/mimo/mimo_system.py` — spatial multiplexing
8. `src/wavelets/wavelet_transform.py` — time-frequency analysis
9. `src/kalman/kalman_filter.py` — optimal estimation
10. `src/lte_simulator/lte_downlink.py` — full LTE physical layer
11. `src/otfs/otfs_system.py` — next-gen 6G modulation
12. `src/ml_classifier/modulation_classifier.py` — AI for DSP

---

## 👤 Author

**Rizwan** — Cloud & Communications Engineering Student, Munich
GitHub: [@rizwan66](https://github.com/rizwan66)
