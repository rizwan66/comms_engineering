# 📡 Digital Signal Processing — Complete Learning Project

> A comprehensive, hands-on DSP learning resource covering theory, simulations, and real-world communication engineering applications.

---

## 🗂️ Project Structure

```
dsp/
├── docs/                        # In-depth theory notes
│   ├── 01_signals_and_systems.md
│   ├── 02_filters.md
│   ├── 03_modulation.md
│   ├── 04_noise_cancellation.md
│   ├── 05_transceivers.md
│   └── 06_communication_engineering.md
│
├── src/                         # Python implementations
│   ├── signals/                 # Signal generation & analysis
│   ├── filters/                 # FIR, IIR, Butterworth, Chebyshev
│   ├── modulation/              # AM, FM, PM, QAM, OFDM
│   ├── noise_cancellation/      # LMS, RLS, Wiener, ANC
│   └── transceivers/            # TX/RX chain implementations
│
├── notebooks/                   # Jupyter notebooks with visualizations
├── simulations/                 # Full system simulations
└── web/                         # Interactive HTML visualizer
```

---

## 🧠 Topics Covered

| Module | Topics |
|--------|--------|
| **Signals & Systems** | Continuous/discrete signals, Fourier, Laplace, Z-Transform, convolution, sampling |
| **Filters** | FIR, IIR, Butterworth, Chebyshev, Elliptic, Notch, adaptive filters |
| **Modulation** | AM, DSB-SC, FM, PM, FSK, PSK, QAM, OFDM, spread spectrum |
| **Noise Cancellation** | LMS, RLS, Wiener filter, ANC, spectral subtraction |
| **Transceivers** | RF front-end, ADC/DAC, baseband processing, demodulation |
| **Communication Eng.** | Channel models, BER, SNR, Shannon capacity, link budgets |

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/rizwan66/dsp.git
cd dsp

# Install dependencies
pip install -r requirements.txt

# Run a simulation
python simulations/full_chain_demo.py

# Launch the interactive web visualizer
open web/index.html
```

---

## 📦 Requirements

```
numpy
scipy
matplotlib
jupyter
sounddevice  # for real-time audio DSP
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 📚 Learning Path

1. Start with `docs/01_signals_and_systems.md` — build mathematical foundations
2. Explore `src/signals/` — generate and visualize signals in Python
3. Move to `docs/02_filters.md` then `src/filters/` — design digital filters
4. Study `docs/03_modulation.md` + `src/modulation/` — implement AM/FM/QAM
5. Read `docs/04_noise_cancellation.md` — understand adaptive algorithms
6. Connect everything in `simulations/full_chain_demo.py`
7. Use the interactive `web/index.html` to experiment visually

---

## 👤 Author

**Rizwan** — Cloud & DSP Engineering Student  
GitHub: [@rizwan66](https://github.com/rizwan66)
