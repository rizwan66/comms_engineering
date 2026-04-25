# 📡 Communications Engineering — DSP Learning Project

> Comprehensive, hands-on DSP covering theory + Python simulations for signals, filters, modulation, noise cancellation, transceivers, OFDM, FEC, MIMO, and GNU Radio.

---

## 🗂️ Project Structure

```
comms_engineering/
├── src/
│   ├── signals/generator.py           # Signal generation, FFT, STFT, convolution
│   ├── filters/design.py              # FIR, IIR, Butterworth, LMS, RLS, Notch
│   ├── modulation/schemes.py          # AM, FM, BPSK, QPSK, QAM, OFDM, BER
│   ├── noise_cancellation/canceller.py # Spectral sub, Wiener, LMS/RLS ANC
│   ├── transceivers/chain.py          # Full TX→Channel→RX chain, eye diagram
│   ├── ofdm/ofdm_system.py            # ★ CP, IFFT/FFT, multipath, ZF equaliser
│   ├── fec/channel_coding.py          # ★ LDPC+BP, Turbo, Viterbi, Shannon limit
│   ├── synchronisation/pll.py         # ★ 2nd-order PLL, Costas, Mueller-Müller
│   ├── mimo/mimo_system.py            # ★ ZF/MMSE equaliser, Alamouti, capacity
│   └── gnu_radio/gr_blocks.py         # ★ GNU Radio blocks, RTL-SDR/USRP flowgraph
├── simulations/full_chain_demo.py
├── docs/                              # 6 theory markdown files
└── assets/plots/                      # 35 pre-generated figures
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/rizwan66/comms_engineering.git
cd comms_engineering && pip install -r requirements.txt

python src/ofdm/ofdm_system.py        # OFDM + cyclic prefix + multipath
python src/fec/channel_coding.py      # LDPC/Turbo vs Shannon limit
python src/synchronisation/pll.py     # Costas PLL + M&M timing recovery
python src/mimo/mimo_system.py        # MIMO ZF/MMSE/Alamouti + capacity
python src/gnu_radio/gr_blocks.py     # GNU Radio simulation flowgraph
python simulations/full_chain_demo.py # Full system master demo
```

---

## 📊 Key Formulas

| Concept | Formula |
|---------|---------|
| Shannon capacity | C = B·log₂(1 + SNR) |
| OFDM (no ISI) | CP length ≥ max multipath delay |
| MIMO capacity | C = log₂ det(I + ρ/Nₜ·HHᴴ) |
| ZF equaliser | W = (HᴴH)⁻¹Hᴴ |
| MMSE equaliser | W = (HᴴH + σ²I)⁻¹Hᴴ |
| Costas error | e[n] = sign(I)·Q |
| M&M timing | e[n] = d̂[n-1]·x[n] − d̂[n]·x[n-1] |
| BPSK BER | Q(√(2Eb/N₀)) |

---

## 👤 Author
**Rizwan** · [@rizwan66](https://github.com/rizwan66)
