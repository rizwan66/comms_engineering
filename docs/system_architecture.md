# System Architecture & Module Map

## Project Overview

This project implements a complete, layered communications engineering stack in pure Python (NumPy/SciPy). Every formula is coded from first principles — no black-box RF libraries — so you can trace signal transformations at every step.

---

## Module Dependency Map

```mermaid
graph TD
    subgraph FOUNDATION["Layer 0 — Foundation"]
        SIG["signals/generator.py<br/>Signal gen · FFT · PSD · STFT"]
        FIL["filters/design.py<br/>IIR · FIR · LMS · RLS · Notch"]
    end

    subgraph BASEBAND["Layer 1 — Baseband"]
        MOD["modulation/schemes.py<br/>AM · FM · BPSK · QPSK · QAM · OFDM"]
        NC["noise_cancellation/canceller.py<br/>Wiener · Spectral Sub · ANC"]
        CH["channel.py<br/>AWGN · Multipath · Rayleigh · Rician"]
        COD["coding.py<br/>Hamming(7,4) · Repetition · BER theory"]
    end

    subgraph ADVANCED["Layer 2 — Advanced PHY"]
        TC["transceivers/chain.py<br/>Full TX→RX · RRC · Eye diagram"]
        OFDM["ofdm/ofdm_system.py<br/>Pilots · LS/MMSE · PAPR"]
        FEC["fec/channel_coding.py<br/>LDPC · Viterbi · Convolutional"]
        MIMO["mimo/mimo_system.py<br/>ZF · MMSE · Alamouti · Capacity"]
        SYNC["synchronisation/pll.py<br/>PLL · Costas · Mueller-Müller"]
    end

    subgraph SIGNAL_PROC["Layer 2 — Signal Processing"]
        WAV["wavelets/wavelet_transform.py<br/>CWT · DWT · Denoising"]
        KAL["kalman/kalman_filter.py<br/>KF · EKF · RTS Smoother"]
    end

    subgraph SYSTEMS["Layer 3 — Complete Systems"]
        LTE["lte_simulator/lte_downlink.py<br/>PDSCH · EPA · Viterbi · MMSE"]
        OTFS["otfs/otfs_system.py<br/>DD domain · ISFFT/SFFT · Heisenberg"]
        ML["ml_classifier/modulation_classifier.py<br/>CNN · 8 classes · IQ features"]
        GR["gnu_radio/gr_blocks.py<br/>RTL-SDR · USRP · Flowgraph"]
        LNK["link.py<br/>TCP socket · Full pipeline"]
    end

    SIG --> MOD
    SIG --> TC
    FIL --> NC
    FIL --> TC
    MOD --> TC
    MOD --> OFDM
    CH --> TC
    CH --> MIMO
    COD --> LNK
    TC --> OFDM
    TC --> LTE
    FEC --> LTE
    OFDM --> LTE
    OFDM --> OTFS
    MIMO --> LTE
    SYNC --> TC
    WAV --> KAL
    ML --> GR
```

---

## Data Flow: End-to-End Communication Chain

```mermaid
flowchart LR
    A([Source Text / Bits]) --> B[Channel Encoder\nHamming · LDPC · Conv]
    B --> C[Modulator\nBPSK / QPSK / QAM]
    C --> D[Pulse Shaper\nRRC β=0.35]
    D --> E[Up-converter\nfc carrier mix]
    E --> F{Wireless Channel}
    F --> G[AWGN]
    F --> H[Multipath]
    F --> I[Rayleigh / Rician Fading]
    G & H & I --> J[Down-converter]
    J --> K[Matched Filter\nRRC receive]
    K --> L[Equalizer\nZF / MMSE]
    L --> M[Synchronizer\nPLL / Costas / M&M]
    M --> N[Demodulator\nHard / Soft decisions]
    N --> O[Channel Decoder\nViterbi / BP / Hamming]
    O --> P([Recovered Bits / Text])

    style A fill:#2196F3,color:#fff
    style P fill:#4CAF50,color:#fff
    style F fill:#FF5722,color:#fff
```

---

## OFDM Frame Structure (Time-Frequency Grid)

```
Frequency (subcarriers)
  ▲
  │  [P][ ][ ][ ][P][ ][ ][ ][P]  ← Symbol 0 (P = pilot, [ ] = data)
  │  [ ][P][ ][ ][ ][P][ ][ ][ ]  ← Symbol 1
  │  [ ][ ][P][ ][ ][ ][P][ ][ ]  ← Symbol 2
  │  ...
  └──────────────────────────────► Time (OFDM symbols)
       ◄──── OFDM Frame ───────►
```

```mermaid
flowchart LR
    subgraph TX["OFDM Transmitter"]
        b[Bits] --> qm[QAM Mapper]
        qm --> pi[Pilot Insert]
        pi --> ifft[N-point IFFT]
        ifft --> cp[+Cyclic Prefix]
        cp --> dac[DAC / RF Up]
    end

    subgraph CH2["Channel"]
        dac --> mpc[Multipath + AWGN]
    end

    subgraph RX["OFDM Receiver"]
        mpc --> adc[ADC / RF Down]
        adc --> rcp[Remove CP]
        rcp --> fft[N-point FFT]
        fft --> ls[LS Channel Est.]
        ls --> mmse[MMSE Equalizer]
        mmse --> dem[QAM DeMapper]
        dem --> ob[Bits Out]
    end
```

---

## MIMO Spatial Multiplexing Architecture

```mermaid
graph LR
    subgraph TX_MIMO["MIMO TX  (Nt antennas)"]
        bits2[Bits] --> split[Stream Split]
        split --> s1[Stream 1\nBPSK/QPSK]
        split --> s2[Stream 2\nBPSK/QPSK]
        split --> sn[Stream Nt]
    end

    subgraph CHANNEL_MIMO["MIMO Channel  H: Nr×Nt"]
        H[Complex Rayleigh\nMatrix H]
    end

    subgraph RX_MIMO["MIMO RX  (Nr antennas)"]
        r1[Rx Ant 1]
        r2[Rx Ant 2]
        rn[Rx Ant Nr]
        r1 & r2 & rn --> eq[ZF / MMSE\nEqualizer W]
        eq --> comb[Symbol Combiner]
        comb --> bits_o[Bits Out]
    end

    s1 & s2 & sn --> H --> r1 & r2 & rn
```

---

## LTE Downlink Physical Layer Stack

```mermaid
flowchart TD
    A[Transport Block] --> B[CRC-16 Attachment]
    B --> C[Conv Encoder\nRate 1/2  K=7]
    C --> D[Rate Matching\nPuncturing / Repetition]
    D --> E[QAM Mapper\nQPSK / 16QAM / 64QAM]
    E --> F[Resource Mapper\n600 active subcarriers]
    F --> G[CRS Pilot Insert\n1 in 6 subcarriers]
    G --> H[1024-pt IFFT]
    H --> I[Cyclic Prefix Add\n72 samples normal CP]
    I --> J[EPA Multipath Channel]
    J --> K[CP Remove + FFT]
    K --> L[LS Channel Estimate\nat CRS pilots]
    L --> M[MMSE Equalization]
    M --> N[Soft QAM DeMapper\nLLR output]
    N --> O[Viterbi Decoder]
    O --> P[CRC Check]
    P --> Q[Transport Block Out]
```

---

## Module Quick-Reference

| Module | File | Key API |
|--------|------|---------|
| Signal Generator | `src/signals/generator.py` | `sine_wave`, `compute_fft`, `spectrogram` |
| Filter Design | `src/filters/design.py` | `butterworth_lpf`, `lms_filter`, `rls_filter` |
| Modulation | `src/modulation/schemes.py` | `am_modulate`, `qpsk_modulate`, `ofdm_modulate` |
| Noise Cancel | `src/noise_cancellation/canceller.py` | `spectral_subtraction`, `wiener_filter_freq`, `lms_anc` |
| Transceivers | `src/transceivers/chain.py` | `transmitter`, `root_raised_cosine_filter` |
| OFDM | `src/ofdm/ofdm_system.py` | `OFDMSystem.transmit`, `.receive`, `.compute_ber` |
| FEC | `src/fec/channel_coding.py` | `LDPC.encode/decode`, `ConvolutionalCode.viterbi_decode` |
| MIMO | `src/mimo/mimo_system.py` | `zf_equaliser`, `mmse_equaliser`, `alamouti_encode` |
| Sync | `src/synchronisation/pll.py` | `AnalogPLL`, `CostasLoop`, `MuellerMuller` |
| Wavelets | `src/wavelets/wavelet_transform.py` | `cwt_morlet`, `dwt_multilevel`, soft-threshold |
| Kalman | `src/kalman/kalman_filter.py` | `KalmanFilter`, `EKF`, `build_constant_velocity_kf` |
| ML AMC | `src/ml_classifier/modulation_classifier.py` | `generate_dataset`, CNN train/predict |
| GNU Radio | `src/gnu_radio/gr_blocks.py` | `SourceBlock`, `LowPassFilterBlock`, RTL-SDR/USRP |
| LTE | `src/lte_simulator/lte_downlink.py` | `conv_encode`, `viterbi_decode`, PDSCH pipeline |
| OTFS | `src/otfs/otfs_system.py` | `otfs_tx`, `otfs_rx`, `isfft`, `sfft` |
