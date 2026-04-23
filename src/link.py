"""
Live Transmission Over Network Sockets
======================================

Transmitter serializes a modulated signal and streams IQ samples to a
receiver over a TCP socket. The receiver demodulates and prints the result.

This is NOT a radio — it's a way to see the full TX-RX pipeline run across
two processes, like a real software-defined radio link with a loopback
"channel". You can inject noise or fading in the TX path before streaming.

Usage:
    # Terminal 1:
    python -m src.link receive --port 9000
    # Terminal 2:
    python -m src.link transmit --port 9000 --message "Hello" --snr 12
"""
from __future__ import annotations
import argparse
import socket
import struct
import numpy as np

from .digital import (qpsk_modulate, qpsk_demodulate,
                      pulse_shape, matched_filter, rrc_filter)
from .channel import awgn
from .coding import hamming74_encode, hamming74_decode, bit_error_rate


# ---------------------------------------------------------------------------
# Wire format:
#   [4-byte BE uint32 : payload_len_in_samples]
#   [4-byte BE uint32 : original_text_len_in_bytes]
#   [payload_len * 8 bytes : float32 I, float32 Q, ...]
# ---------------------------------------------------------------------------

HEADER = struct.Struct("!II")


def text_to_bits(text: str) -> np.ndarray:
    """ASCII -> bit array (MSB first per byte)."""
    arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.int8)


def bits_to_text(bits: np.ndarray) -> str:
    """Bit array -> UTF-8 text, ignoring trailing garbage."""
    # Truncate to byte boundary
    n = (len(bits) // 8) * 8
    arr = np.packbits(bits[:n].astype(np.uint8))
    try:
        return arr.tobytes().decode("utf-8", errors="replace")
    except Exception:
        return arr.tobytes().hex()


def serialize(iq: np.ndarray, text_len: int) -> bytes:
    """Pack IQ samples into the wire format."""
    header = HEADER.pack(len(iq), text_len)
    # Interleave I and Q, convert to float32
    flat = np.empty(2 * len(iq), dtype=np.float32)
    flat[0::2] = iq.real
    flat[1::2] = iq.imag
    return header + flat.tobytes()


def deserialize(sock: socket.socket) -> tuple[np.ndarray, int]:
    """Read a full frame from the socket and return (iq, original_text_len)."""
    hdr = _recvall(sock, HEADER.size)
    n_samples, text_len = HEADER.unpack(hdr)
    payload = _recvall(sock, n_samples * 8)
    flat = np.frombuffer(payload, dtype=np.float32)
    iq = flat[0::2] + 1j * flat[1::2]
    return iq, text_len


def _recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed early")
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Transmit / receive end-to-end
# ---------------------------------------------------------------------------

SPS = 8
BETA = 0.35
SPAN = 10


def build_transmit_signal(text: str, snr_db: float | None = None,
                          use_coding: bool = True) -> tuple[np.ndarray, int]:
    """
    Convert text -> Hamming-coded -> QPSK -> RRC-shaped -> (optional AWGN).
    Returns (baseband_complex_signal, original_text_byte_length).
    """
    bits = text_to_bits(text)
    text_len = len(text.encode("utf-8"))
    if use_coding:
        bits = hamming74_encode(bits)
    # Pad to even bit count for QPSK
    if len(bits) % 2:
        bits = np.concatenate([bits, [0]])
    symbols = qpsk_modulate(bits)
    shaped, _ = pulse_shape(symbols, BETA, SPAN, SPS)
    if snr_db is not None:
        shaped = awgn(shaped, snr_db)
    return shaped.astype(np.complex64), text_len


def receive_signal(iq: np.ndarray, text_len: int,
                   use_coding: bool = True) -> tuple[str, float]:
    """
    Run matched filter, sample, demodulate, decode.
    Returns (decoded_text, estimated_BER_if_we_knew_bits).
    """
    h = rrc_filter(BETA, SPAN, SPS)
    symbols_all = matched_filter(iq, h, SPS, SPAN)

    # The transmitter sent exactly len(iq) / SPS symbols.
    # We recover that count from the IQ length so we never
    # hand the wrong number of bits to hamming74_decode.
    n_tx_symbols = len(iq) // SPS
    symbols = symbols_all[:n_tx_symbols]

    bits = qpsk_demodulate(symbols)           # exactly 2 * n_tx_symbols bits

    if use_coding:
        # hamming74_encode pads to multiple of 7; trim to that multiple
        n_coded = (len(bits) // 7) * 7
        bits = bits[:n_coded]
        bits, n_corr = hamming74_decode(bits)

    # Trim to original text length
    wanted_bits = text_len * 8
    bits = bits[:wanted_bits]
    return bits_to_text(bits), 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_receiver(port: int, use_coding: bool = True):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)
        print(f"[RX] Listening on 127.0.0.1:{port} ...")
        conn, addr = s.accept()
        with conn:
            print(f"[RX] Connected by {addr}")
            iq, text_len = deserialize(conn)
            print(f"[RX] Received {len(iq)} IQ samples "
                  f"(expecting {text_len} bytes of text)")
            text, _ = receive_signal(iq, text_len, use_coding=use_coding)
            print(f"[RX] Decoded message: {text!r}")


def run_transmitter(port: int, message: str, snr_db: float | None,
                    use_coding: bool = True):
    iq, text_len = build_transmit_signal(message, snr_db, use_coding)
    print(f"[TX] Message: {message!r}")
    print(f"[TX] {len(iq)} IQ samples at {SPS} sps, SNR = {snr_db} dB, "
          f"coding = {use_coding}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("127.0.0.1", port))
        s.sendall(serialize(iq, text_len))
        print(f"[TX] Sent {len(iq)} samples.")


def main():
    parser = argparse.ArgumentParser(description="Communication link over TCP.")
    sub = parser.add_subparsers(dest="mode", required=True)

    rx = sub.add_parser("receive")
    rx.add_argument("--port", type=int, default=9000)
    rx.add_argument("--no-coding", action="store_true")

    tx = sub.add_parser("transmit")
    tx.add_argument("--port", type=int, default=9000)
    tx.add_argument("--message", type=str, required=True)
    tx.add_argument("--snr", type=float, default=None,
                    help="SNR in dB (omit for noiseless)")
    tx.add_argument("--no-coding", action="store_true")

    args = parser.parse_args()
    if args.mode == "receive":
        run_receiver(args.port, use_coding=not args.no_coding)
    else:
        run_transmitter(args.port, args.message, args.snr,
                        use_coding=not args.no_coding)


if __name__ == "__main__":
    main()
