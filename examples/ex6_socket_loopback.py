"""
Example 6 — Socket loopback test.

Spins up the receiver in a thread, then runs the transmitter against it.
Demonstrates that the full CLI pipeline works end-to-end on real sockets.
"""
import sys
import time
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.link import run_receiver, run_transmitter


def main():
    port = 9876
    messages = [
        ("Short and sweet.", None,   True),
        ("Noise test at 15 dB.", 15, True),
        ("Very noisy at 5 dB!",   5, True),
        ("Uncoded very noisy at 5 dB!", 5, False),
    ]

    print("=" * 65)
    print("Socket loopback test")
    print("=" * 65)

    for i, (msg, snr, coding) in enumerate(messages, 1):
        print(f"\n--- Test {i}/{len(messages)} ---")
        print(f"  Message : {msg!r}")
        print(f"  SNR     : {snr} dB" if snr is not None else "  SNR     : noiseless")
        print(f"  Coding  : {'Hamming(7,4)' if coding else 'none'}")

        # Start receiver in a thread
        rx_thread = threading.Thread(
            target=run_receiver, args=(port, coding), daemon=True
        )
        rx_thread.start()
        time.sleep(0.15)  # Let the receiver bind before we connect

        # Transmit
        run_transmitter(port, msg, snr, coding)

        # Wait for receiver to finish
        rx_thread.join(timeout=3.0)
        port += 1   # use a fresh port each round (cleaner shutdown)
        time.sleep(0.1)

    print("\n" + "=" * 65)
    print("All tests complete.")


if __name__ == "__main__":
    main()
