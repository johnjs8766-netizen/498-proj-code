import serial
import wave
import struct
import time
import sys

# ===== User settings =====
PORT = "COM3"          # Change to your serial port, e.g. "COM5" on Windows or "/dev/ttyACM0" on Linux
BAUD = 921600
SAMPLE_RATE = 16000
DURATION_SEC = 3
OUTPUT_WAV = "recorded_3s.wav"

MAGIC = b"\xAA\x55"
CHUNK_SAMPLES_EXPECTED = 256
BYTES_PER_SAMPLE = 2
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SEC


def read_exact(ser, n):
    """Read exactly n bytes from serial."""
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise TimeoutError(f"Timeout while reading {n} bytes from serial.")
        buf.extend(chunk)
    return bytes(buf)


def find_magic(ser):
    """Scan serial stream until MAGIC header is found."""
    state = 0
    while True:
        b = ser.read(1)
        if not b:
            raise TimeoutError("Timeout while searching for packet header.")
        if state == 0:
            if b == MAGIC[:1]:
                state = 1
        elif state == 1:
            if b == MAGIC[1:2]:
                return
            elif b == MAGIC[:1]:
                state = 1
            else:
                state = 0


def main():
    print(f"Opening serial port {PORT} at {BAUD} baud...")
    ser = serial.Serial(PORT, BAUD, timeout=2)

    try:
        # Optional: give board time after opening serial, in case it resets
        time.sleep(2)

        collected = bytearray()
        samples_collected = 0

        print("Receiving audio packets...")

        while samples_collected < TARGET_SAMPLES:
            # Find packet header
            find_magic(ser)

            # Read sample count (uint16 little-endian)
            count_bytes = read_exact(ser, 2)
            count = struct.unpack("<H", count_bytes)[0]

            payload_size = count * BYTES_PER_SAMPLE
            payload = read_exact(ser, payload_size)

            # If packet is larger than needed for final chunk, trim it
            remaining_samples = TARGET_SAMPLES - samples_collected
            if count > remaining_samples:
                payload = payload[:remaining_samples * BYTES_PER_SAMPLE]
                count = remaining_samples

            collected.extend(payload)
            samples_collected += count

            print(f"\rCollected {samples_collected}/{TARGET_SAMPLES} samples", end="", flush=True)

        print("\nDone receiving audio.")

        # Save as WAV
        with wave.open(OUTPUT_WAV, "wb") as wf:
            wf.setnchannels(1)         # mono
            wf.setsampwidth(2)         # 16-bit = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(collected)

        print(f"Saved WAV file: {OUTPUT_WAV}")

    finally:
        ser.close()


if __name__ == "__main__":
    main()
