"""
Audio Capture and UDP Sender (Python version)
Захоплює звук з мікрофона та відправляє через UDP
"""

import socket
import pyaudio
import time
import sys

# Параметри
TARGET_IP = "127.0.0.1"
TARGET_PORT = 8889
SAMPLE_RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 4096

def main():
    print("🎤 Audio Capture -> UDP Sender")
    print("="*40)
    print(f"Target: {TARGET_IP}:{TARGET_PORT}")
    print(f"Format: {SAMPLE_RATE} Hz, {CHANNELS} channels, 16-bit")
    print(f"Chunk size: {CHUNK} frames")
    print("Press Ctrl+C to stop...\n")

    # UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # PyAudio
    audio = pyaudio.PyAudio()

    # Вибрати пристрій
    print("Available audio devices:")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} (channels: {info['maxInputChannels']})")

    # Використати дефолтний пристрій
    device_index = None  # None = default device

    try:
        # Відкрити stream
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )

        print(f"\n✓ Audio stream opened")
        print("✓ Recording started\n")

        packet_count = 0
        total_bytes = 0
        start_time = time.time()

        while True:
            # Читати аудіо
            data = stream.read(CHUNK, exception_on_overflow=False)

            # Відправити через UDP
            sock.sendto(data, (TARGET_IP, TARGET_PORT))

            packet_count += 1
            total_bytes += len(data)

            # Статистика кожні 10 пакетів
            if packet_count % 10 == 0:
                elapsed = time.time() - start_time
                kb_per_sec = (total_bytes / 1024) / elapsed
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] Packets: {packet_count:5d} | "
                      f"Bytes: {total_bytes:8d} | "
                      f"Speed: {kb_per_sec:6.2f} KB/s")

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        audio.terminate()
        sock.close()
        print("✅ Stopped cleanly")

if __name__ == "__main__":
    main()
