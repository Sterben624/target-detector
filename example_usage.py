import time
from drone_detection_system import DroneDetectionSystem


def main():
    system = DroneDetectionSystem(
        mode='quad_plus_shahed',
        use_bandpass=False,
        detection_threshold=0.5,
        positive_threshold=3,
        udp_port=8889,
        audio_channels=2
    )

    try:
        system.start()
        print("Drone detection system started...")
        print(f"Listening on UDP port {system.audio_receiver.port}")
        print("Send audio with: python audio_sender.py")
        print("Press Ctrl+C to stop\n")

        while system.running:
            system.update()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        system.stop()
        print("System stopped")


if __name__ == "__main__":
    main()
