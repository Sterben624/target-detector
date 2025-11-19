import numpy as np
import time
from collections import deque

from audio_listener.audio_receiver import AudioReceiver
from shahed_detector.shahed_detector_module import ShahedDetector
from tdoa_localizer.tdoa_localizer import TDOALocalizer
from utils.logger_utils import setup_logger


class DroneDetectionSystem:
    # Доступні режими роботи
    MODES = {
        'quad_plus_shahed': {
            'fullband': 'shahed_detector/models/quad_plus_shahed_fullband.pth',
            'bandpass': 'shahed_detector/models/quad_plus_shahed_bandpass.pth',
            'description': 'Квадрокоптер + Шахед'
        },
        'quad_plus_pure_shahed': {
            'fullband': 'shahed_detector/models/quad_plus_pure_shahed_fullband.pth',
            'bandpass': 'shahed_detector/models/quad_plus_pure_shahed_bandpass.pth',
            'description': 'Квадрокоптер + Шахед + Чистий Шахед'
        }
    }

    def __init__(self,
                 sample_rate=44100,
                 detection_threshold=0.5,
                 positive_threshold=3,
                 buffer_duration=1.0,
                 mode='quad_plus_shahed',
                 use_bandpass=False,
                 model_path=None,
                 freq_min=700.0,
                 freq_max=850.0,
                 udp_port=8889,
                 audio_channels=2):
        """
        Drone Detection System з ResNet6-CBAM моделлю

        Args:
            sample_rate: Частота дискретизації сокету (44100 Hz)
            detection_threshold: Поріг ймовірності для детекції (0.5)
            positive_threshold: Кількість позитивних детекцій для підтвердження (3)
            buffer_duration: Тривалість буфера в секундах (1.0)
            mode: Режим роботи ('quad_plus_shahed' або 'quad_plus_pure_shahed')
            use_bandpass: Використовувати bandpass фільтр (700-850 Hz)
            model_path: Прямий шлях до weights файлу (перевизначає mode)
            freq_min: Мінімальна частота bandpass (700.0)
            freq_max: Максимальна частота bandpass (850.0)
            udp_port: UDP порт для прийому аудіо (8889)
            audio_channels: Кількість аудіо каналів (1=mono, 2=stereo)
        """
        self.sample_rate = sample_rate
        self.detection_threshold = detection_threshold
        self.positive_threshold = positive_threshold
        self.buffer_size = int(sample_rate * buffer_duration)

        self.logger = setup_logger(
            name="DroneDetectionSystem",
            level="INFO",
            file_level="DEBUG",
            console_output=True,
            log_file="logs/drone_detection.log"
        )

        self.audio_receiver = AudioReceiver(port=udp_port, channels=audio_channels, save_to_buffer=True, save_to_file=True)

        if model_path is None:
            if mode not in self.MODES:
                raise ValueError(f"Unknown mode: {mode}. Available modes: {list(self.MODES.keys())}")

            band_type = 'bandpass' if use_bandpass else 'fullband'
            model_path = self.MODES[mode][band_type]
            mode_desc = self.MODES[mode]['description']
        else:
            mode_desc = "Custom model"

        self.logger.info(f"Mode: {mode_desc}")
        self.logger.info(f"Loading model: {model_path}")
        self.logger.info(f"Bandpass filter: {'ON (700-850 Hz)' if use_bandpass else 'OFF (full spectrum)'}")

        self.detector = ShahedDetector(
            model_path=model_path,
            use_bandpass=use_bandpass,
            freq_min=freq_min,
            freq_max=freq_max,
            sample_rate=16000,
            n_mfcc=40,
            n_mels=128,
            duration=1.0
        )

        detector_info = self.detector.get_info()
        for key, value in detector_info.items():
            self.logger.info(f"  {key}: {value}")

        self.tdoa_localizer = TDOALocalizer(sample_rate=sample_rate)

        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.detection_history = deque(maxlen=10)
        self.positive_detections = 0
        self.samples_since_last_process = 0
        self.overlap_step = self.buffer_size // 2
        self.first_buffer_filled = False
        self.running = False

        self.logger.info("DroneDetectionSystem initialized with ResNet6-CBAM")

    def _process_audio_chunk(self, audio_data):
        """
        Обробляє chunk аудіо через ResNet6-CBAM

        Args:
            audio_data: numpy array shape (samples, 2) для stereo
        """
        if not hasattr(self, '_saved_sample'):
            import wave
            sample_path = 'logs_audio/debug_sample.wav'
            try:
                audio_int16 = (audio_data * 32768.0).astype(np.int16)
                with wave.open(sample_path, 'wb') as wf:
                    wf.setnchannels(audio_int16.shape[1] if len(audio_int16.shape) == 2 else 1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_int16.tobytes())
                self.logger.info(f"Debug sample saved to {sample_path}")
            except Exception as e:
                self.logger.error(f"Failed to save debug sample: {e}")
            self._saved_sample = True

        if len(audio_data.shape) == 2 and audio_data.shape[1] == 2:
            channel_energy = np.sqrt(np.mean(audio_data ** 2, axis=0))
            loudest_channel = np.argmax(channel_energy)
            mono_audio = audio_data[:, loudest_channel]
            self.logger.debug(f"Stereo input - Ch0 energy: {channel_energy[0]:.2f}, "
                            f"Ch1 energy: {channel_energy[1]:.2f}, "
                            f"Selected: {loudest_channel}")
        else:
            mono_audio = audio_data.flatten()
            self.logger.debug(f"Mono input - Shape: {audio_data.shape}")

        result = self.detector.predict(mono_audio, original_sr=self.sample_rate)

        probability = result['probability']
        confidence = result['confidence']
        predicted_class = result['predicted_class']
        class_name = result['class_name']

        audio_rms = np.sqrt(np.mean(mono_audio ** 2))
        audio_peak = np.max(np.abs(mono_audio))
        self.logger.debug(f"Audio stats - RMS: {audio_rms:.4f}, Peak: {audio_peak:.4f}, "
                         f"Shape: {mono_audio.shape}, Min: {mono_audio.min():.1f}, Max: {mono_audio.max():.1f}")

        is_drone = predicted_class == 1
        self.detection_history.append({
            'timestamp': time.time(),
            'probability': probability,
            'is_drone': is_drone,
            'confidence': confidence,
            'class_name': class_name
        })

        prob_no_shahed = result['probabilities']['без_шахеда']
        prob_shahed = result['probabilities']['з_шахедом']

        if is_drone:
            self.positive_detections += 1
            self.logger.warning(f"Shahed detected! Prob: {probability:.3f}, "
                             f"Conf: {confidence:.3f}, "
                             f"Count: {self.positive_detections}, "
                             f"[No:{prob_no_shahed:.3f} | Yes:{prob_shahed:.3f}]")
        else:
            self.positive_detections = 0
            self.logger.info(f"No shahed. Prob: {probability:.3f}, "
                          f"Conf: {confidence:.3f}, "
                          f"[No:{prob_no_shahed:.3f} | Yes:{prob_shahed:.3f}]")
        if self.positive_detections >= self.positive_threshold:
            position = self._trigger_localization(audio_data)
            self.logger.info(f"Shahed confirmed! Position: {position}")
            self.positive_detections = 0

        return probability, is_drone

    def _trigger_localization(self, audio_data):
        """Запускає TDOA локалізацію"""
        self.logger.info("Triggering TDOA localization...")

        if len(audio_data.shape) != 2 or audio_data.shape[1] != 2:
            self.logger.warning("TDOA localization requires stereo audio - skipping")
            return None

        try:
            position_data = self.tdoa_localizer.calculate_position(audio_data)

            if position_data['valid']:
                self.logger.info(f"Shahed localized: angle={position_data['angle_deg']:.1f}°, "
                               f"norm_shift={position_data['norm_shift']:.3f}")
            else:
                self.logger.warning("TDOA localization failed - position invalid")

            return position_data

        except Exception as e:
            self.logger.error(f"TDOA localization error: {e}")
            return None

    def update(self):
        """Основний цикл обробки"""
        if not self.running:
            return

        try:
            data = self.audio_receiver.receive_data()
            if data:
                self.audio_receiver.save_data(data)
                self._process_received_data(data)
        except Exception as e:
            self.logger.debug(f"No data received or error: {e}")

    def _process_received_data(self, data):
        """Обробка даних з сокету"""
        data_int = np.frombuffer(data, dtype=np.int16)
        if len(data_int) < 1:
            return

        channels = self.audio_receiver.channels

        if not hasattr(self, '_logged_packet_info'):
            self.logger.info(f"First packet: {len(data)} bytes, {len(data_int)} samples (int16), channels: {channels}")
            self._logged_packet_info = True

        if channels == 1:
            for sample_value in data_int:
                stereo_sample = np.array([sample_value, sample_value], dtype=np.int16)
                self._add_sample_to_buffer(stereo_sample)
        elif channels == 2:
            data_stereo = data_int.reshape(-1, 2)
            for stereo_sample in data_stereo:
                self._add_sample_to_buffer(stereo_sample)
        else:
            self.logger.error(f"Unsupported number of channels: {channels}")
            return

    def _add_sample_to_buffer(self, sample):
        """Додає семпл до буфера"""
        self.audio_buffer.append(sample)
        self.samples_since_last_process += 1

        if self._should_process_buffer():
            audio_chunk = np.array(list(self.audio_buffer), dtype=np.int16)
            audio_chunk_float = audio_chunk.astype(np.float32) / 32768.0
            self._process_audio_chunk(audio_chunk_float)

    def _should_process_buffer(self):
        """Перевіряє чи потрібно обробляти буфер"""
        if not self.first_buffer_filled:
            if len(self.audio_buffer) == self.buffer_size:
                self.first_buffer_filled = True
                self.samples_since_last_process = 0
                return True
            return False

        if self.samples_since_last_process >= self.overlap_step:
            self.samples_since_last_process = 0
            return True

        return False

    def start(self):
        """Запуск системи"""
        if self.running:
            self.logger.warning("System already running")
            return

        self.logger.info("Starting drone detection system with ResNet6-CBAM...")

        if self.audio_receiver.save_to_file:
            self.audio_receiver.setup_audio_file()
        self.audio_receiver.setup_socket()
        self.running = True

        self.logger.info("Drone detection system started")

    def stop(self):
        """Зупинка системи"""
        if not self.running:
            return

        self.logger.info("Stopping drone detection system...")

        self.running = False
        self.audio_receiver.cleanup()

        self.logger.info("Drone detection system stopped")

    def get_status(self):
        """Отримання статусу системи"""
        return {
            'running': self.running,
            'positive_detections': self.positive_detections,
            'buffer_level': len(self.audio_buffer),
            'recent_detections': list(self.detection_history)[-5:] if self.detection_history else [],
            'detector_info': self.detector.get_info()
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Drone Detection System with ResNet6-CBAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  quad_plus_shahed         - Quadcopter + Shahed (default)
  quad_plus_pure_shahed    - Quadcopter + Shahed + Pure Shahed

Examples:
  python drone_detection_system.py
  python drone_detection_system.py --mode quad_plus_shahed --bandpass
  python drone_detection_system.py --mode quad_plus_pure_shahed
  python drone_detection_system.py --model custom_model.pth
        """
    )
    parser.add_argument('--mode', type=str, default='quad_plus_shahed',
                       choices=['quad_plus_shahed', 'quad_plus_pure_shahed'],
                       help='Detection mode (default: quad_plus_shahed)')
    parser.add_argument('--bandpass', action='store_true',
                       help='Use bandpass filter 700-850 Hz (default: full spectrum)')
    parser.add_argument('--model', type=str, default=None,
                       help='Custom path to model weights (overrides --mode)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold (default: 0.5)')
    parser.add_argument('--port', type=int, default=8889,
                       help='UDP port for audio reception (default: 8889)')
    parser.add_argument('--channels', type=int, default=2, choices=[1, 2],
                       help='Audio channels: 1=mono, 2=stereo (default: 2)')

    args = parser.parse_args()

    system = DroneDetectionSystem(
        mode=args.mode,
        use_bandpass=args.bandpass,
        model_path=args.model,
        detection_threshold=args.threshold,
        udp_port=args.port,
        audio_channels=args.channels
    )

    try:
        system.start()

        print("\n" + "="*60)
        print("Drone Detection System Running")
        print("="*60)
        print(f"Model: ResNet6-CBAM")
        print(f"Mode: {args.mode}")
        print(f"Filter: {'Bandpass (700-850 Hz)' if args.bandpass else 'Full Spectrum'}")
        print(f"Threshold: {args.threshold}")
        print("="*60)
        print("Press Ctrl+C to stop...\n")

        while system.running:
            system.update()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        system.stop()
