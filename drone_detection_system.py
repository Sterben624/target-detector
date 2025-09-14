import numpy as np
import torch
import torch.nn.functional as F
import joblib
from collections import deque
import time

from audio_listener.audio_receiver import AudioReceiver
from drone_detector.band_extractor import BandExtractor
from drone_detector.band_features_extractor import BandFeatureExtractor
from drone_detector.drone_detection_network_simple import DroneDetectionNetwork
from tdoa_localizer.tdoa_localizer import TDOALocalizer
from utils.logger_utils import setup_logger


class DroneDetectionSystem:
    def __init__(self,
                 sample_rate=44100,
                 detection_threshold=0.5,
                 positive_threshold=3,
                 buffer_duration=1.0,
                 model_path=None,
                 scaler_path=None):

        self.sample_rate = sample_rate
        self.detection_threshold = detection_threshold
        self.positive_threshold = positive_threshold
        self.buffer_size = int(sample_rate * buffer_duration)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = setup_logger(
            name="DroneDetectionSystem",
            level="INFO",
            file_level="DEBUG",
            console_output=True,
            log_file="logs/drone_detection.log"
        )

        self.audio_receiver = AudioReceiver(save_to_buffer=True, save_to_file=True)
        self.band_extractor = BandExtractor()
        self.feature_extractor = BandFeatureExtractor(sr=sample_rate)
        self.detection_network = DroneDetectionNetwork()
        self.detection_network.eval()
        self.tdoa_localizer = TDOALocalizer(sample_rate=sample_rate)

        self.scaler = None
        if scaler_path:
            self._load_scaler(scaler_path)

        if model_path:
            self._load_model(model_path)

        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.detection_history = deque(maxlen=10)
        self.positive_detections = 0
        self.samples_since_last_process = 0
        self.overlap_step = self.buffer_size // 2
        self.first_buffer_filled = False

        self.running = False

        self.logger.info("DroneDetectionSystem initialized")

    def _load_scaler(self, scaler_path):
        try:
            self.scaler = joblib.load(scaler_path)
            self.logger.info(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {e}")

    def _load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.detection_network.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.detection_network.load_state_dict(checkpoint)
            self.detection_network.eval()
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

    def _process_audio_chunk(self, audio_data):
        # Извлекаем моно аудио
        mono_audio = audio_data[:, 0] if len(audio_data.shape) == 2 else audio_data

        # Нормализуем int16 в float32 как librosa
        mono_audio = mono_audio.astype(np.float32) / 32768.0

        # Фильтруем полосу 650-800 Hz
        band_signal = self.band_extractor.extract_band(mono_audio)

        # Извлекаем признаки
        features = self.feature_extractor.extract(band_signal)

        # Нормализуем признаки
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()

        # Предсказание
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.detection_network(features_tensor)
            probability = prediction.item()
            confidence = max(probability, 1 - probability)

        is_drone = probability > self.detection_threshold

        self.detection_history.append({
            'timestamp': time.time(),
            'probability': probability,
            'is_drone': is_drone,
            'features': features,
            'confidence': confidence
        })

        if is_drone:
            self.positive_detections += 1
            self.logger.warning(f"Drone detected! Probability: {probability:.3f} "
                             f"confidence: {confidence:.3f} "
                           f"(Positive count: {self.positive_detections})")
        else:
            # self.positive_detections = max(0, self.positive_detections - 1)
            self.positive_detections = 0
            self.logger.info(f"Not found drone... Probability: {probability:.3f} "
                    f"confidence: {confidence:.3f} "
                f"(Positive count: {self.positive_detections})")

        if self.positive_detections >= self.positive_threshold:
            position = self._trigger_localization(audio_data)
            self.logger.info(f"Drone confirmed! Position: {position}")
            self.positive_detections = 0

        return probability, is_drone

    def _trigger_localization(self, audio_data):
        self.logger.info("Triggering TDOA localization...")

        if len(audio_data.shape) != 2 or audio_data.shape[1] != 2:
            self.logger.warning("TDOA localization requires stereo audio - skipping")
            return

        try:
            position_data = self.tdoa_localizer.calculate_position(audio_data)

            if position_data['valid']:
                self.logger.info(f"Drone localized: angle={position_data['angle_deg']:.1f}°, "
                               f"norm_shift={position_data['norm_shift']:.3f}")
            else:
                self.logger.warning("TDOA localization failed - position invalid")

            return position_data

        except Exception as e:
            self.logger.error(f"TDOA localization error: {e}")
            return None

    def update(self):
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
        data_int = np.frombuffer(data, dtype=np.int16)
        if len(data_int) < 2:
            return

        audio_stereo = data_int.reshape(-1, 2)
        for sample in audio_stereo:
            self._add_sample_to_buffer(sample)

    def _add_sample_to_buffer(self, sample):
        self.audio_buffer.append(sample)
        self.samples_since_last_process += 1

        if self._should_process_buffer():
            audio_chunk = np.array(list(self.audio_buffer))
            self._process_audio_chunk(audio_chunk)

    def _should_process_buffer(self):
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
        if self.running:
            self.logger.warning("System already running")
            return

        self.logger.info("Starting drone detection system...")

        if self.audio_receiver.save_to_file:
            self.audio_receiver.setup_audio_file()
        self.audio_receiver.setup_socket()
        self.running = True

        self.logger.info("Drone detection system started")

    def stop(self):
        if not self.running:
            return

        self.logger.info("Stopping drone detection system...")

        self.running = False
        self.audio_receiver.cleanup()

        self.logger.info("Drone detection system stopped")

    def get_status(self):
        return {
            'running': self.running,
            'positive_detections': self.positive_detections,
            'buffer_level': len(self.audio_buffer),
            'recent_detections': list(self.detection_history)[-5:] if self.detection_history else []
        }


if __name__ == "__main__":
    system = DroneDetectionSystem(model_path="drone_detector/best_drone_detection_model.pth",
                                  scaler_path="drone_detector/feature_scaler.pkl")

    try:
        system.start()

        while system.running:
            system.update()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        system.stop()