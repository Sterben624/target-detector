"""
Shahed Detector Module
Використовує ResNet6-CBAM для детекції звуку шахедів на основі MFCC ознак
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
from scipy.signal import butter, filtfilt
from pathlib import Path

from .resnet6_cbam import create_resnet6_cbam


class ShahedDetector:
    """
    Детектор шахедів на основі ResNet6-CBAM та MFCC ознак

    Підтримує два режими:
    - Full spectrum (all bands) - рекомендовано
    - Bandpass 700-850 Hz - для шумного середовища
    """

    def __init__(
        self,
        model_path: str,
        use_bandpass: bool = False,
        freq_min: float = 700.0,
        freq_max: float = 850.0,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_mels: int = 128,
        duration: float = 1.0,
        device: str = None
    ):
        """
        Ініціалізація детектора

        Args:
            model_path: Шлях до weights файлу (.pth)
            use_bandpass: Використовувати bandpass фільтр (700-850 Hz)
            freq_min: Мінімальна частота bandpass
            freq_max: Максимальна частота bandpass
            sample_rate: Частота дискретизації (16000 Hz)
            n_mfcc: Кількість MFCC коефіцієнтів (40)
            n_mels: Кількість mel-фільтрів (128)
            duration: Тривалість сегменту (1.0 секунда)
            device: 'cuda', 'cpu' або None (auto)
        """
        self.use_bandpass = use_bandpass
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.duration = duration
        self.target_length = int(duration * sample_rate)

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Ініціалізація моделі
        self.model = create_resnet6_cbam(
            num_classes=2,
            dropout_rate=0.2,
            reduction_ratio=16
        )

        # Завантаження weights
        self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': 512,
                'hop_length': 256,
                'n_mels': self.n_mels,
                'center': False
            }
        )

    def _load_model(self, model_path: str):
        """Завантажує модель з checkpoint"""
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def _apply_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Застосовує bandpass фільтр (Butterworth 4-го порядку)"""
        nyquist = self.sample_rate / 2
        low = self.freq_min / nyquist
        high = self.freq_max / nyquist

        b, a = butter(N=4, Wn=[low, high], btype='band')
        filtered = filtfilt(b, a, audio)

        return filtered

    def preprocess_audio(self, audio: np.ndarray, original_sr: int) -> torch.Tensor:
        """
        Обробляє аудіо та витягує MFCC ознаки

        Args:
            audio: Numpy array з аудіо (моно або стерео)
            original_sr: Оригінальна частота дискретизації

        Returns:
            MFCC tensor shape: (1, n_mfcc, time_frames)
        """
        # Конвертація в float32 якщо int16
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        # Вибір моно каналу (гучніший)
        if audio.ndim == 2:
            # Стерео -> моно (вибір гучнішого каналу)
            channel_energy = np.sqrt(np.mean(audio ** 2, axis=0))
            loudest_channel = np.argmax(channel_energy)
            audio = audio[:, loudest_channel]
        elif audio.ndim > 2:
            audio = audio.flatten()

        # Конвертація в torch tensor
        waveform = torch.from_numpy(audio).float().unsqueeze(0)

        # Resample якщо потрібно
        if original_sr != self.sample_rate:
            resampler = T.Resample(orig_freq=original_sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Bandpass фільтр (опціонально)
        if self.use_bandpass:
            audio_np = waveform.squeeze(0).numpy()
            filtered = self._apply_bandpass_filter(audio_np)
            waveform = torch.from_numpy(filtered).float().unsqueeze(0)

        # Pre-emphasis (посилення високих частот)
        # Використовуємо librosa.effects.preemphasis щоб відповідати training pipeline
        audio_np = waveform.squeeze(0).numpy()
        pre_emphasized = librosa.effects.preemphasis(audio_np)
        waveform = torch.from_numpy(pre_emphasized).float().unsqueeze(0)

        # Padding/trimming до потрібної довжини
        current_length = waveform.shape[1]
        if current_length < self.target_length:
            # Padding
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > self.target_length:
            # Trimming
            waveform = waveform[:, :self.target_length]

        # MFCC екстракція
        mfcc = self.mfcc_transform(waveform)

        return mfcc.unsqueeze(0)  # (1, 1, n_mfcc, time_frames)

    def predict(self, audio: np.ndarray, original_sr: int) -> dict:
        """
        Класифікує аудіо

        Args:
            audio: Numpy array з аудіо
            original_sr: Оригінальна частота дискретизації

        Returns:
            dict з результатами:
                - 'predicted_class': 0 (без шахеда) або 1 (з шахедом)
                - 'class_name': 'Без шахеда' або 'З шахедом'
                - 'probability': Ймовірність класу 1 (з шахедом)
                - 'confidence': Впевненість (max probability)
                - 'probabilities': dict з ймовірностями обох класів
        """
        # Preprocessing
        mfcc = self.preprocess_audio(audio, original_sr)
        mfcc = mfcc.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(mfcc)
            probabilities = torch.softmax(outputs, dim=1)
            prob_shahed = probabilities[0, 1].item()
            prob_no_shahed = probabilities[0, 0].item()

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        # Результати
        class_names = ['Без шахеда', 'З шахедом']

        return {
            'predicted_class': predicted_class,
            'class_name': class_names[predicted_class],
            'probability': prob_shahed,  # Ймовірність SHAHED (class 1)
            'confidence': confidence,
            'probabilities': {
                'без_шахеда': prob_no_shahed,
                'з_шахедом': prob_shahed
            }
        }

    def predict_from_buffer(self, audio_buffer: np.ndarray, original_sr: int) -> dict:
        """
        Класифікує аудіо з буфера (зручно для real-time обробки)

        Args:
            audio_buffer: Numpy array з аудіо (моно або стерео)
            original_sr: Оригінальна частота дискретизації

        Returns:
            dict з результатами prediction
        """
        return self.predict(audio_buffer, original_sr)

    def get_info(self) -> dict:
        """Повертає інформацію про детектор"""
        return {
            'model': 'ResNet6-CBAM',
            'use_bandpass': self.use_bandpass,
            'freq_range': f"{self.freq_min}-{self.freq_max} Hz" if self.use_bandpass else "Full spectrum",
            'sample_rate': self.sample_rate,
            'n_mfcc': self.n_mfcc,
            'n_mels': self.n_mels,
            'duration': self.duration,
            'device': str(self.device)
        }


# Тест модуля
if __name__ == "__main__":
    import sys

    print("Testing ShahedDetector...")

    # Створити детектор (full spectrum)
    try:
        detector = ShahedDetector(
            model_path="best_model_all_bands.pth",
            use_bandpass=False
        )
        print("\n✓ Full spectrum detector loaded")
        print(detector.get_info())
    except Exception as e:
        print(f"\n✗ Failed to load detector: {e}")
        sys.exit(1)

    # Тест на рандомному аудіо
    print("\nTesting on random audio...")
    test_audio = np.random.randn(16000).astype(np.float32)  # 1 секунда @ 16kHz

    try:
        result = detector.predict(test_audio, original_sr=16000)
        print(f"\n✓ Prediction successful:")
        print(f"  Class: {result['class_name']}")
        print(f"  Probability: {result['probability']:.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        sys.exit(1)

    print("\n✓ All tests passed!")
