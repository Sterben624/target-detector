#!/usr/bin/env python3
"""
Простой экстрактор частотной полосы 650-750 Hz для детекции Шахед.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class BandExtractor:
    """Простой экстрактор частотной полосы для детекции Шахед."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.freq_low = 650.0
        self.freq_high = 800.0
        
    def extract_band(self, signal: np.ndarray) -> np.ndarray:
        """Фильтрация полосы 650-750 Hz."""
        nyquist = self.sr / 2
        low_norm = self.freq_low / nyquist
        high_norm = self.freq_high / nyquist
        
        b, a = butter(4, [low_norm, high_norm], btype='band')
        return filtfilt(b, a, signal)
    
    def process(self, input_file: str) -> np.ndarray:
        """
        Обработка файла: загрузка и фильтрация полосы 650-750 Hz.
        Для стерео файлов выбирает канал с максимальной энергией в целевой полосе.
        """
        signal, _ = librosa.load(input_file, sr=self.sr, mono=False)
        
        if signal.ndim == 2:  # Стерео файл
            # Фильтруем каждый канал отдельно
            left_filtered = self.extract_band(signal[0])
            right_filtered = self.extract_band(signal[1])
            
            # Вычисляем энергию в целевой полосе для каждого канала
            left_energy = np.sum(left_filtered ** 2)
            right_energy = np.sum(right_filtered ** 2)
            
            # Выбираем канал с большей энергией
            if left_energy > right_energy:
                return left_filtered
            else:
                return right_filtered
        else:  # Моно файл
            return self.extract_band(signal)
    
def main():
    """Простой тест экстрактора."""
    extractor = BandExtractor()
    
    test_file = "raw_audio/cuted_5s/target_only/target_equal_gain_000.wav"
    if Path(test_file).exists():
        filtered = extractor.process(test_file)
        
        # Сохранить результат
        output_file = "test_filtered_650_750.wav"
        max_val = np.max(np.abs(filtered))
        if max_val > 0:
            normalized = filtered / max_val * 0.9
            sf.write(output_file, normalized, extractor.sr)
            print(f"✅ Отфильтрованный файл сохранен: {output_file}")
    else:
        print(f"❌ Тестовый файл не найден: {test_file}")


if __name__ == "__main__":
    main()