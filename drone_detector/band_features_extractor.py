#!/usr/bin/env python3
"""
Экстрактор признаков специально для полосы 650-800 Hz.
Оптимизирован для детекции Шахед в узком частотном диапазоне.
"""

import numpy as np
import librosa


class BandFeatureExtractor:
    
    def __init__(self, sr: int = 44100, freq_low: float = 700.0, freq_high: float = 850.0, n_mfcc: int = 8):
        self.sr = sr
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.n_mfcc = n_mfcc
        self.freq_center = (freq_low + freq_high) / 2
        
    def _select_louder_channel(self, audio: np.ndarray) -> np.ndarray:
        """Select the channel with higher RMS energy"""
        if len(audio.shape) == 1:
            return audio

        if audio.shape[1] != 2:
            return audio.mean(axis=1)

        ch1_rms = np.sqrt(np.mean(audio[:, 0] ** 2))
        ch2_rms = np.sqrt(np.mean(audio[:, 1] ** 2))

        return audio[:, 0] if ch1_rms >= ch2_rms else audio[:, 1]

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Извлекает признаки из отфильтрованного сигнала 650-800 Hz:
        
        1. RMS энергия (1)
        2. Пиковая амплитуда (1) 
        3. Zero crossing rate (1)
        4. Спектральный пик в полосе (1)
        5. Позиция пика относительно центра (1)
        6. Ширина главного пика (1)
        7. Мини-MFCC (8 коэффициентов для узкой полосы)
        8. Статистики спектра (mean, std, skew, kurt) (4)
        9. Гармоническая чистота (1)
        10. Спектральная резкость (1) 
        11. Временная когерентность (1)
        
        Всего: 21 признак
        """
        audio = self._select_louder_channel(audio)

        if len(audio) == 0:
            return np.zeros(21)
            
        # 1. RMS энергия
        rms_energy = np.sqrt(np.mean(audio ** 2))
        
        # 2. Пиковая амплитуда
        peak_amplitude = np.max(np.abs(audio))
        
        # 3. Zero crossing rate
        zero_crossings = np.diff(np.sign(audio))
        zcr = np.sum(np.abs(zero_crossings)) / (2 * len(audio))
        
        # 4-8. Спектральные признаки
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        # Ограничить анализ только нашей полосой
        band_mask = (freqs >= self.freq_low) & (freqs <= self.freq_high)
        band_fft = fft[band_mask]
        band_freqs = freqs[band_mask]
        
        if len(band_fft) > 0:
            # 4. Спектральный пик в полосе
            peak_idx = np.argmax(band_fft)
            spectral_peak = band_fft[peak_idx]
            
            # 5. Позиция пика относительно центра полосы (-1 до 1)
            peak_freq = band_freqs[peak_idx]
            peak_position = (peak_freq - self.freq_center) / ((self.freq_high - self.freq_low) / 2)
            
            # 6. Ширина главного пика (на уровне -3dB)
            half_max = spectral_peak / 2
            indices_above_half = np.where(band_fft >= half_max)[0]
            if len(indices_above_half) > 1:
                peak_width = band_freqs[indices_above_half[-1]] - band_freqs[indices_above_half[0]]
            else:
                peak_width = (self.freq_high - self.freq_low) / len(band_fft)  # Минимальная ширина
            
            # Нормализация ширины пика
            peak_width_norm = peak_width / (self.freq_high - self.freq_low)
            
            # 7. Статистики спектра в полосе
            spec_mean = np.mean(band_fft)
            spec_std = np.std(band_fft)
            
            # Skewness (асимметрия)
            if spec_std > 0:
                spec_skew = np.mean(((band_fft - spec_mean) / spec_std) ** 3)
            else:
                spec_skew = 0
                
            # Kurtosis (эксцесс)
            if spec_std > 0:
                spec_kurt = np.mean(((band_fft - spec_mean) / spec_std) ** 4) - 3
            else:
                spec_kurt = 0
                
        else:
            spectral_peak = 0
            peak_position = 0
            peak_width_norm = 1
            spec_mean = 0
            spec_std = 0
            spec_skew = 0
            spec_kurt = 0
        
        # 8. Мини-MFCC (8 коэффициентов)
        try:
            # Для узкой полосы используем 8 MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc, 
                                       fmin=self.freq_low, fmax=self.freq_high)
            mfccs_mean = mfccs.mean(axis=1)
        except:
            mfccs_mean = np.zeros(8)
        
        # 9. Гармоническая чистота - отношение энергии главного пика к сумме всех пиков
        if len(band_fft) > 0 and np.sum(band_fft) > 0:
            harmonic_purity = spectral_peak / np.sum(band_fft)
        else:
            harmonic_purity = 0
        
        # 10. Спектральная резкость - крутизна спадов вокруг главного пика
        if len(band_fft) > 2:
            peak_idx = np.argmax(band_fft)
            # Взять окрестности пика (±2 бина)
            left_idx = max(0, peak_idx - 2)
            right_idx = min(len(band_fft), peak_idx + 3)
            
            # Вычислить градиенты слева и справа от пика
            if peak_idx > 0:
                left_gradient = np.mean(np.diff(band_fft[left_idx:peak_idx+1]))
            else:
                left_gradient = 0
                
            if peak_idx < len(band_fft) - 1:
                right_gradient = np.mean(np.diff(band_fft[peak_idx:right_idx]))
            else:
                right_gradient = 0
            
            # Средняя крутизна (абсолютная)
            spectral_sharpness = (abs(left_gradient) + abs(right_gradient)) / 2
        else:
            spectral_sharpness = 0
        
        # 11. Временная когерентность - автокорреляция огибающей
        # Вычислить огибающую сигнала
        if len(audio) > 100:
            # Разбить сигнал на окна
            window_size = len(audio) // 10  # 10 окон
            if window_size > 10:
                envelope = []
                for i in range(0, len(audio) - window_size, window_size):
                    window = audio[i:i + window_size]
                    rms = np.mean(window ** 2)
                    envelope.append(np.sqrt(max(0, rms)))
                
                envelope = np.array(envelope)
                
                # Автокорреляция огибающей с лагом 1
                if len(envelope) > 1:
                    temporal_coherence = np.corrcoef(envelope[:-1], envelope[1:])[0, 1]
                    if np.isnan(temporal_coherence):
                        temporal_coherence = 0
                else:
                    temporal_coherence = 0
            else:
                temporal_coherence = 0
        else:
            temporal_coherence = 0
        
        # Собрать все признаки
        features = np.array([
            rms_energy,          # 1
            peak_amplitude,      # 2  
            zcr,                # 3
            spectral_peak,      # 4
            peak_position,      # 5
            peak_width_norm,    # 6
            spec_mean,          # 7
            spec_std,           # 8
            spec_skew,          # 9
            spec_kurt,          # 10
            *mfccs_mean,        # 11-18 (8 MFCC)
            harmonic_purity,    # 19
            spectral_sharpness, # 20
            temporal_coherence  # 21
        ])
        
        # Проверка на NaN/Inf и замена на нули
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def get_feature_names(self) -> list:
        """Возвращает названия признаков."""
        return [
            'rms_energy',
            'peak_amplitude', 
            'zero_crossing_rate',
            'spectral_peak',
            'peak_position',
            'peak_width',
            'spectral_mean',
            'spectral_std', 
            'spectral_skew',
            'spectral_kurt',
            'mfcc_1',
            'mfcc_2', 
            'mfcc_3',
            'mfcc_4',
            'mfcc_5',
            'mfcc_6',
            'mfcc_7',
            'mfcc_8',
            'harmonic_purity',
            'spectral_sharpness',
            'temporal_coherence'
        ]


def main():
    """Тест экстрактора признаков."""
    from pathlib import Path
    from band_extractor import BandExtractor
    import soundfile as sf
    
    print("🔍 Тестирование BandFeatureExtractor...")
    
    # Создать экстракторы
    band_extractor = BandExtractor()
    feature_extractor = BandFeatureExtractor()
    
    # Тестовый файл
    test_file = "raw_audio/cuted_5s/target_only/target_equal_gain_000.wav"
    
    if Path(test_file).exists():
        print(f"📁 Тестовый файл: {test_file}")
        
        # Извлечь полосу 650-800 Hz
        filtered_signal = band_extractor.process(test_file)
        print(f"📊 Длина отфильтрованного сигнала: {len(filtered_signal)} сэмплов")
        
        # Извлечь признаки
        features = feature_extractor.extract(filtered_signal)
        feature_names = feature_extractor.get_feature_names()
        
        print(f"🎯 Извлечено признаков: {len(features)}")
        print("\n📋 ПРИЗНАКИ:")
        print("-" * 40)
        
        for name, value in zip(feature_names, features):
            print(f"{name:20s}: {value:10.6f}")
        
        # Проверка на корректность
        if not (np.isnan(features).any() or np.isinf(features).any()):
            print("\n✅ Все признаки корректны (нет NaN/Inf)")
        else:
            print("\n❌ Найдены некорректные значения!")
        
        print(f"\n💡 Эти {len(features)} признаков оптимизированы для полосы 650-800 Hz")
        
        # Показать новые признаки отдельно
        print("\n🆕 НОВЫЕ ПРИЗНАКИ:")
        print("-" * 40)
        new_features = ['harmonic_purity', 'spectral_sharpness', 'temporal_coherence']
        for name in new_features:
            idx = feature_names.index(name)
            print(f"{name:20s}: {features[idx]:10.6f}")
        
    else:
        print(f"❌ Тестовый файл не найден: {test_file}")


if __name__ == "__main__":
    main()