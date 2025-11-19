# Drone Detection System

Система детекції дронів-шахедів на основі ResNet6-CBAM та акустичних сигналів.

## Запуск

### Базовий запуск
```bash
python drone_detection_system.py
```

### З параметрами
```bash
python drone_detection_system.py --mode quad_plus_shahed --bandpass --port 8889
```

### Тест на файлі
```bash
python test_with_file.py my_audio.wav
python test_with_file.py my_audio.wav --mode quad_plus_pure_shahed --bandpass
```

### Як підсистема
```python
from drone_detection_system import DroneDetectionSystem

system = DroneDetectionSystem(
    mode='quad_plus_shahed',
    use_bandpass=False,
    udp_port=8889
)

system.start()
while system.running:
    system.update()
system.stop()
```

Див. `example_usage.py` для детального прикладу.

## Параметри командної строки

```
--mode                  Режим роботи (quad_plus_shahed | quad_plus_pure_shahed)
--bandpass              Використовувати bandpass фільтр 700-850 Hz
--model <path>          Власний шлях до моделі (перевизначає --mode)
--threshold <float>     Поріг детекції (default: 0.5)
--port <int>            UDP порт (default: 8889)
--channels <1|2>        Кількість аудіо каналів: 1=mono, 2=stereo (default: 2)
```

## Структура файлів

### Основні файли
- **`drone_detection_system.py`** - Головний модуль системи
- **`example_usage.py`** - Приклад використання як підсистеми

### Допоміжні файли (можна видалити)
- **`test_with_file.py`** - Тест детектора на аудіо файлі
- **`audio_sender.py`** - Відправка аудіо через UDP (для тестування)

### Директорії
- **`shahed_detector/`** - Детектор на базі ResNet6-CBAM
  - `shahed_detector_module.py` - Модуль детектора
  - `resnet6_cbam.py` - Архітектура моделі
  - `cbam.py` - CBAM attention
  - **`models/`** - Файли весів:
    - `quad_plus_shahed_fullband.pth` - Квадрокоптер + Шахед (весь спектр)
    - `quad_plus_shahed_bandpass.pth` - Квадрокоптер + Шахед (700-850 Hz)
    - `quad_plus_pure_shahed_fullband.pth` - Квадрокоптер + Шахед + Чистий шахед (весь спектр)
    - `quad_plus_pure_shahed_bandpass.pth` - Квадрокоптер + Шахед + Чистий шахед (700-850 Hz)

- **`audio_listener/`** - Прийом аудіо
  - `audio_receiver.py` - UDP приймач (порт налаштовується через параметр `port`)

- **`tdoa_localizer/`** - TDOA локалізація
- **`utils/`** - Логування та обробка сигналів

## Режими роботи

### `quad_plus_shahed` (default)
Модель навчена на:
- Клас 0: Квадрокоптер
- Клас 1: Квадрокоптер + Шахед

### `quad_plus_pure_shahed`
Модель навчена на:
- Клас 0: Квадрокоптер
- Клас 1: Квадрокоптер + Шахед + Чистий шахед (без квадрокоптера)

### Bandpass vs Full Spectrum
- **Full spectrum** (default) - використовує весь частотний діапазон
- **--bandpass** - фільтрує 700-850 Hz (для шумного середовища)

## UDP налаштування

**Default:**
- Порт: `8889`
- Формат: mono або stereo, 16-bit, 44100 Hz
- Підтримка: автоматично визначає моно/стерео з AudioReceiver

**Mono:**
- Дублюється в обидва канали для TDOA
- Використовується для детекції

**Stereo:**
- Вибирається гучніший канал для детекції
- Обидва канали використовуються для TDOA локалізації

**Змінити порт:**
- Через CLI: `--port 9000`
- Через код: `DroneDetectionSystem(udp_port=9000)`

**Змінити кількість каналів:**
- Через CLI: `--channels 1` (моно) або `--channels 2` (стерео)
- Через код: `DroneDetectionSystem(audio_channels=1)`

## Параметри DroneDetectionSystem

```python
DroneDetectionSystem(
    mode='quad_plus_shahed',        # Режим роботи
    use_bandpass=False,              # Bandpass фільтр 700-850 Hz
    model_path=None,                 # Власний шлях до моделі
    detection_threshold=0.5,         # Поріг детекції (0-1)
    positive_threshold=3,            # К-сть підряд детекцій для локалізації
    udp_port=8889,                   # UDP порт
    audio_channels=2,                # Кількість каналів (1=mono, 2=stereo)
    sample_rate=44100,               # Частота дискретизації
    buffer_duration=1.0,             # Тривалість буфера (секунди)
    freq_min=700.0,                  # Мін частота bandpass
    freq_max=850.0                   # Макс частота bandpass
)
```

## Встановлення

```bash
pip install -r requirements.txt
```

Основні залежності: `torch`, `torchaudio`, `numpy`, `scipy`, `librosa`

## Логи та вивід

- `logs/drone_detection.log` - Головний лог
- `logs/audio_receiver.log` - Лог прийому аудіо
- `logs_audio/received_audio_*.wav` - Записані аудіо файли
- `logs_audio/debug_sample.wav` - Перший chunk для дебагу
