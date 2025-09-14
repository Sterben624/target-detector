import numpy as np
import matplotlib.pyplot as plt


class TDOALocalizer:
    def __init__(self, sample_rate=44100, mic_distance=0.10, sound_speed=343.0):
        self.sample_rate = sample_rate
        self.mic_distance = mic_distance
        self.sound_speed = sound_speed
        self.max_tdoa = mic_distance / sound_speed

    def gcc_phat(self, sig1, sig2):
        n = sig1.shape[0] + sig2.shape[0]
        SIG1 = np.fft.rfft(sig1, n=n)
        SIG2 = np.fft.rfft(sig2, n=n)
        R = SIG1 * np.conj(SIG2)
        R /= np.abs(R) + 1e-15
        cc = np.fft.irfft(R, n=n)
        max_shift = int(n / 2)
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift]))
        shift = np.argmax(np.abs(cc)) - max_shift
        tdoa = shift / float(self.sample_rate)
        return tdoa, shift

    def calculate_position(self, audio_buffer):
        if audio_buffer.shape[1] != 2:
            raise ValueError("Audio buffer must have 2 channels (stereo)")

        ch1 = audio_buffer[:, 0]
        ch2 = audio_buffer[:, 1]

        tdoa, shift = self.gcc_phat(ch1, ch2)

        if abs(tdoa) > self.max_tdoa:
            return {
                'tdoa': tdoa,
                'shift': shift,
                'angle_deg': None,
                'norm_shift': 0,
                'valid': False
            }

        angle_rad = np.arcsin(tdoa * self.sound_speed / self.mic_distance)
        angle_deg = np.degrees(angle_rad)
        norm_shift = np.clip(np.sin(angle_rad), -1, 1)

        return {
            'tdoa': tdoa,
            'shift': shift,
            'angle_deg': angle_deg,
            'norm_shift': norm_shift,
            'valid': True
        }

    def process_stereo_buffer(self, audio_buffer):
        ch1 = audio_buffer[:, 0]
        ch2 = audio_buffer[:, 1]

        fft_ch1 = np.abs(np.fft.rfft(ch1))
        fft_ch2 = np.abs(np.fft.rfft(ch2))

        position_data = self.calculate_position(audio_buffer)

        return {
            'ch1': ch1,
            'ch2': ch2,
            'fft_ch1': fft_ch1,
            'fft_ch2': fft_ch2,
            'position': position_data
        }


class TDOAVisualizer:
    def __init__(self, chunk_size=1024, sample_rate=44100):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.paused = False

        self.fig, self.axs = plt.subplots(3, 2, figsize=(12, 9))
        self.time_ax1, self.freq_ax1 = self.axs[0]
        self.time_ax2, self.freq_ax2 = self.axs[1]
        self.pos_ax, _ = self.axs[2]

        self._setup_plots()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _setup_plots(self):
        x_time = np.arange(0, self.chunk_size)
        x_freq = np.fft.rfftfreq(self.chunk_size, 1/self.sample_rate)

        self.line_time1, = self.time_ax1.plot(x_time, np.zeros(self.chunk_size), label="Time Domain CH1")
        self.line_freq1, = self.freq_ax1.plot(x_freq, np.zeros(len(x_freq)), label="FFT CH1")

        self.line_time2, = self.time_ax2.plot(x_time, np.zeros(self.chunk_size), label="Time Domain CH2")
        self.line_freq2, = self.freq_ax2.plot(x_freq, np.zeros(len(x_freq)), label="FFT CH2")

        self.pos_ax.set_title("Положення джерела звуку (GCC-PHAT)")
        self.pos_ax.set_xlim(-1, 1)
        self.pos_ax.set_ylim(-0.5, 0.5)
        self.pos_ax.set_xlabel("Відхилення від центру (норм.)")
        self.pos_ax.get_yaxis().set_visible(False)
        self.pos_marker, = self.pos_ax.plot([0], [0], marker='o', color='green', markersize=20)
        self.pos_ax.axvline(0, color='gray', linestyle='--', linewidth=1)

        for ax in [self.time_ax1, self.time_ax2]:
            ax.set_ylim(-32768, 32767)
            ax.set_xlim(0, self.chunk_size)
            ax.legend()

        for ax in [self.freq_ax1, self.freq_ax2]:
            ax.set_ylim(0, 10000)
            ax.set_xlim(0, self.sample_rate / 2)
            ax.legend()

        plt.tight_layout()
        plt.ion()
        plt.show()

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
            print("⏸️ Paused" if self.paused else "▶️ Resumed")

    def update(self, processed_data):
        if self.paused:
            plt.pause(0.1)
            return

        position = processed_data['position']

        if position['valid']:
            title = f"Положення (кут: {position['angle_deg']:.1f}°; norm: {position['norm_shift']:.2f}; TDOA={position['tdoa']:.5f})"
        else:
            title = f"Положення джерела звуку: не визначено (TDOA={position['tdoa']:.5f}, shift={position['shift']})"

        self.pos_ax.set_title(title)

        self.line_time1.set_ydata(processed_data['ch1'])
        self.line_time2.set_ydata(processed_data['ch2'])
        self.line_freq1.set_ydata(processed_data['fft_ch1'])
        self.line_freq2.set_ydata(processed_data['fft_ch2'])
        self.pos_marker.set_data([position['norm_shift']], [0])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    import pyaudio

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    DEVICE_INDEX = 0

    localizer = TDOALocalizer(sample_rate=RATE)
    visualizer = TDOAVisualizer(chunk_size=CHUNK, sample_rate=RATE)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=DEVICE_INDEX,
                    frames_per_buffer=CHUNK)

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            data_int = np.frombuffer(data, dtype=np.int16)
            audio_buffer = data_int.reshape(-1, 2)

            processed_data = localizer.process_stereo_buffer(audio_buffer)

            position = processed_data['position']
            print(f"TDOA: {position['tdoa']:.6f} с, shift: {position['shift']}, norm_shift: {position['norm_shift']:.2f}")

            visualizer.update(processed_data)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()