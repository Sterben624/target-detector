import socket
import wave
from datetime import datetime
from utils.logger_utils import setup_logger
from utils.signal_handler import SignalHandler

logger = setup_logger(
    name="AudioReceiver",
    level="INFO",
    file_level="DEBUG",
    console_output=True,
    log_file="logs/audio_receiver.log"
)

class AudioReceiver:
    def __init__(self, host='0.0.0.0', port=8889,
                 save_to_buffer=True, save_to_file=True, audio_file=None):
        self.host = host
        self.port = port
        self.buffer = []
        self.sock = None
        self.audio_file = audio_file
        self.sample_rate = 44100
        self.channels = 2
        self.sample_width = 2
        self.save_to_buffer = save_to_buffer
        self.save_to_file = save_to_file
        self.signal_handler = SignalHandler()
        self.wav_file = None
        
        if audio_file is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            audio_file = f'logs_audio/received_audio_{timestamp}.wav'
        self.audio_file = audio_file
        logger.info(f"AudioReceiver initialized: buffer={save_to_buffer}, file={save_to_file}")

    def setup_audio_file(self):
        """Initialize the audio file for writing"""
        try:
            self.wav_file = wave.open(self.audio_file, 'wb')
            self.wav_file.setnchannels(self.channels)
            self.wav_file.setsampwidth(self.sample_width)
            self.wav_file.setframerate(self.sample_rate)
            logger.info(f"Audio file {self.audio_file} initialized.")
        except Exception as e:
            logger.error(f"Error initializing audio file: {e}")

    def setup_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(1.0)
        logger.info(f"Socket set up on {self.host}:{self.port}")

    def receive_data(self, buffer_size=65536):
        if not self.sock:
            raise RuntimeError("Socket not initialized. Call setup_socket() first.")
        
        try:
            data, addr = self.sock.recvfrom(buffer_size)
            logger.info(f"Received {len(data)} bytes from {addr}")
            return data
        except socket.timeout:
            logger.warning("Socket timed out waiting for data.")
            return None
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None

    def save_data(self, data):
        if self.save_to_buffer:
            self._save_to_buffer(data)
        if self.save_to_file:
            if not self._save_to_wav(data):
                logger.error("Failed to save audio data to file.")
        logger.info(f"Data saved: {len(data)} bytes")
        
    def _save_to_buffer(self, data):
        self.buffer.append(data)
        logger.debug(f"Buffer len now: {len(self.buffer)}")

    def _save_to_wav(self, data):
        try:
            if self.wav_file:
                self.wav_file.writeframes(data)
                logger.debug(f"Audio data saved: {len(data)} bytes")
                return True
            else:
                logger.error("WAV file not initialized")
                return False
        except Exception as e:
            logger.error(f"Error saving audio data: {e}")
            return False

    def close_audio_file(self):
        """Properly close the WAV file"""
        if self.wav_file:
            try:
                self.wav_file.close()
                logger.info("WAV file closed properly")
            except Exception as e:
                logger.error(f"Error closing WAV file: {e}")

    def cleanup(self):
        """Clean up resources"""
        self.close_audio_file()
        if self.sock:
            self.sock.close()
            logger.info("Socket closed")

if __name__ == "__main__":
    my_socket = AudioReceiver(save_to_buffer=True, save_to_file=True)
    my_socket.setup_audio_file()
    my_socket.setup_socket()
    
    try:
        while not my_socket.signal_handler.shutdown_requested:
            data = my_socket.receive_data()
            if not data:
                continue
            my_socket.save_data(data)
    finally:
        my_socket.cleanup()
        logger.info("Exiting...")