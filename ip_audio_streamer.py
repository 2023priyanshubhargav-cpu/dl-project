"""
IP Webcam Audio Streamer
Captures audio from IP Webcam (e.g., Android IP Webcam app) 
and streams it into a queue for speech recognition.
"""

import requests
import numpy as np
import threading
import queue
import struct

class IPWebcamAudioStreamer:
    """
    Streams audio from IP Webcam HTTP endpoint.
    IP Webcam app provides audio via HTTP stream at: http://[IP]:8080/audio.wav
    """
    
    def __init__(self, ip_address, port=8080, sample_rate=48000, chunk_size=None):
        """
        Args:
            ip_address: IP or hostname of the device running IP Webcam (e.g., "192.168.2.1")
            port: Port number (default 8080)
            sample_rate: Audio sample rate (Hz)
            chunk_size: Number of samples per chunk (default calculated from sample_rate)
        """
        self.ip_address = ip_address
        self.port = port
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size or int(sample_rate * 1.0)  # 1.0 seconds
        self.base_url = f"http://{ip_address}:{port}"
        self.audio_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.stream_thread = None
        
    def start(self):
        """Start streaming audio from IP Webcam."""
        if self.is_running:
            print("[WARN] Audio stream already running")
            return
        
        self.is_running = True
        self.stream_thread = threading.Thread(target=self._stream_audio_loop, daemon=True)
        self.stream_thread.start()
        print(f"✓ IP Webcam audio stream started: {self.base_url}")
        
    def stop(self):
        """Stop streaming audio."""
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        print("✓ IP Webcam audio stream stopped")
        
    def _stream_audio_loop(self):
        """Main loop to fetch and queue audio chunks."""
        audio_url = f"{self.base_url}/audio.wav"
        
        while self.is_running:
            try:
                # Stream audio from IP Webcam
                response = requests.get(audio_url, stream=True, timeout=5)
                if response.status_code != 200:
                    print(f"[WARN] IP Webcam audio returned status {response.status_code}, retrying...")
                    continue
                
                # Skip WAV header (44 bytes)
                wav_header = response.raw.read(44)
                if len(wav_header) < 44:
                    print("[WARN] Incomplete WAV header, retrying...")
                    continue
                
                print("[INFO] IP Webcam audio connection established")
                
                # Read audio chunks
                bytes_per_sample = 2  # 16-bit audio
                chunk_bytes = self.chunk_size * bytes_per_sample
                
                while self.is_running:
                    audio_chunk = response.raw.read(chunk_bytes)
                    if not audio_chunk:
                        print("[WARN] IP Webcam stream ended, reconnecting...")
                        break
                    
                    # Convert bytes to float32 numpy array
                    try:
                        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
                        audio_float32 = audio_int16.astype(np.float32) / 32768.0
                        
                        # Queue audio chunk
                        if not self.audio_queue.full():
                            self.audio_queue.put(audio_float32)
                    except Exception as e:
                        print(f"[WARN] Error converting audio: {e}")
                        break
                
            except requests.exceptions.RequestException as e:
                print(f"[WARN] IP Webcam connection error: {e}. Retrying in 0.5 seconds...")
                if self.is_running:
                    import time
                    time.sleep(0.5)
            except Exception as e:
                print(f"[ERROR] Unexpected error in audio stream: {e}")
                if self.is_running:
                    import time
                    time.sleep(0.5)
    
    def get_chunk(self, timeout=1.0):
        """
        Get next audio chunk from queue.
        
        Returns:
            np.ndarray: Audio chunk as float32, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# Example usage
if __name__ == "__main__":
    import time
    
    # Test connection
    print("Testing IP Webcam audio stream...")
    streamer = IPWebcamAudioStreamer(ip_address="192.168.2.1", port=8080)
    
    try:
        streamer.start()
        time.sleep(2)
        
        # Try to read a few chunks
        for i in range(3):
            chunk = streamer.get_chunk(timeout=3.0)
            if chunk is not None:
                print(f"✓ Chunk {i+1}: {len(chunk)} samples")
            else:
                print(f"✗ Chunk {i+1}: No data received")
            time.sleep(0.5)
        
    finally:
        streamer.stop()
    
    print("Test complete")
