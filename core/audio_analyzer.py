"""
Audio Modality Analyzer
Captures microphone audio stream using sounddevice, computes RMS energy level,
performs Voice Activity Detection (VAD), and tracks participant speaking time.
"""

import time
import math
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

class AudioAnalyzer:
    def __init__(self, sample_rate=16000, block_size=1024, vad_threshold=0.035):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.vad_threshold = vad_threshold
        
        self.stream = None
        self.is_running = False
        self.current_volume = 0.0
        self.is_speaking = False
        self.total_speaking_seconds = 0.0
        self.session_start_time = time.time()
        self._last_tick = time.time()
        self.speech_bouts_count = 0
        self._was_speaking = False
        self.simulated_mode = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback executed for each audio chunk by sounddevice."""
        if status:
            pass
        # Calculate Root Mean Square (RMS) volume
        rms = float(np.sqrt(np.mean(indata**2)))
        self.current_volume = rms
        self.is_speaking = rms > self.vad_threshold

        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now

        if self.is_speaking:
            self.total_speaking_seconds += dt
            if not self._was_speaking:
                self.speech_bouts_count += 1
        self._was_speaking = self.is_speaking

    def start(self, device_index=None):
        """Starts the audio monitoring stream."""
        if not SOUNDDEVICE_AVAILABLE:
            self.simulated_mode = True
            self.is_running = True
            return True

        try:
            self.session_start_time = time.time()
            self._last_tick = time.time()
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                device=device_index,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_running = True
            self.simulated_mode = False
            return True
        except Exception as e:
            # Fall back to simulated audio monitoring if hardware device fails
            print(f"Audio device init failed ({e}), using simulated audio monitor.")
            self.simulated_mode = True
            self.is_running = True
            return False

    def stop(self):
        """Stops the audio stream."""
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def get_metrics(self):
        """Returns current acoustic metrics dictionary."""
        if self.simulated_mode:
            # In simulated mode, emit light fluctuating ambient background noise
            t = time.time()
            # Occasional simulated vocal activity every 15 seconds
            cycle = math.sin(t / 2.0)
            self.current_volume = max(0.01, 0.02 + 0.015 * cycle)
            self.is_speaking = False

        elapsed = max(1.0, time.time() - self.session_start_time)
        speaking_ratio = round((self.total_speaking_seconds / elapsed) * 100.0, 1)

        # Normalized volume on 0 - 100 scale for UI bars
        norm_vol = float(np.clip((self.current_volume / (self.vad_threshold * 2.5)) * 100.0, 0.0, 100.0))

        return {
            "volume_rms": round(self.current_volume, 4),
            "volume_pct": round(norm_vol, 1),
            "is_speaking": self.is_speaking,
            "speaking_duration_sec": round(self.total_speaking_seconds, 1),
            "speaking_ratio_pct": speaking_ratio,
            "speech_bouts": self.speech_bouts_count,
            "hardware_active": not self.simulated_mode
        }
