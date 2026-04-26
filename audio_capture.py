"""
Echo-Logic Hub — Audio Capture Module
=====================================
Provides audio input sources for the STT pipeline.
Supports live microphone capture via PyAudio and a mock source for testing.
"""

import os
import time
import queue
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class AudioConfig:
    """Audio capture parameters."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_frames: int = 4096        # ~0.256s per chunk at 16kHz
    dtype: str = "int16"


# ═══════════════════════════════════════════════════════════════
# Abstract Base
# ═══════════════════════════════════════════════════════════════

class AudioSource(ABC):
    """Abstract interface for audio input sources."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self._active = False
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

    @property
    def is_active(self) -> bool:
        return self._active

    @abstractmethod
    def start(self) -> None:
        """Begin capturing audio."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing audio and release resources."""

    def get_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """Retrieve the next audio chunk from the queue.

        Returns None if no data is available within the timeout.
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ═══════════════════════════════════════════════════════════════
# Live Microphone Source (PyAudio)
# ═══════════════════════════════════════════════════════════════

class LiveAudioSource(AudioSource):
    """Captures audio from the system microphone using PyAudio.

    Uses a non-blocking callback stream to push fixed-size chunks
    into an internal queue for downstream consumption.
    """

    def __init__(self, config: AudioConfig | None = None):
        super().__init__(config)
        self._stream = None
        self._pa = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio non-blocking callback — pushes raw audio to the queue."""
        import pyaudio

        if status:
            logger.warning(f"PyAudio status flag: {status}")

        audio_chunk = np.frombuffer(in_data, dtype=np.int16).copy()

        try:
            self._audio_queue.put_nowait(audio_chunk)
        except queue.Full:
            # Drop oldest chunk to prevent backpressure
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            self._audio_queue.put_nowait(audio_chunk)

        return (None, pyaudio.paContinue)

    def start(self) -> None:
        """Open the microphone stream."""
        if self._active:
            logger.warning("LiveAudioSource is already active.")
            return

        try:
            import pyaudio
        except ImportError:
            raise RuntimeError(
                "PyAudio is not installed. Install it with: pip install pyaudio\n"
                "On Windows you may need: pip install pipwin && pipwin install pyaudio"
            )

        self._pa = pyaudio.PyAudio()

        FORMAT_MAP = {"int16": pyaudio.paInt16, "float32": pyaudio.paFloat32}
        pa_format = FORMAT_MAP.get(self.config.dtype, pyaudio.paInt16)

        self._stream = self._pa.open(
            format=pa_format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_frames,
            stream_callback=self._audio_callback,
        )
        self._stream.start_stream()
        self._active = True
        logger.info(
            f"🎤 Live audio capture started "
            f"(rate={self.config.sample_rate}, chunk={self.config.chunk_frames})"
        )

    def stop(self) -> None:
        """Close the microphone stream and release PyAudio resources."""
        self._active = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        logger.info("🎤 Live audio capture stopped.")


# ═══════════════════════════════════════════════════════════════
# Mock Audio Source (Testing)
# ═══════════════════════════════════════════════════════════════

class MockAudioSource(AudioSource):
    """Simulates incoming audio data for UI testing without a microphone.

    Generates synthetic speech-like patterns with configurable speaker
    changes to exercise the full UI pipeline.
    """

    def __init__(
        self,
        config: AudioConfig | None = None,
        chunk_interval: float = 0.25,
    ):
        super().__init__(config)
        self._chunk_interval = chunk_interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _generate_chunk(self, t: float) -> np.ndarray:
        """Create a synthetic audio chunk resembling speech cadence.

        Uses a combination of sine waves with amplitude modulation
        to simulate voiced segments.
        """
        n_samples = self.config.chunk_frames
        sr = self.config.sample_rate

        time_axis = np.linspace(t, t + n_samples / sr, n_samples, endpoint=False)

        # Fundamental + harmonics simulating voiced speech
        f0 = 120 + 30 * np.sin(2 * np.pi * 0.3 * t)  # Varying pitch
        signal = (
            0.5 * np.sin(2 * np.pi * f0 * time_axis)
            + 0.3 * np.sin(2 * np.pi * 2 * f0 * time_axis)
            + 0.1 * np.sin(2 * np.pi * 3 * f0 * time_axis)
        )

        # Amplitude envelope simulating speech rhythm
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * time_axis)
        signal *= envelope

        # Normalize to int16 range at ~50% volume
        signal = (signal * 16000).astype(np.int16)
        return signal

    def _producer_loop(self) -> None:
        """Background thread that generates and enqueues audio chunks."""
        t = 0.0
        dt = self.config.chunk_frames / self.config.sample_rate

        logger.info("🔊 Mock audio producer started.")

        while not self._stop_event.is_set():
            chunk = self._generate_chunk(t)
            try:
                self._audio_queue.put(chunk, timeout=0.5)
            except queue.Full:
                pass  # Drop if consumer can't keep up
            t += dt
            time.sleep(self._chunk_interval)

        logger.info("🔊 Mock audio producer stopped.")

    def start(self) -> None:
        """Start the mock audio producer thread."""
        if self._active:
            logger.warning("MockAudioSource is already active.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._producer_loop, name="MockAudioProducer", daemon=True
        )
        self._thread.start()
        self._active = True
        logger.info("🔊 Mock audio source started.")

    def stop(self) -> None:
        """Stop the mock audio producer thread."""
        self._active = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        # Drain the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("🔊 Mock audio source stopped.")


# ═══════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════

def create_audio_source(config: AudioConfig | None = None) -> AudioSource:
    """Create the appropriate audio source based on environment config.

    Returns MockAudioSource when USE_MOCK_AUDIO=true, otherwise LiveAudioSource.
    """
    use_mock = os.getenv("USE_MOCK_AUDIO", "true").lower() == "true"

    if use_mock:
        logger.info("Using MockAudioSource (USE_MOCK_AUDIO=true)")
        return MockAudioSource(config)
    else:
        logger.info("Using LiveAudioSource (USE_MOCK_AUDIO=false)")
        return LiveAudioSource(config)
