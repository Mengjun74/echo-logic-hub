import logging
import os
import queue
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("stt_engine")


@dataclass
class TranscriptSegment:
    id: str
    speaker_id: str
    timestamp: str
    text: str
    start_seconds: float
    end_seconds: float
    is_final: bool


class MockWavStream:
    """Reads a local wav file and simulates a real-time microphone stream."""

    def __init__(self, wav_path: str, chunk_size: int = 480):
        self.wav_path = wav_path
        self.chunk_size = chunk_size
        self.wf = None
        self._active = False
        self._queue = queue.Queue()
        self._thread = None

    def start(self):
        if not os.path.exists(self.wav_path):
            raise FileNotFoundError(f"Wav file not found: {self.wav_path}")

        self.wf = wave.open(self.wav_path, "rb")
        if self.wf.getframerate() != 16000 or self.wf.getnchannels() != 1:
            raise ValueError("Wav file must be 16kHz mono")

        self._active = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._active = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self.wf:
            self.wf.close()
            self.wf = None

    def _stream_loop(self):
        dt = self.chunk_size / 16000.0
        while self._active:
            data = self.wf.readframes(self.chunk_size)
            if not data:
                break

            if len(data) < self.chunk_size * 2:
                data += b"\x00" * (self.chunk_size * 2 - len(data))

            chunk = np.frombuffer(data, dtype=np.int16)
            self._queue.put(chunk)
            time.sleep(dt)

        self._active = False

    def get_chunk(self, timeout=0.1) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class TranscriptionManager:
    """Manages audio capture and transcription with mock fallback support."""

    MOCK_SEGMENTS = (
        "Let's review the rollout plan and call out any blockers.",
        "We should tighten latency before the next demo and keep the scope focused.",
        "Action item: document the test path for mock mode and production mode.",
        "Open question: do we want diarization enabled by default on CPU-only machines?",
    )

    def __init__(self, use_mock: bool | None = None):
        self.sample_rate = 16000
        self._is_streaming = False
        self._thread = None
        self._segment_counter = 0
        self._mock_segment_seconds = 2.4
        self._mode = "mock"

        if use_mock is None:
            use_mock = os.getenv("USE_MOCK_NEMO", "true").lower() == "true"

        self.whisper = None
        self.diarization = None
        self.vad = None
        self.torch = None
        self.device = "cpu"
        self.compute_type = "int8"

        if use_mock:
            logger.info("Using mock transcription mode.")
            return

        self._initialize_real_backend()

    @property
    def mode(self) -> str:
        return self._mode

    def _initialize_real_backend(self) -> None:
        try:
            import torch
            from faster_whisper import WhisperModel
            from pyannote.audio import Pipeline
            import webrtcvad
        except Exception as exc:
            logger.warning("Falling back to mock STT because dependencies are unavailable: %s", exc)
            self._mode = "mock"
            return

        self.torch = torch
        self.vad = webrtcvad.Vad(3)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        logger.info("Initializing STT engine on %s", self.device.upper())

        try:
            self.whisper = WhisperModel(
                "large-v3-turbo",
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=False,
            )
        except Exception as exc:
            logger.warning("Whisper initialization failed, using mock STT instead: %s", exc)
            self._mode = "mock"
            self.whisper = None
            return

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN is not configured. Speaker diarization will stay disabled.")

        try:
            self.diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        except TypeError:
            try:
                self.diarization = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token,
                )
            except Exception as exc:
                logger.warning("Pyannote initialization failed: %s", exc)
                self.diarization = None
        except Exception as exc:
            logger.warning("Pyannote initialization failed: %s", exc)
            self.diarization = None

        if self.diarization is not None and self.device == "cuda":
            try:
                self.diarization.to(torch.device("cuda"))
            except Exception as exc:
                logger.warning("Failed moving diarization pipeline to CUDA: %s", exc)
                self.diarization = None

        self._mode = "real"

    def start_stream(self, audio_source, result_queue: queue.Queue):
        if self._is_streaming:
            return

        self._is_streaming = True
        self._thread = threading.Thread(
            target=self._processing_loop,
            args=(audio_source, result_queue),
            daemon=True,
        )
        self._thread.start()

    def stop_stream(self):
        self._is_streaming = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _processing_loop(self, audio_source, result_queue: queue.Queue):
        audio_source.start()
        logger.info("STT processing loop started in %s mode.", self._mode)

        try:
            if self._mode == "real":
                self._run_real_loop(audio_source, result_queue)
            else:
                self._run_mock_loop(audio_source, result_queue)
        finally:
            audio_source.stop()
            logger.info("STT processing loop stopped.")

    def _run_mock_loop(self, audio_source, result_queue: queue.Queue):
        buffered_samples = 0
        speech_samples = 0
        segment_start_seconds = 0.0
        elapsed_seconds = 0.0

        while self._is_streaming:
            chunk = audio_source.get_chunk(timeout=0.1)

            if chunk is None:
                if not getattr(audio_source, "_active", True):
                    break
                continue

            chunk_seconds = len(chunk) / self.sample_rate
            buffered_samples += len(chunk)
            elapsed_seconds += chunk_seconds

            amplitude = float(np.abs(chunk).mean()) if len(chunk) else 0.0
            if amplitude > 250:
                speech_samples += len(chunk)

            buffered_seconds = buffered_samples / self.sample_rate
            speech_seconds = speech_samples / self.sample_rate
            should_emit = buffered_seconds >= self._mock_segment_seconds and speech_seconds >= 0.8

            if should_emit:
                self._emit_mock_segment(
                    result_queue=result_queue,
                    start_seconds=segment_start_seconds,
                    end_seconds=elapsed_seconds,
                )
                segment_start_seconds = elapsed_seconds
                buffered_samples = 0
                speech_samples = 0

        remaining_seconds = buffered_samples / self.sample_rate
        speech_seconds = speech_samples / self.sample_rate
        if remaining_seconds >= 0.8 and speech_seconds >= 0.4:
            self._emit_mock_segment(
                result_queue=result_queue,
                start_seconds=segment_start_seconds,
                end_seconds=elapsed_seconds,
            )

    def _emit_mock_segment(
        self,
        result_queue: queue.Queue,
        start_seconds: float,
        end_seconds: float,
    ) -> None:
        text = self.MOCK_SEGMENTS[self._segment_counter % len(self.MOCK_SEGMENTS)]
        speaker_number = (self._segment_counter % 3) + 1
        speaker_id = f"Speaker {speaker_number:02d}"
        self._segment_counter += 1
        result_queue.put(
            TranscriptSegment(
                id=f"seg_{int(time.time() * 1000)}_{self._segment_counter}",
                speaker_id=speaker_id,
                timestamp=datetime.now().strftime("%H:%M:%S"),
                text=text,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                is_final=True,
            )
        )

    def _run_real_loop(self, audio_source, result_queue: queue.Queue):
        buffer = []
        silence_frames = 0
        max_silence_frames = int(1.5 * self.sample_rate / 480)
        min_speech_frames = int(0.5 * self.sample_rate / 480)
        chunk_size = 480
        raw_audio_queue = deque()

        while self._is_streaming:
            chunk = audio_source.get_chunk(timeout=0.1)

            if chunk is not None:
                raw_audio_queue.extend(chunk.tolist())
            elif not getattr(audio_source, "_active", True):
                break

            while len(raw_audio_queue) >= chunk_size:
                frame_data = [raw_audio_queue.popleft() for _ in range(chunk_size)]
                frame = np.array(frame_data, dtype=np.int16)

                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)

                if is_speech:
                    silence_frames = 0
                    buffer.extend(frame_data)
                else:
                    silence_frames += 1
                    if buffer:
                        buffer.extend(frame_data)

                if silence_frames >= max_silence_frames:
                    if len(buffer) > min_speech_frames * chunk_size:
                        audio_np = np.array(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        self._process_real_segment(audio_np, result_queue)

                    buffer.clear()
                    silence_frames = 0

        if len(buffer) > min_speech_frames * chunk_size:
            audio_np = np.array(buffer, dtype=np.int16).astype(np.float32) / 32768.0
            self._process_real_segment(audio_np, result_queue)

    def _process_real_segment(self, audio_np: np.ndarray, result_queue: queue.Queue):
        segments, _ = self.whisper.transcribe(audio_np, beam_size=5)
        whisper_segments = list(segments)
        if not whisper_segments:
            return

        diar_result = None
        if self.diarization is not None:
            waveform = self.torch.from_numpy(audio_np).unsqueeze(0)
            try:
                diar_result = self.diarization(
                    {"waveform": waveform, "sample_rate": self.sample_rate}
                )
            except Exception as exc:
                logger.warning("Diarization error: %s", exc)

        for whisper_segment in whisper_segments:
            text = whisper_segment.text.strip()
            if not text:
                continue

            speaker_id = "Speaker 01"
            if diar_result is not None:
                speaker_id = self._resolve_speaker_id(
                    diar_result=diar_result,
                    start_seconds=whisper_segment.start,
                    end_seconds=whisper_segment.end,
                )

            result_queue.put(
                TranscriptSegment(
                    id=f"seg_{int(time.time() * 1000)}_{whisper_segment.start}",
                    speaker_id=speaker_id,
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    text=text,
                    start_seconds=whisper_segment.start,
                    end_seconds=whisper_segment.end,
                    is_final=True,
                )
            )

    @staticmethod
    def _resolve_speaker_id(diar_result, start_seconds: float, end_seconds: float) -> str:
        speaker_id = "Speaker 01"
        max_overlap = 0.0

        for turn, _, speaker_label in diar_result.itertracks(yield_label=True):
            overlap = max(0.0, min(end_seconds, turn.end) - max(start_seconds, turn.start))
            if overlap > max_overlap:
                max_overlap = overlap
                speaker_id = speaker_label.replace("SPEAKER_", "Speaker ")

        return speaker_id
