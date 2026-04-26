import os
import time
import wave
import queue
import logging
import threading
from datetime import datetime
from collections import deque
import numpy as np
import torch
from dataclasses import dataclass
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import webrtcvad
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

# ═══════════════════════════════════════════════════════════════
# Mock Stream for Testing
# ═══════════════════════════════════════════════════════════════

class MockWavStream:
    """Reads a local .wav file and simulates a real-time microphone stream."""
    
    def __init__(self, wav_path: str, chunk_size: int = 480):
        self.wav_path = wav_path
        self.chunk_size = chunk_size  # For 16kHz, 480 frames = 30ms
        self.wf = None
        self._active = False
        self._queue = queue.Queue()
        self._thread = None
        
    def start(self):
        if not os.path.exists(self.wav_path):
            raise FileNotFoundError(f"Wav file not found: {self.wav_path}")
        
        self.wf = wave.open(self.wav_path, 'rb')
        if self.wf.getframerate() != 16000 or self.wf.getnchannels() != 1:
            raise ValueError("Wav file must be 16kHz mono")
            
        self._active = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        self._active = False
        if self._thread:
            self._thread.join()
        if self.wf:
            self.wf.close()
            
    def _stream_loop(self):
        # Frame duration in seconds
        dt = self.chunk_size / 16000.0
        while self._active:
            data = self.wf.readframes(self.chunk_size)
            if not data:
                break
            
            # Pad if last chunk is too small
            if len(data) < self.chunk_size * 2:  # 2 bytes per frame (int16)
                data += b'\x00' * (self.chunk_size * 2 - len(data))
                
            chunk = np.frombuffer(data, dtype=np.int16)
            self._queue.put(chunk)
            time.sleep(dt) # Simulate real-time
            
    def get_chunk(self, timeout=0.1) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

# ═══════════════════════════════════════════════════════════════
# Core Transcription Engine
# ═══════════════════════════════════════════════════════════════

class TranscriptionManager:
    """Manages audio capture, VAD, STT (faster-whisper), and Diarization (pyannote)."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.vad = webrtcvad.Vad(3) # Aggressiveness level 3
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        logger.info(f"Initializing STT Engine on {self.device.upper()}...")
        
        # Load Faster-Whisper
        self.whisper = WhisperModel(
            "large-v3-turbo", 
            device=self.device, 
            compute_type=self.compute_type,
            local_files_only=False
        )
        
        # Load Pyannote Diarization Pipeline
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not found in .env! Pyannote may fail to download.")
        
        try:
            self.diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            # fallback to `token` if `use_auth_token` is rejected (depends on pyannote version)
        except TypeError:
            try:
                self.diarization = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
            except Exception as e:
                logger.error(f"Failed to load Pyannote Pipeline with 'token': {e}")
                self.diarization = None
        except Exception as e:
            logger.error(f"Failed to load Pyannote Pipeline: {e}")
            self.diarization = None
            
        if getattr(self, 'diarization', None) and self.device == "cuda":
            self.diarization.to(torch.device("cuda"))
            
        self._is_streaming = False
        self._thread = None
        
    def start_stream(self, audio_source, result_queue: queue.Queue):
        if self._is_streaming:
            return
        
        self._is_streaming = True
        self._thread = threading.Thread(
            target=self._processing_loop, 
            args=(audio_source, result_queue),
            daemon=True
        )
        self._thread.start()
        
    def stop_stream(self):
        self._is_streaming = False
        if self._thread:
            self._thread.join()
            
    def _processing_loop(self, audio_source, result_queue: queue.Queue):
        audio_source.start()
        
        buffer = []
        silence_frames = 0
        max_silence_frames = int(1.5 * 16000 / 480) # 1.5 seconds of silence = sentence boundary
        min_speech_frames = int(0.5 * 16000 / 480) # 0.5 seconds minimum to process
        
        chunk_size = 480 # 30ms at 16kHz
        # We need a chunk_buffer to accumulate exactly chunk_size from audio_source
        # Because webrtcvad needs exactly 10, 20, or 30ms chunks
        
        raw_audio_queue = deque()
        
        logger.info("STT Engine processing loop started.")
        
        while self._is_streaming:
            chunk = audio_source.get_chunk(timeout=0.1)
            
            if chunk is not None:
                raw_audio_queue.extend(chunk.tolist())
            elif not getattr(audio_source, '_active', True):
                # Stream has stopped and no more chunks
                break
            
            # Process exactly 30ms (480 samples) chunks
            while len(raw_audio_queue) >= chunk_size:
                frame_data = [raw_audio_queue.popleft() for _ in range(chunk_size)]
                frame = np.array(frame_data, dtype=np.int16)
                
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                
                if is_speech:
                    silence_frames = 0
                    buffer.extend(frame_data)
                else:
                    silence_frames += 1
                    if len(buffer) > 0:
                        buffer.extend(frame_data)
                
                # Check for Sentence Boundary
                if silence_frames >= max_silence_frames:
                    if len(buffer) > min_speech_frames * chunk_size:
                        # Process buffer
                        audio_np = np.array(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        self._process_segment(audio_np, result_queue)
                    
                    buffer.clear()
                    silence_frames = 0
                    
        # Process any remaining audio
        if len(buffer) > min_speech_frames * chunk_size:
            audio_np = np.array(buffer, dtype=np.int16).astype(np.float32) / 32768.0
            self._process_segment(audio_np, result_queue)
            
        audio_source.stop()
        logger.info("STT Engine processing loop stopped.")
        
    def _process_segment(self, audio_np: np.ndarray, result_queue: queue.Queue):
        """Runs Whisper and Pyannote on the audio segment and aligns them."""
        # 1. Run Whisper
        segments, info = self.whisper.transcribe(audio_np, beam_size=5)
        whisper_segments = list(segments)
        
        if not whisper_segments:
            return
            
        # 2. Run Pyannote
        # Pyannote expects a specific input format: {"waveform": Tensor(1, N), "sample_rate": 16000}
        diar_result = None
        if self.diarization:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
            try:
                diar_result = self.diarization({"waveform": waveform, "sample_rate": self.sample_rate})
            except Exception as e:
                logger.error(f"Diarization error: {e}")
                
        # 3. Align
        for w_seg in whisper_segments:
            w_start = w_seg.start
            w_end = w_seg.end
            text = w_seg.text.strip()
            
            if not text:
                continue
                
            speaker_id = "Speaker 01" # Default
            
            if diar_result:
                # Find speaker with max overlap
                max_overlap = 0.0
                for turn, _, spk in diar_result.itertracks(yield_label=True):
                    overlap = max(0, min(w_end, turn.end) - max(w_start, turn.start))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        speaker_id = spk.replace("SPEAKER_", "Speaker ")
                        
            # Output
            now = datetime.now().strftime("%H:%M:%S")
            segment = TranscriptSegment(
                id=f"seg_{int(time.time()*1000)}_{w_start}",
                speaker_id=speaker_id,
                timestamp=now,
                text=text,
                start_seconds=w_start,
                end_seconds=w_end,
                is_final=True
            )
            result_queue.put(segment)
