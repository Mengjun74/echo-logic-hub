"""
Echo-Logic Hub — NeMo STT Engine
=================================
Modularized class for handling NVIDIA Parakeet ASR and Sortformer
diarization. Includes a mock engine for testing without GPU/NeMo.
"""

import os
import uuid
import time
import queue
import logging
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A single segment of transcribed speech with speaker attribution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    speaker_id: str = "Speaker 01"
    timestamp: str = "00:00:00"
    text: str = ""
    is_final: bool = True
    start_seconds: float = 0.0


@dataclass
class DiarSegment:
    """A speaker diarization segment with time boundaries."""
    speaker_id: str
    start_time: float
    end_time: float


class STTEngine(ABC):
    """Abstract interface for speech-to-text engines."""

    @abstractmethod
    def initialize(self) -> None:
        """Load models and prepare for inference."""

    @abstractmethod
    def process_stream(self, audio_queue, result_queue, stop_event) -> None:
        """Main processing loop consuming audio and producing segments."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release all model resources."""


class NeMoSTTEngine(STTEngine):
    """Production engine using NVIDIA Parakeet RNNT + Sortformer."""

    def __init__(self, asr_model_name="nvidia/parakeet-rnnt-1.1b",
                 diar_model_name="nvidia/diar_sortformer_4spk-v1",
                 sample_rate=16000, diar_buffer_seconds=15.0):
        self.asr_model_name = asr_model_name
        self.diar_model_name = diar_model_name
        self.sample_rate = sample_rate
        self.diar_buffer_seconds = diar_buffer_seconds
        self._asr_model = None
        self._diar_model = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        try:
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError:
            raise RuntimeError(
                "NeMo toolkit not installed. pip install nemo_toolkit[asr] "
                "or set USE_MOCK_NEMO=true"
            )
        logger.info(f"Loading ASR model: {self.asr_model_name}")
        local_path = os.getenv("LOCAL_STT_MODEL_PATH", "").strip()
        if local_path and os.path.exists(local_path):
            self._asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(local_path)
        else:
            self._asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name=self.asr_model_name)
        self._asr_model.eval()
        logger.info(f"Loading Diarization model: {self.diar_model_name}")
        self._diar_model = SortformerEncLabelModel.from_pretrained(self.diar_model_name)
        self._diar_model.eval()
        self._initialized = True
        logger.info("✅ NeMo engine initialized.")

    def _transcribe_buffer(self, audio_buffer):
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
            sf.write(tmp, audio_buffer, self.sample_rate)
        try:
            output = self._asr_model.transcribe([tmp])
            return output[0].text if hasattr(output[0], "text") else str(output[0])
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _diarize_buffer(self, audio_buffer):
        import soundfile as sf
        import json
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wav_path = wf.name
            sf.write(wav_path, audio_buffer, self.sample_rate)
        manifest = wav_path.replace(".wav", "_manifest.json")
        try:
            with open(manifest, "w") as mf:
                json.dump({"audio_filepath": wav_path, "offset": 0,
                           "duration": len(audio_buffer) / self.sample_rate}, mf)
            predicted = self._diar_model.diarize(audio=manifest, batch_size=1)
            segs = []
            if predicted:
                for seg in predicted:
                    segs.append(DiarSegment(
                        speaker_id=f"Speaker {int(seg.get('speaker', 0)) + 1:02d}",
                        start_time=seg.get("start", 0.0),
                        end_time=seg.get("end", 0.0)))
            return segs
        finally:
            for p in [wav_path, manifest]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    @staticmethod
    def _fmt_ts(seconds):
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def process_stream(self, audio_queue, result_queue, stop_event):
        if not self._initialized:
            self.initialize()
        acc, total, seg_n = [], 0, 0
        buf_thresh = int(self.diar_buffer_seconds * self.sample_rate)
        logger.info("🎙️ NeMo stream processing started.")
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                continue
            acc.append(chunk)
            total += len(chunk)
            if total >= buf_thresh:
                full = np.concatenate(acc)
                text = self._transcribe_buffer(full)
                diar = self._diarize_buffer(full)
                speaker = "Speaker 01"
                if diar:
                    times = {}
                    for d in diar:
                        times[d.speaker_id] = times.get(d.speaker_id, 0) + d.end_time - d.start_time
                    speaker = max(times, key=times.get)
                elapsed = total / self.sample_rate
                seg_n += 1
                seg = TranscriptSegment(speaker_id=speaker, timestamp=self._fmt_ts(elapsed),
                                        text=text.strip(), start_seconds=elapsed - self.diar_buffer_seconds)
                if seg.text:
                    result_queue.put(seg)
                overlap = int(2.0 * self.sample_rate)
                if len(full) > overlap:
                    acc, total = [full[-overlap:]], overlap
                else:
                    acc, total = [], 0
        logger.info("🎙️ NeMo stream processing stopped.")

    def shutdown(self):
        self._asr_model = self._diar_model = None
        self._initialized = False


_MOCK_TRANSCRIPTS = [
    "I think we should focus on the deployment pipeline first before adding any new features.",
    "That makes sense. The CI/CD pipeline has been a bottleneck for weeks now.",
    "Has anyone looked into the latency issues on the inference endpoint?",
    "Yes, I ran some benchmarks yesterday. The P95 latency is around 340 milliseconds.",
    "We could try batching the requests. That helped with the recommendation service.",
    "Good idea. Let me pull up the architecture diagram so we can trace the hot path.",
    "I also noticed the model checkpoint loads from cold storage every container restart.",
    "That's definitely a problem. We should cache the model weights in the container image.",
    "What about the memory footprint? The 1.1B parameter model needs at least 4 gigs of VRAM.",
    "We could quantize to INT8. The accuracy drop is less than half a percent on our eval suite.",
    "Let's create a JIRA ticket for the quantization experiment and assign it to ML ops.",
    "Agreed. I'll also set up a dashboard to monitor the latency improvements in real time.",
    "One more thing. The diarization accuracy drops significantly with more than three speakers.",
    "We might need to fine-tune the Sortformer model on our meeting recordings dataset.",
    "I can start collecting labeled data from the last month of team syncs if that helps.",
    "Perfect. Let's regroup on Friday with the benchmark results and a revised timeline.",
]


class MockNeMoEngine(STTEngine):
    """Mock engine producing realistic fake transcript segments."""

    def __init__(self, segment_interval=3.0, num_speakers=3):
        self.segment_interval = segment_interval
        self.num_speakers = min(num_speakers, 4)
        self._initialized = False

    def initialize(self):
        self._initialized = True
        logger.info("✅ Mock NeMo engine initialized.")

    def process_stream(self, audio_queue, result_queue, stop_event):
        self.initialize()
        idx = 0
        t0 = time.time()
        speakers = [f"Speaker {i + 1:02d}" for i in range(self.num_speakers)]
        logger.info("🔊 Mock NeMo stream started.")
        while not stop_event.is_set():
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break
            elapsed = time.time() - t0
            h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
            seg = TranscriptSegment(
                speaker_id=speakers[idx % len(speakers)],
                timestamp=f"{h:02d}:{m:02d}:{s:02d}",
                text=_MOCK_TRANSCRIPTS[idx % len(_MOCK_TRANSCRIPTS)],
                start_seconds=elapsed)
            result_queue.put(seg)
            logger.info(f"[Mock {idx + 1}] {seg.speaker_id} @ {seg.timestamp}: {seg.text[:60]}...")
            idx += 1
            stop_event.wait(timeout=self.segment_interval)
        logger.info("🔊 Mock NeMo stream stopped.")

    def shutdown(self):
        self._initialized = False


def create_stt_engine():
    """Factory: MockNeMoEngine when USE_MOCK_NEMO=true, else NeMoSTTEngine."""
    if os.getenv("USE_MOCK_NEMO", "true").lower() == "true":
        logger.info("Using MockNeMoEngine")
        return MockNeMoEngine()
    logger.info("Using NeMoSTTEngine")
    return NeMoSTTEngine()
