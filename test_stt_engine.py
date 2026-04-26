import os
import queue
import time
import unittest

from audio_capture import MockAudioSource
from stt_engine import TranscriptSegment, TranscriptionManager


class TranscriptionManagerTests(unittest.TestCase):
    def test_mock_mode_is_enabled_from_env(self):
        original = os.environ.get("USE_MOCK_NEMO")
        os.environ["USE_MOCK_NEMO"] = "true"
        try:
            manager = TranscriptionManager()
            self.assertEqual(manager.mode, "mock")
        finally:
            if original is None:
                os.environ.pop("USE_MOCK_NEMO", None)
            else:
                os.environ["USE_MOCK_NEMO"] = original

    def test_mock_stream_produces_segments(self):
        manager = TranscriptionManager(use_mock=True)
        result_queue = queue.Queue()
        audio_source = MockAudioSource()

        manager.start_stream(audio_source, result_queue)
        deadline = time.time() + 6.0
        collected = []

        try:
            while time.time() < deadline and not collected:
                try:
                    collected.append(result_queue.get(timeout=0.5))
                except queue.Empty:
                    continue
        finally:
            manager.stop_stream()

        self.assertTrue(collected, "Expected at least one transcript segment in mock mode.")
        self.assertIsInstance(collected[0], TranscriptSegment)
        self.assertTrue(collected[0].text)


if __name__ == "__main__":
    unittest.main()
