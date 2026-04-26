"""
Echo-Logic Hub — Gemini Client
================================
Wrapper around the google-generativeai SDK for sending context
to Google Gemini 1.5 Pro. Includes a mock client for testing.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


class GeminiClientBase(ABC):
    """Abstract interface for Gemini LLM clients."""

    @abstractmethod
    def execute(self, system_prompt: str, context_segments: List[str]) -> str:
        """Send system prompt + selected segments to the LLM.

        Args:
            system_prompt: The instruction set / system prompt.
            context_segments: Ordered list of transcript segment strings.

        Returns:
            The LLM's response text.
        """


class GeminiClient(GeminiClientBase):
    """Production client using google-generativeai SDK."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._model = None
        self._configure()

    def _configure(self) -> None:
        """Initialize the Gemini SDK with the API key."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError(
                "google-generativeai is not installed. "
                "Install with: pip install google-generativeai"
            )

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file.\n"
                "Get a key at: https://aistudio.google.com/app/apikey"
            )

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=self.model_name)
        logger.info(f"✅ Gemini client configured (model={self.model_name})")

    def execute(self, system_prompt: str, context_segments: List[str]) -> str:
        """Build payload, log it, send to Gemini, and return response."""

        # Build the concatenated payload
        segments_block = "\n\n".join(context_segments)
        full_payload = (
            f"{system_prompt}\n\n"
            f"{'=' * 60}\n"
            f"TRANSCRIPT CONTEXT ({len(context_segments)} segments)\n"
            f"{'=' * 60}\n\n"
            f"{segments_block}"
        )

        # MANDATORY: Log the payload for validation
        print("\n" + "=" * 70)
        print("[PAYLOAD LOG] Sending to Gemini")
        print("=" * 70)
        print(f"[PAYLOAD LOG] System Prompt: {system_prompt[:120]}...")
        print(f"[PAYLOAD LOG] Segments ({len(context_segments)} selected):")
        for seg_text in context_segments:
            print(f"  -> {seg_text[:100]}...")
        print(f"[PAYLOAD LOG] Total payload length: {len(full_payload):,} characters")
        print("=" * 70 + "\n")

        try:
            response = self._model.generate_content(full_payload)
            result = response.text
            logger.info(f"Gemini response received ({len(result)} chars)")
            return result
        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"


_MOCK_RESPONSES = [
    """## Analysis Summary

Based on the provided transcript segments, here are the key takeaways:

### Action Items
1. **Deployment Pipeline** — Prioritize CI/CD improvements before feature work
2. **Latency Investigation** — P95 at 340ms exceeds target; request batching recommended
3. **Model Caching** — Embed weights in container image to avoid cold-storage loads
4. **INT8 Quantization** — Create JIRA ticket for the ML ops team

### Risk Assessment
- **High Risk:** Container restart causing model reload latency spikes
- **Medium Risk:** Diarization accuracy degradation beyond 3 speakers
- **Low Risk:** Memory footprint (manageable with quantization)

### Recommended Next Steps
Schedule a follow-up meeting for Friday to review benchmark results and revised timeline.
The team should focus on the batching implementation as a quick win for latency reduction.""",

    """## Meeting Insights

### Participants Detected
Three distinct speakers were identified in the conversation.

### Discussion Topics
1. **Infrastructure** — CI/CD pipeline bottleneck, deployment strategy
2. **Performance** — Inference latency, model loading times, memory optimization
3. **ML Operations** — Model quantization, diarization fine-tuning, data collection

### Sentiment Analysis
The overall tone is **constructive and action-oriented**. Team members are proactively
identifying issues and proposing concrete solutions.

### Key Decision
The team agreed to pursue INT8 quantization as the primary optimization strategy,
with an expected accuracy impact of less than 0.5%.""",

    """## Executive Briefing

### Context
Engineering team sync discussing production system optimization.

### Critical Findings
| Issue | Severity | Owner | ETA |
|-------|----------|-------|-----|
| CI/CD bottleneck | High | DevOps | This sprint |
| P95 latency (340ms) | High | Backend | 2 weeks |
| Cold storage model load | Medium | ML Ops | 1 week |
| Speaker diarization accuracy | Low | ML Research | TBD |

### Budget Impact
INT8 quantization could reduce GPU costs by ~40% while maintaining 99.5% accuracy.

### Recommendation
Proceed with the proposed optimization roadmap. Regroup Friday for progress review.""",
]


class MockGeminiClient(GeminiClientBase):
    """Mock client returning canned responses for testing."""

    def __init__(self):
        self._response_idx = 0
        logger.info("✅ Mock Gemini client initialized.")

    def execute(self, system_prompt: str, context_segments: List[str]) -> str:
        """Return a canned response and log the payload."""

        segments_block = "\n\n".join(context_segments)
        full_payload = f"{system_prompt}\n\n---\n\n{segments_block}"

        # MANDATORY: Log the payload for validation
        print("\n" + "=" * 70)
        print("[PAYLOAD LOG] Mock Gemini -- Payload Validation")
        print("=" * 70)
        print(f"[PAYLOAD LOG] System Prompt: {system_prompt[:120]}...")
        print(f"[PAYLOAD LOG] Segments ({len(context_segments)} selected):")
        for seg_text in context_segments:
            print(f"  -> {seg_text[:100]}...")
        print(f"[PAYLOAD LOG] Total payload length: {len(full_payload):,} characters")
        print("=" * 70 + "\n")

        response = _MOCK_RESPONSES[self._response_idx % len(_MOCK_RESPONSES)]
        self._response_idx += 1
        logger.info(f"Mock Gemini response #{self._response_idx} returned.")
        return response


def create_gemini_client() -> GeminiClientBase:
    """Factory: MockGeminiClient when USE_MOCK_GEMINI=true, else GeminiClient."""
    if os.getenv("USE_MOCK_GEMINI", "true").lower() == "true":
        logger.info("Using MockGeminiClient")
        return MockGeminiClient()
    logger.info("Using GeminiClient")
    return GeminiClient()
