"""
Echo-Logic Hub — Main Streamlit Application
=============================================
Real-time STT + LLM Orchestration Dashboard.
Run with: streamlit run app.py
"""

import os
import sys
import time
import queue
import logging
import threading
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────
load_dotenv()

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echo-logic-hub")

# ── Local imports ─────────────────────────────────────────────
from audio_capture import create_audio_source
from stt_engine import TranscriptionManager, TranscriptSegment
from gemini_client import create_gemini_client


# ═══════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Echo-Logic Hub",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
# Load Custom CSS
# ═══════════════════════════════════════════════════════════════

def load_css():
    """Inject custom CSS from the assets directory."""
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found: {css_path}")

load_css()


# ═══════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════

DEFAULTS = {
    "transcript_segments": [],
    "selected_segment_ids": set(),
    "chat_history": [],
    "system_prompt": (
        "You are an expert meeting analyst. Analyze the following transcript "
        "segments and provide:\n"
        "1. Key discussion points\n"
        "2. Action items with owners\n"
        "3. Decisions made\n"
        "4. Open questions or risks"
    ),
    "is_streaming": False,
    "audio_source": None,
    "stt_engine": None,
    "gemini_client_instance": None,
    "audio_queue": None,
    "result_queue": None,
    "stop_event": None,
    "processing_thread": None,
    "audio_thread_ref": None,
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════
# Helper: Initialize Clients (Lazy)
# ═══════════════════════════════════════════════════════════════

def get_gemini_client():
    """Lazy-initialize the Gemini client."""
    if st.session_state.gemini_client_instance is None:
        st.session_state.gemini_client_instance = create_gemini_client()
    return st.session_state.gemini_client_instance


# ═══════════════════════════════════════════════════════════════
# Streaming Control
# ═══════════════════════════════════════════════════════════════

def start_streaming():
    """Initialize audio capture + STT processing in background threads."""
    if st.session_state.is_streaming:
        return

    audio_src = create_audio_source()
    stt_eng = TranscriptionManager()
    rq = queue.Queue(maxsize=200)

    # stt_eng handles the internal thread reading from audio_src
    stt_eng.start_stream(audio_src, rq)

    st.session_state.audio_source = audio_src
    st.session_state.stt_engine = stt_eng
    st.session_state.result_queue = rq
    st.session_state.is_streaming = True

    logger.info("▶️ Streaming started.")


def stop_streaming():
    """Signal background threads to stop and clean up."""
    if not st.session_state.is_streaming:
        return

    if st.session_state.stt_engine:
        st.session_state.stt_engine.stop_stream()

    st.session_state.is_streaming = False
    st.session_state.audio_source = None
    st.session_state.stt_engine = None
    st.session_state.result_queue = None

    logger.info("⏹️ Streaming stopped.")


def drain_result_queue():
    """Pull all available segments from the result queue into session state."""
    rq = st.session_state.result_queue
    if rq is None:
        return 0
    count = 0
    while True:
        try:
            segment = rq.get_nowait()
            st.session_state.transcript_segments.append(segment)
            count += 1
        except queue.Empty:
            break
    return count


def clear_session():
    """Reset all transcript and chat data."""
    stop_streaming()
    st.session_state.transcript_segments = []
    st.session_state.selected_segment_ids = set()
    st.session_state.chat_history = []
    # Clear checkbox states
    for key in list(st.session_state.keys()):
        if key.startswith("cb_"):
            del st.session_state[key]
    logger.info("🗑️ Session cleared.")


# ═══════════════════════════════════════════════════════════════
# Helper: Render Speech Card
# ═══════════════════════════════════════════════════════════════

SPEAKER_COLORS = {
    "Speaker 01": "speaker-01",
    "Speaker 02": "speaker-02",
    "Speaker 03": "speaker-03",
    "Speaker 04": "speaker-04",
}


def render_speech_card(segment: TranscriptSegment, index: int):
    """Render a single speech card with checkbox for selection."""
    spk_class = SPEAKER_COLORS.get(segment.speaker_id, "speaker-01")
    is_selected = segment.id in st.session_state.selected_segment_ids
    selected_class = "speech-card-selected" if is_selected else ""

    # Checkbox for selection
    cb_key = f"cb_{segment.id}"
    st.checkbox(
        label=f"**{segment.speaker_id}**  ·  `{segment.timestamp}`",
        value=is_selected,
        key=cb_key,
    )

    # Note: selection state is synced globally before sidebar rendering.

    # Render the card content with custom HTML
    st.markdown(
        f"""<div class="speech-card {spk_class} {selected_class}">
            <div class="speech-card-header">
                <span class="speaker-badge {spk_class}">{segment.speaker_id}</span>
                <span class="speech-timestamp">{segment.timestamp}</span>
            </div>
            <p class="speech-text">{segment.text}</p>
        </div>""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# Helper: Render Chat Entry
# ═══════════════════════════════════════════════════════════════

def render_chat_entry(entry: dict):
    """Render a single chat history entry."""
    st.markdown(
        f"""<div class="chat-entry">
            <div class="chat-entry-header">
                <span style="color: var(--accent-indigo); font-weight: 600;">
                    ✦ Gemini Response
                </span>
                <span class="chat-entry-meta">
                    {entry.get('timestamp', '')} · {entry.get('segment_count', 0)} segments
                </span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Show the context used (collapsible)
    with st.expander("📋 View Input Context", expanded=False):
        st.markdown(
            f"**System Prompt:**\n```\n{entry.get('system_prompt', '')[:300]}...\n```"
        )
        st.markdown(f"**Segments Used:** {entry.get('segment_count', 0)}")

    # Show the response
    st.markdown(entry.get("response", ""))
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

# Sync selected_segment_ids from checkbox states before sidebar renders
st.session_state.selected_segment_ids = {
    s.id for s in st.session_state.transcript_segments
    if st.session_state.get(f"cb_{s.id}", False)
}

with st.sidebar:
    st.markdown("# 🎙️ Echo-Logic Hub")
    st.markdown(
        '<p style="color: var(--text-muted); font-size: 0.82rem; margin-top: -10px;">'
        "Real-time STT + LLM Orchestration</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Mode Indicators ───────────────────────────────────────
    mock_audio = os.getenv("USE_MOCK_AUDIO", "true").lower() == "true"
    mock_nemo = os.getenv("USE_MOCK_NEMO", "true").lower() == "true"
    mock_gemini = os.getenv("USE_MOCK_GEMINI", "true").lower() == "true"

    st.markdown("##### System Status")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"**Audio:** {'🟡 Mock' if mock_audio else '🟢 Live'}"
        )
        st.markdown(
            f"**NeMo:** {'🟡 Mock' if mock_nemo else '🟢 GPU'}"
        )
    with col_b:
        st.markdown(
            f"**Gemini:** {'🟡 Mock' if mock_gemini else '🟢 API'}"
        )

    st.markdown("---")

    # ── Streaming Controls ────────────────────────────────────
    st.markdown("##### Audio Control")

    if not st.session_state.is_streaming:
        if st.button("🎙️ Start Listening", use_container_width=True, type="primary"):
            start_streaming()
            st.rerun()
    else:
        if st.button("⏹️ Stop Listening", use_container_width=True, type="secondary"):
            stop_streaming()
            st.rerun()

    # Streaming status
    if st.session_state.is_streaming:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-top:8px;">'
            '<span class="status-dot active"></span>'
            '<span style="color:#10b981;font-size:0.85rem;font-weight:500;">'
            "Listening...</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-top:8px;">'
            '<span class="status-dot inactive"></span>'
            '<span style="color:var(--text-muted);font-size:0.85rem;">'
            "Idle</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Session Stats ─────────────────────────────────────────
    st.markdown("##### Session")
    n_segs = len(st.session_state.transcript_segments)
    n_sel = len(st.session_state.selected_segment_ids)
    n_chats = len(st.session_state.chat_history)

    c1, c2, c3 = st.columns(3)
    c1.metric("Segments", n_segs)
    c2.metric("Selected", n_sel)
    c3.metric("Queries", n_chats)

    st.markdown("---")

    # ── Clear Session ─────────────────────────────────────────
    if st.button("🗑️ Clear Session", use_container_width=True):
        clear_session()
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# MAIN LAYOUT — Two Columns
# ═══════════════════════════════════════════════════════════════

# Drain any new segments from the background thread
new_count = drain_result_queue()

left_col, right_col = st.columns([3, 2], gap="large")


# ── LEFT PANEL: Live Transcript Stream ────────────────────────

with left_col:
    # Header
    n_segs = len(st.session_state.transcript_segments)
    status_dot = "active" if st.session_state.is_streaming else "inactive"

    st.markdown(
        f"""<div class="panel-header">
            <span class="panel-header-icon">📝</span>
            Live Transcript Stream
            <span class="segment-count">{n_segs}</span>
            <span class="status-dot {status_dot}" style="margin-left:auto;"></span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Selection controls
    if n_segs > 0:
        sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 3])
        with sel_col1:
            if st.button("Select All", key="select_all", use_container_width=True):
                st.session_state.selected_segment_ids = {
                    s.id for s in st.session_state.transcript_segments
                }
                for s in st.session_state.transcript_segments:
                    st.session_state[f"cb_{s.id}"] = True
                st.rerun()
        with sel_col2:
            if st.button("Deselect All", key="deselect_all", use_container_width=True):
                st.session_state.selected_segment_ids = set()
                for s in st.session_state.transcript_segments:
                    st.session_state[f"cb_{s.id}"] = False
                st.rerun()

    # Transcript cards
    if n_segs == 0:
        st.markdown(
            """<div class="empty-state">
                <div class="empty-state-icon">🎤</div>
                <div class="empty-state-text">
                    No transcript segments yet.<br>
                    Click <strong>Start Listening</strong> in the sidebar to begin.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        transcript_container = st.container(height=600)
        with transcript_container:
            for idx, segment in enumerate(st.session_state.transcript_segments):
                render_speech_card(segment, idx)


# ── RIGHT PANEL: Gemini Workspace ─────────────────────────────

with right_col:
    st.markdown(
        """<div class="panel-header">
            <span class="panel-header-icon">✦</span>
            Gemini Workspace
        </div>""",
        unsafe_allow_html=True,
    )

    # System Prompt
    st.markdown(
        '<div class="system-prompt-container">',
        unsafe_allow_html=True,
    )
    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=150,
        key="system_prompt_input",
        placeholder="Enter your instructions for Gemini here...",
        label_visibility="collapsed",
    )
    st.session_state.system_prompt = system_prompt
    st.markdown("</div>", unsafe_allow_html=True)

    # Execute Button
    n_selected = len(st.session_state.selected_segment_ids)
    btn_label = f"⚡ Execute Selected Context ({n_selected} segments)"
    btn_disabled = n_selected == 0

    st.markdown('<div class="execute-btn-container">', unsafe_allow_html=True)
    execute_clicked = st.button(
        btn_label,
        disabled=btn_disabled,
        use_container_width=True,
        key="execute_btn",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Handle execution
    if execute_clicked and n_selected > 0:
        # Gather selected segments in chronological order
        selected_segments = [
            s for s in st.session_state.transcript_segments
            if s.id in st.session_state.selected_segment_ids
        ]
        selected_segments.sort(key=lambda s: s.start_seconds)

        context_strings = [
            f"[{s.speaker_id} @ {s.timestamp}] {s.text}"
            for s in selected_segments
        ]

        with st.spinner("✦ Gemini is thinking..."):
            client = get_gemini_client()
            response = client.execute(system_prompt, context_strings)

        # Add to chat history
        st.session_state.chat_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "system_prompt": system_prompt,
            "segment_count": len(selected_segments),
            "segments_used": context_strings,
            "response": response,
        })
        st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Chat History
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <span style="font-weight:600;color:var(--text-primary);">
                Chat History
            </span>
            <span class="segment-count">{len(st.session_state.chat_history)}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_history:
        st.markdown(
            """<div class="empty-state" style="padding:32px 16px;">
                <div class="empty-state-icon">✦</div>
                <div class="empty-state-text">
                    No queries yet.<br>
                    Select transcript segments and click Execute.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        chat_container = st.container(height=400)
        with chat_container:
            for entry in reversed(st.session_state.chat_history):
                render_chat_entry(entry)


# ═══════════════════════════════════════════════════════════════
# Auto-Refresh During Streaming
# ═══════════════════════════════════════════════════════════════

if st.session_state.is_streaming:
    time.sleep(1.5)
    st.rerun()
