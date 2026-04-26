# Echo-Logic Hub — Setup Guide

> **Real-time Speech-to-Text + LLM Orchestration Dashboard**
> NVIDIA Parakeet ASR · Sortformer Diarization · Google Gemini · Streamlit

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start (Mock Mode)](#2-quick-start-mock-mode)
3. [Full Installation (GPU Mode)](#3-full-installation-gpu-mode)
4. [Configuration](#4-configuration)
5. [Running the App](#5-running-the-app)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites

### System Requirements

| Component       | Mock Mode          | Full GPU Mode                    |
|----------------|--------------------|----------------------------------|
| Python          | 3.10+              | 3.10+                            |
| OS              | Windows / Linux    | Windows / Linux                  |
| GPU             | Not needed         | NVIDIA GPU (4+ GB VRAM)          |
| CUDA Toolkit    | Not needed         | 11.8 or 12.x                    |
| Microphone      | Not needed         | Required for live audio          |

### Software Dependencies

- **Git** — [git-scm.com](https://git-scm.com/)
- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **pip** — Comes with Python (ensure it's updated: `python -m pip install --upgrade pip`)

---

## 2. Quick Start (Mock Mode)

Mock mode lets you test the full UI without a GPU, NeMo installation, microphone, or API keys.

### Step 1: Clone / Navigate to the Project

```bash
cd d:\projects\echo-logic-hub
```

### Step 2: Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install streamlit google-generativeai python-dotenv numpy soundfile
```

> **Note:** In mock mode, you do NOT need PyAudio or nemo_toolkit.

### Step 4: Configure Environment

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

The default `.env` has all mock flags set to `true` — no changes needed for testing.

### Step 5: Launch

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Click **Start Listening** in the sidebar to see mock transcript segments appear in real time.

---

## 3. Full Installation (GPU Mode)

### Step 1: Install CUDA Toolkit

Download and install the CUDA Toolkit matching your GPU driver:

- **CUDA 12.x** (recommended): [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Verify installation:
```bash
nvcc --version
nvidia-smi
```

### Step 2: Install PyTorch with CUDA

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU access:
```python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Step 3: Install NeMo Toolkit

```bash
pip install nemo_toolkit[asr]
```

> **Note:** NeMo has many dependencies. If you encounter conflicts, consider using a fresh virtual environment or a conda environment.

### Step 4: Install PortAudio (Required for PyAudio)

**Windows:**
```powershell
# Option A: Using pipwin
pip install pipwin
pipwin install pyaudio

# Option B: Pre-built wheel
pip install PyAudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio
pip install PyAudio
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install portaudio-devel
pip install PyAudio
```

**macOS:**
```bash
brew install portaudio
pip install PyAudio
```

### Step 5: Install All Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Get API Keys

#### NVIDIA NGC API Key
1. Create an account at [ngc.nvidia.com](https://ngc.nvidia.com/)
2. Go to **Setup** → **API Key**
3. Generate and copy your key

#### Google Gemini API Key
1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the key

### Step 7: Configure Environment

Edit your `.env` file:

```env
GEMINI_API_KEY=your_actual_gemini_key
NVIDIA_NGC_API_KEY=your_actual_ngc_key
LOCAL_STT_MODEL_PATH=

# Disable mock modes for production use
USE_MOCK_AUDIO=false
USE_MOCK_NEMO=false
USE_MOCK_GEMINI=false
```

### Step 8: First Run (Model Download)

On the first run with GPU mode, NeMo will download the Parakeet and Sortformer models from NGC. This requires:
- A valid NGC API key
- ~4 GB of disk space for model weights
- ~5-10 minutes depending on connection speed

```bash
streamlit run app.py
```

---

## 4. Configuration

### Environment Variables

| Variable              | Description                                    | Default  |
|-----------------------|------------------------------------------------|----------|
| `GEMINI_API_KEY`      | Google Gemini API key                          | —        |
| `NVIDIA_NGC_API_KEY`  | NVIDIA NGC API key for model downloads          | —        |
| `LOCAL_STT_MODEL_PATH`| Path to a local `.nemo` checkpoint (optional)  | —        |
| `USE_MOCK_AUDIO`      | Use simulated audio instead of microphone       | `true`   |
| `USE_MOCK_NEMO`       | Use fake STT engine (no GPU needed)             | `true`   |
| `USE_MOCK_GEMINI`     | Use canned Gemini responses (no API key needed) | `true`   |

### Mixed Mode Examples

You can mix real and mock components:

```env
# Real Gemini API with mock audio/STT (test LLM integration)
USE_MOCK_AUDIO=true
USE_MOCK_NEMO=true
USE_MOCK_GEMINI=false
GEMINI_API_KEY=your_key_here

# Real audio + STT with mock Gemini (test ASR without API costs)
USE_MOCK_AUDIO=false
USE_MOCK_NEMO=false
USE_MOCK_GEMINI=true
```

---

## 5. Running the App

### Standard Launch

```bash
streamlit run app.py
```

### With Custom Port

```bash
streamlit run app.py --server.port 8080
```

### With Browser Auto-Open Disabled

```bash
streamlit run app.py --server.headless true
```

### Usage Workflow

1. Click **🎙️ Start Listening** in the sidebar
2. Transcript segments will appear in the left panel as speech is detected
3. Check the boxes next to segments you want to analyze
4. Optionally edit the **System Prompt** in the right panel
5. Click **⚡ Execute Selected Context**
6. View Gemini's response in the Chat History section
7. Click **🗑️ Clear Session** to reset everything

---

## 6. Troubleshooting

### PyAudio Installation Fails (Windows)

```
ERROR: Could not build wheels for pyaudio
```

**Solution:** Install the pre-built binary:
```powershell
pip install pipwin
pipwin install pyaudio
```

Or download a `.whl` file from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

### NeMo Import Errors

```
ModuleNotFoundError: No module named 'nemo'
```

**Solution:** Ensure NeMo is installed with ASR extras:
```bash
pip install nemo_toolkit[asr]
```

If conflicts occur, try installing from source:
```bash
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

### CUDA Out of Memory

```
torch.cuda.OutOfMemoryError
```

**Solution:** Try a smaller model variant:
- Edit `nemo_engine.py` and change `parakeet-rnnt-1.1b` to `parakeet-rnnt-0.6b`
- Or increase `diar_buffer_seconds` to process less audio at once

### Streamlit Port Already in Use

```
Address already in use
```

**Solution:**
```bash
streamlit run app.py --server.port 8502
```

### No Audio Input Detected

- Ensure your microphone is connected and set as the default input device
- Check system audio permissions (Settings → Privacy → Microphone)
- Try running with mock audio first to verify the UI works: `USE_MOCK_AUDIO=true`
