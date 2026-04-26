import os
import time
import queue
import urllib.request
import threading
from stt_engine import TranscriptionManager, MockWavStream

def main():
    test_wav = "test_sample.wav"
    
    # Generate a synthetic wav file instead of downloading
    if not os.path.exists(test_wav):
        print("Generating synthetic audio file...")
        import wave
        import numpy as np
        
        sample_rate = 16000
        duration = 3.0 # 3 seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Synthetic speech-like fundamental + harmonics + envelope
        f0 = 120 + 30 * np.sin(2 * np.pi * 0.3 * t)
        signal = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
        signal *= envelope
        
        audio_data = (signal * 16000).astype(np.int16)
        
        with wave.open(test_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        print("Generated.")

    print("\n--- Testing STT Engine ---")
    
    stt_eng = TranscriptionManager()
    rq = queue.Queue()
    
    mock_src = MockWavStream(test_wav)
    
    stt_eng.start_stream(mock_src, rq)
    
    print("Engine started, waiting for results (this may take a bit for the first run to download models)...\n")
    
    # Collect results for a few seconds
    start_time = time.time()
    results = []
    
    try:
        while True:
            # We assume the mock stream plays faster or real time
            # Test wav is ~2.5 seconds
            try:
                res = rq.get(timeout=1.0)
                print(f"✅ Result received: [{res.speaker_id} @ {res.timestamp}] {res.text}")
                results.append(res)
            except queue.Empty:
                if not mock_src._active and time.time() - start_time > 10:
                    # Stream ended and we waited a bit
                    break
                
            if time.time() - start_time > 120:
                print("Test timed out.")
                break
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        
    stt_eng.stop_stream()
    print(f"\n--- Test complete. Got {len(results)} segments. ---")

if __name__ == "__main__":
    main()
