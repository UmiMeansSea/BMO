"""
BMO Voice Test - Kokoro-82M TTS Playback
========================================
Synthesizes and plays BMO's French greeting offline via Kokoro ONNX.
"""

import sys
import os
from pathlib import Path
import numpy as np

MODELS_DIR = Path(r"D:\BMO-Research\models")
MODEL_FILE = MODELS_DIR / "kokoro-v1.0.onnx"
VOICES_FILE = MODELS_DIR / "voices-v1.0.bin"

def test_voice():
    print("=" * 60)
    print("  BMO LOCAL TTS VOICE TEST (Kokoro-82M)")
    print("=" * 60)
    print(f"[*] Model Path  : {MODEL_FILE}")
    print(f"[*] Voices Path : {VOICES_FILE}")

    try:
        import sounddevice as sd
        from kokoro_onnx import Kokoro
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("  Run: pip install kokoro-onnx sounddevice numpy")
        sys.exit(1)

    if not MODEL_FILE.exists() or not VOICES_FILE.exists():
        print(f"[!] Model or voices file missing in {MODELS_DIR}")
        print("  Run: python download_kokoro.py")
        sys.exit(1)

    # Patch kokoro_onnx speed tensor data type bug (float32 vs int32)
    def patched_create_audio(self, phonemes, voice, speed):
        tokens = np.array(self.tokenizer.tokenize(phonemes[:510]), dtype=np.int64)
        voice_style = voice[len(tokens)]
        tokens_input = [[0, *tokens, 0]]
        inputs = {
            "input_ids": tokens_input,
            "style": np.array(voice_style, dtype=np.float32),
            "speed": np.array([speed], dtype=np.float32),
        }
        audio = self.sess.run(None, inputs)[0]
        return audio, 24000

    Kokoro._create_audio = patched_create_audio

    print("\nLoading Kokoro TTS engine into memory...")
    kokoro = Kokoro(str(MODEL_FILE), str(VOICES_FILE))

    french_text = "Bonjour ! Je m'appelle BMO. Je suis prêt à t'aider avec ton français."
    print(f"\n[*] Synthesizing text: \"{french_text}\"")

    try:
        audio, sample_rate = kokoro.create(french_text, voice="ff_siwis", speed=1.0, lang="fr-fr")
    except Exception:
        audio, sample_rate = kokoro.create(french_text, voice="af_bella", speed=1.0)

    # Handle shape (1, N) or (N,)
    audio_flat = audio.squeeze()

    print(f"[OK] Audio generated! Sample rate: {sample_rate} Hz, Array shape: {audio_flat.shape}")
    print("[*] Playing audio through speakers...")

    sd.play(audio_flat, sample_rate)
    sd.wait()

    print("\n" + "=" * 60)
    print("  [SUCCESS] Audio playback completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_voice()
