"""
BMO Kokoro TTS Downloader
=========================
Downloads Kokoro-82M ONNX model and voices file from Hugging Face directly to D:\BMO-Research\models\
"""

import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path(r"D:\BMO-Research\models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  BMO KOKORO TTS MODEL DOWNLOADER")
print("=" * 60)
print(f"  Target Directory: {MODELS_DIR}")

target_model = MODELS_DIR / "kokoro-v1.0.onnx"
target_voices = MODELS_DIR / "voices-v1.0.bin"

try:
    if not target_model.exists():
        print("\n1/2. Downloading ONNX model file (kokoro-v1.0.onnx)...")
        model_path = hf_hub_download(
            repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
            filename="onnx/model.onnx",
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        src_model = MODELS_DIR / "onnx" / "model.onnx"
        if src_model.exists():
            shutil.move(str(src_model), str(target_model))
        print(f"  [OK] Model saved to: {target_model}")
    else:
        print(f"\n1/2. [OK] ONNX model file exists: {target_model}")

    if not target_voices.exists():
        print("\n2/2. Downloading voices binary file (voices-v1.0.bin)...")
        voices_path = hf_hub_download(
            repo_id="qte123/kokoro-voices",
            filename="models/voices-v1.0.bin",
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        src_voices = MODELS_DIR / "models" / "voices-v1.0.bin"
        if src_voices.exists():
            shutil.move(str(src_voices), str(target_voices))
        print(f"  [OK] Voices file saved to: {target_voices}")
    else:
        print(f"\n2/2. [OK] Voices file exists: {target_voices}")

    print("\n" + "=" * 60)
    print("  Kokoro-82M TTS download complete!")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] Download failed: {e}")
    sys.exit(1)
