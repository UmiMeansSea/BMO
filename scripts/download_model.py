"""
BMO Model Downloader
====================
Downloads the Qwen2.5-1.5B-Instruct 4-bit GGUF model from Hugging Face.
Saves to D:\\BMO-Research\\models\ to keep large files off C drive.

This is the recommended "BMO brain" for entry-level student laptops:
  - Size:   ~1.1 GB on disk
  - RAM:    ~1.5 GB when loaded
  - Speed:  2-8 tokens/sec on a modern CPU (no GPU needed)
  - Why:    Qwen2.5 has outstanding multilingual French/English support

The model is saved as 'bmo-model-4bit.gguf' in this folder.

Usage:
    python download_model.py
"""

import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────────────────────
REPO_ID   = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME  = "qwen2.5-3b-instruct-q4_k_m.gguf"

OUTPUT_NAME  = "bmo-model-3b-4bit.gguf"
MODELS_DIR   = Path(r"D:\BMO-Research\models")
OUTPUT_PATH  = MODELS_DIR / OUTPUT_NAME

# Pointer file: tells local_sanity_check.py where the model lives
POINTER_PATH = Path(__file__).parent.parent / ".model_path"

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"[OK] Model already exists: {OUTPUT_NAME} ({size_mb:.0f} MB)")
        print(f"     Path: {OUTPUT_PATH}")
        # Always refresh the pointer file
        POINTER_PATH.write_text(str(OUTPUT_PATH), encoding="utf-8")
        print(f"\n     To re-download, delete the file and run again.")
        return

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("[ERROR] huggingface-hub not found.")
        print("  Run: pip install huggingface-hub")
        sys.exit(1)

    print("=" * 56)
    print("  BMO MODEL DOWNLOADER")
    print("=" * 56)
    print(f"  Repo     : {REPO_ID}")
    print(f"  File     : {FILENAME}")
    print(f"  Save as  : {OUTPUT_NAME}")
    print(f"  Dest     : {MODELS_DIR}  (D drive - not C)")
    print("=" * 56)
    print("\nStarting download (~1.1 GB, may take a few minutes)...")
    print("Progress will appear below.\n")

    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )

        # Rename to standard BMO name if different
        downloaded = Path(downloaded_path)
        if downloaded.name != OUTPUT_NAME:
            downloaded.rename(OUTPUT_PATH)
            print(f"\n[OK] Renamed to {OUTPUT_NAME}")
        else:
            print(f"\n[OK] Saved as {OUTPUT_NAME}")

        size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        # Write pointer file so sanity check knows where to find it
        POINTER_PATH.write_text(str(OUTPUT_PATH), encoding="utf-8")
        print(f"     Size: {size_mb:.0f} MB")
        print(f"     Path: {OUTPUT_PATH}  (on D drive)")
        print(f"     Pointer: {POINTER_PATH}")
        print("\n" + "=" * 56)
        print("  Download complete. You can now run:")
        print("    python local_sanity_check.py")
        print("=" * 56)

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nManual download alternative:")
        print(f"  1. Go to: https://huggingface.co/{REPO_ID}")
        print(f"  2. Download: {FILENAME}")
        print(f"  3. Rename it to: {OUTPUT_NAME}")
        print(f"  4. Place it in: {OUTPUT_PATH.parent}")
        sys.exit(1)

if __name__ == "__main__":
    main()
