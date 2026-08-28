import urllib.request
import sys
import time
from pathlib import Path

MODELS_DIR = Path(r"D:\BMO-Research\models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEST_PATH = MODELS_DIR / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

URL = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

print(f"[*] Downloading Qwen 2.5 7B Instruct from:\n    {URL}")
print(f"[*] Destination: {DEST_PATH}")

start_time = time.time()

def reporthook(count, block_size, total_size):
    global start_time
    duration = max(time.time() - start_time, 0.001)
    downloaded = count * block_size
    percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
    speed = (downloaded / (1024 * 1024)) / duration
    downloaded_mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f"\rDownloading: {percent}% [{downloaded_mb:.1f}/{total_mb:.1f} MB] ({speed:.2f} MB/s)")
    sys.stdout.flush()

try:
    urllib.request.urlretrieve(URL, str(DEST_PATH), reporthook)
    print("\n[OK] Download complete!")
except Exception as e:
    print(f"\n[!] Download failed: {e}")
