import os
import shutil
import subprocess
import sys
from pathlib import Path

def build_bmo():
    print("=" * 60)
    print("  BMO STANDALONE AUTOMATED PACKAGER")
    print("=" * 60)

    root_dir = Path(__file__).parent
    dist_dir = root_dir / "dist" / "bmo_desktop"
    models_dest = dist_dir / "models"

    # 1. Check for PyInstaller
    print("\n[1/4] Checking dependencies...")
    try:
        import PyInstaller
        print("[OK] PyInstaller is installed.")
    except ImportError:
        print("[!] PyInstaller not found. Installing via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 2. Run PyInstaller Compilation
    print("\n[2/4] Compiling src/bmo_desktop.py into a standalone app...")
    script_path = root_dir / "src" / "bmo_desktop.py"
    if not script_path.exists():
        script_path = root_dir / "bmo_desktop.py"

    cmd = [
        "pyinstaller",
        "--noconsole",
        "--onedir",
        "--clean",
        str(script_path)
    ]
    subprocess.check_call(cmd)
    print("[OK] Compilation finished successfully!")

    # 3. Handle Model Assets
    print("\n[3/4] Copying local models to distribution folder...")
    source_models_dir = Path(r"D:\BMO-Research\models")
    
    if source_models_dir.exists():
        if not models_dest.exists():
            os.makedirs(models_dest)
        
        required_models = [
            "bmo-model-4bit.gguf",
            "kokoro-v1.0.onnx",
            "voices-v1.0.bin"
        ]
        
        for model_name in required_models:
            src_file = source_models_dir / model_name
            dst_file = models_dest / model_name
            if src_file.exists():
                print(f"  -> Copying {model_name}...")
                shutil.copy(src_file, dst_file)
            else:
                print(f"  [!] Warning: Could not find {model_name} in source directory.")
        print("[OK] Model assets bundled!")
    else:
        print(f"  [!] Source models directory not found at {source_models_dir}. Please copy your 'models' folder manually into: {models_dest}")

    # 4. Final Summary
    print("\n" + "=" * 60)
    print("  [SUCCESS] BMO IS READY TO SHIP!")
    print(f"  Your portable folder is located at:")
    print(f"  {dist_dir.resolve()}")
    print("  You can now zip this folder and run it on any target device.")
    print("=" * 60)

if __name__ == "__main__":
    build_bmo()
