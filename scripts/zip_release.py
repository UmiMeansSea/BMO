import zipfile
from pathlib import Path

source_dir = Path("dist/bmo_desktop")
zip_path = Path("bmo_desktop_v1.0.0_windows.zip")

print("Packaging dist/bmo_desktop into bmo_desktop_v1.0.0_windows.zip...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in source_dir.rglob("*"):
        if file.is_file():
            zipf.write(file, file.relative_to(source_dir.parent))
print("ZIP creation complete!")
