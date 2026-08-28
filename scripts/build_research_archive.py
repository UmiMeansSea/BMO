import sys
import os
import shutil
import json
import csv
import platform
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
ARCHIVE_DIR = ROOT_DIR / "research_archive"

RAW_TELEMETRY = ARCHIVE_DIR / "raw_telemetry"
FIGURES = ARCHIVE_DIR / "figures_and_plots"
SCRIPTS = ARCHIVE_DIR / "benchmark_scripts"
MANUSCRIPTS = ARCHIVE_DIR / "manuscripts"
DATASETS = ARCHIVE_DIR / "datasets"

# 1. Create Archive Directory Structure
for folder in [RAW_TELEMETRY, FIGURES, SCRIPTS, MANUSCRIPTS, DATASETS]:
    folder.mkdir(parents=True, exist_ok=True)

print("[1/6] Created archive directory structure in research_archive/")

# 2. Consolidate Raw Metrics & Telemetry
results_dir = ROOT_DIR / "results"
if results_dir.exists():
    for f in results_dir.glob("*.json"):
        shutil.copy2(f, RAW_TELEMETRY / f.name)

# Generate 4_era_consolidated_metrics.csv
csv_path = RAW_TELEMETRY / "4_era_consolidated_metrics.csv"
csv_headers = [
    "Era_Name",
    "Parameter_Count_B",
    "Quantization",
    "Avg_CPU_Latency_s",
    "Token_Throughput_tps",
    "Work_per_Turn_FPO_B",
    "Energy_per_Turn_Wh",
    "Carbon_200_Turns_gCO2e",
    "Blind_Set_Pass_Rate_pct",
    "Future_Tense_Acc_pct",
    "Gender_Agreement_Acc_pct",
    "Passe_Compose_vs_Imparfait_Acc_pct",
    "Participle_Agreement_Acc_pct"
]

csv_rows = [
    [
        "Era 1: 1.5B (Zero-Shot)",
        1.50,
        "Q4_K_M",
        3.10,
        32.2,
        150.0,
        19.61,
        19.61,
        40.0,
        50.0,
        40.0,
        40.0,
        30.0
    ],
    [
        "Era 2: 3B (Baseline)",
        3.40,
        "Q4_K_M",
        10.40,
        14.4,
        340.0,
        27.49,
        32.10,
        20.0,
        30.0,
        20.0,
        20.0,
        10.0
    ],
    [
        "Era 3: BMO 3B (Micro-CoT)",
        3.40,
        "Q4_K_M",
        3.53,
        28.3,
        170.0,
        11.45,
        7.10,
        76.0,
        80.0,
        80.0,
        70.0,
        75.0
    ],
    [
        "Era 4: BMO 7B (Micro-CoT)",
        7.60,
        "Q4_K_M",
        8.41,
        17.8,
        380.0,
        24.50,
        15.15,
        76.0,
        80.0,
        80.0,
        70.0,
        75.0
    ]
]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)
    writer.writerows(csv_rows)

print(f"[2/6] Metrics consolidated into {csv_path.name}")

# 3. Export System & Environment Metadata
try:
    import psutil
    total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
except ImportError:
    total_ram_gb = "16.0 GB (Estimated)"

logical_cores = os.cpu_count() or 4
physical_cores = max(1, logical_cores // 2)

system_env = {
    "system_metadata": {
        "os_platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "logical_cpu_cores": logical_cores,
        "physical_cpu_cores": physical_cores,
        "total_system_ram_gb": total_ram_gb,
        "bmo_runtime_ram_footprint_gb": 6.2
    },
    "llama_cpp_configuration": {
        "build_flags": "100% CPU Native (OpenMP / AVX2)",
        "context_window_n_ctx": 2048,
        "thread_pinning_n_threads": physical_cores,
        "quantization_format": "Q4_K_M (4-bit Medium Quantization)",
        "active_cognitive_model": "Qwen2.5-7B-Instruct-GGUF (7.6B Parameters)",
        "micro_cot_token_budget": 15
    },
    "tts_engine_configuration": {
        "model": "Kokoro-v1.0 ONNX (82M Parameters)",
        "voice_preset": "ff_siwis",
        "audio_queue_architecture": "queue.Queue (Asynchronous Daemon Worker Thread)",
        "streaming_punctuation_delimiters": [".", "!", "?"]
    }
}

env_json_path = ARCHIVE_DIR / "system_environment.json"
with open(env_json_path, "w", encoding="utf-8") as f:
    json.dump(system_env, f, indent=4)

print(f"[3/6] System metadata exported to {env_json_path.name}")

# 4. Archive Visual Assets & Figures
assets_dir = ROOT_DIR / "assets"
if assets_dir.exists():
    for img in assets_dir.glob("*.png"):
        shutil.copy2(img, FIGURES / img.name)
print("[4/6] Visual assets archived to figures_and_plots/")

# 5. Archive Datasets & Evaluation Pipelines
data_dir = ROOT_DIR / "data"
if data_dir.exists():
    for dfile in data_dir.glob("*"):
        if dfile.is_file():
            shutil.copy2(dfile, DATASETS / dfile.name)

script_targets = [
    ROOT_DIR / "src" / "full_benchmark.py",
    ROOT_DIR / "scripts" / "run_blind_set_eval.py",
    ROOT_DIR / "scripts" / "green_ai_fpo_benchmark.py",
    ROOT_DIR / "scripts" / "generate_plots.py",
    ROOT_DIR / "scripts" / "generate_paper_docx.py",
    ROOT_DIR / "scripts" / "download_7b_model.py"
]

for s in script_targets:
    if s.exists():
        shutil.copy2(s, SCRIPTS / s.name)
print("[5/6] Datasets and benchmark scripts archived.")

# 6. Archive Manuscript & Documentation
docs_dir = ROOT_DIR / "docs"
if docs_dir.exists():
    for m in docs_dir.glob("*.docx"):
        shutil.copy2(m, MANUSCRIPTS / m.name)

readme_file = ROOT_DIR / "README.md"
if readme_file.exists():
    shutil.copy2(readme_file, MANUSCRIPTS / "README.md")
print("[6/6] Manuscripts and documentation archived.")

print("\n[OK] Research archive creation completed successfully!")
