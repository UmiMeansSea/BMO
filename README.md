# BMO (Benchmarking Model-quantized Offline Agents for Sustainable Language Tutoring)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Edge Computing](https://img.shields.io/badge/Edge-Local%20Offline-success.svg)](#)
[![Release](https://img.shields.io/badge/Release-v1.0.0--beta-brightgreen.svg)](#)

BMO is an edge-native, voice-to-voice offline French language learning companion designed to run entirely on entry-level student laptops (CPU-only, sub-4GB RAM usage). It acts as an encouraging, warm peer that holds natural conversations while enforcing pedagogical scaffolding for A2/B1 French learners.

---

## 📌 Architecture Overview

BMO operates on a three-tier localized edge pipeline powered by 3 dedicated state engines:

```
┌─────────────────┐      ┌─────────────────────────┐      ┌───────────────────────┐
│ Spoken French   │ ───> │ French-Only Whisper ASR │ ───> │ Name Normalization    │
│ Audio Input     │      │ (language="fr")         │      │ ("BMO"/"Beemo")       │
└─────────────────┘      └─────────────────────────┘      └───────────┬───────────┘
                                                                      │
                                                                      ▼
┌─────────────────┐      ┌─────────────────────────┐      ┌───────────────────────┐
│ Kokoro-82M TTS  │ <─── │ Qwen 3B 4-bit SLM       │ <─── │ State Engine Routing  │
│ (Adaptive Speed)│      │ (Dual FR/EN Generation) │      │ (Roleplay/Scaffold)   │
└─────────────────┘      └─────────────────────────┘      └───────────────────────┘
```

1. **Speech Recognition (Whisper ASR)**: Locked to French (`language="fr"`) with phonetic prompts preserving learner mispronunciations without auto-correcting them.
2. **Name Recognition & Normalization**: Maps phonetic variations (`Beemo`, `Bemo`, `Bi Mo`) to BMO's canonical wake name.
3. **Dynamic Roleplay Engine (`RoleplayEngine`)**: Detects real-world scenario requests (Parisian Café, Boulangerie, Gare de Lyon, Boutique Hotel), maintains in-character persona for 3–4 turns, and delivers an exit debrief.
4. **Adaptive Scaffolding Engine (`ScaffoldingEngine`)**: 3-tiered hint hierarchy providing sentence starters, vocabulary anchors, and hesitation recovery at **`0.70x` slow-playback speed**.
5. **Session Memory Engine (`SessionMemoryEngine`)**: Saves longitudinal metrics to `session_review.json`, generates startup warm-up quizzes on past weak points, and extracts vocabulary on session exit.
6. **BMO Native GUI**: PyWebView desktop dashboard featuring a retro BMO chassis and pastel speech bubbles with styled bottom-right `🌐 Translate` buttons.

---

## 📊 Empirical Benchmark Results

We benchmarked BMO across a 200-question standardized exam of A2/B1 French conversational prompts (50% erroneous, 50% correct) on CPU hardware using CodeCarbon telemetry (`EmissionsTracker`).

| Metric | Measured Value |
|---|---|
| **Total Test Prompts** | 200 rows (A2/B1 CEFR level) |
| **Model Evaluated** | `Qwen2.5-3B-Instruct-GGUF` (Q4_K_M, 2.0 GB) |
| **Inference Hardware** | 100% CPU-only (`n_threads=4`, `n_gpu_layers=0`) |
| **Pedagogical Accuracy** | **66.00%** (132 / 200 passed) |
| **Average Latency** | **13.79 seconds / sentence** |
| **Total Evaluation Time** | **45.98 minutes** (200 sequential turns) |
| **Total Energy Consumed** | **27.49 Wh** (0.027491 kWh) |
| **Energy Per Sentence** | **~0.137 Wh / sentence** |
| **Total CO2e Emissions** | **19.61 g CO2e** (0.019613 kg) |
| **Emissions Per Sentence** | **~0.098 g CO2e / sentence** |

---

## 📂 Repository Structure

```
BMO/
├── assets/
│   └── bmo_full_benchmark_metrics.png   # Publication-quality 300 DPI composite figure
├── data/
│   └── bmo_french_dataset.csv           # Standardized 200-row French A2/B1 benchmark exam
├── docs/
│   └── BMO_Research_Paper.pdf           # Technical research paper and architectural analysis
├── evaluations/
│   ├── full_benchmark.py                # 200-row evaluation runner with CodeCarbon & JSON CoT
│   ├── full_benchmark_results.json      # Complete raw 200-row evaluation outputs & latencies
│   └── emissions.csv                    # CodeCarbon power and CO2 telemetry log
├── scripts/
│   ├── bmo_qa_evaluator.py              # Automated 5-category QA harness
│   ├── download_model.py                # Hugging Face downloader for GGUF weights
│   ├── download_kokoro.py               # Kokoro-82M ONNX model & voice downloader
│   └── generate_plots.py                # Seaborn/Matplotlib plot renderer
├── src/
│   ├── bmo_desktop.py                   # Main PyWebView GUI & Integrated 3-Engine pipeline
│   ├── bmo_dashboard.py                 # Gradio web interface option
│   └── bmo_live.py                      # Async live terminal pipeline
├── build_bmo.py                         # Automated PyInstaller packaging script
├── .gitignore                           # Git ignore rules for models, build, and dist
└── README.md                            # Main project documentation
```

---

## 🚀 Getting Started

### 1. Installation

```bash
pip install llama-cpp-python kokoro-onnx pywhispercpp sounddevice scipy pywebview codecarbon seaborn matplotlib huggingface_hub
```

### 2. Download Models

Download the Qwen 3B GGUF brain and Kokoro-82M ONNX voice engine:

```bash
python scripts/download_model.py
python scripts/download_kokoro.py
```

### 3. Run Desktop Companion App

```bash
python src/bmo_desktop.py
```

### 4. Build Standalone Package

To compile the standalone Windows executable:

```bash
python build_bmo.py
```

The compiled package will be available under `dist/bmo_desktop/`.

---

## 📜 License

MIT License. Free to use for research and educational purposes.
