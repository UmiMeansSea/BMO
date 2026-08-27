# BMO (Benchmarking Model-quantized Offline Agents for Sustainable Language Tutoring)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Edge Computing](https://img.shields.io/badge/Edge-Local%20Offline-success.svg)](#)

BMO is a fully localized, offline French language learning companion designed to run entirely on entry-level student laptops (CPU-only, sub-4GB RAM usage). It acts as an encouraging, warm peer that holds natural conversations while strictly enforcing pedagogical scaffolding for A2/B1 French learners.

This repository contains the benchmark dataset, local inference engines, JSON Chain-of-Thought (CoT) system prompts, and CodeCarbon energy telemetry scripts developed to measure the sustainability and accuracy of quantized edge small language models (SLMs).

---

## 📌 Architecture Overview

BMO operates on a three-tier localized edge architecture:

1. **The Ears (`whisper.cpp`):** Lightweight speech-to-text transcription model converting spoken French/English to text offline.
2. **The Brain (`llama.cpp` + GGUF Quantized SLM):** Sub-3B Small Language Model (`Qwen2.5-3B-Instruct` 4-bit quantized) running strictly on laptop CPU RAM without external GPU or cloud API dependencies.
3. **The Voice (`Kokoro-82M`):** Ultra-lightweight Text-to-Speech engine synthesizing natural conversational French audio.

---

## 📊 Empirical Benchmark Results (Phase 1 & 2 Baseline)

We benchmarked BMO across a 200-question standardized exam of A2/B1 French conversational prompts (50% erroneous, 50% correct) on CPU hardware using CodeCarbon telemetry (`EmissionsTracker`).

| Metric | Measured Value |
|---|---|
| **Total Test Prompts** | 200 rows (A2/B1 CEFR level) |
| **Model Evaluated** | `Qwen2.5-3B-Instruct-GGUF` (Q4_K_M, 2.0 GB) |
| **Inference Hardware** | 100% CPU-only (n_threads=4, n_gpu_layers=0) |
| **Pass Rate (Accuracy)** | **66.00%** (132 / 200 passed) |
| **Average Latency** | **13.79 seconds / sentence** |
| **Total Evaluation Time** | **45.98 minutes** (200 sequential turns) |
| **Total Energy Consumed** | **27.49 Wh** (0.027491 kWh) |
| **Energy Per Sentence** | **~0.137 Wh / sentence** |
| **Total CO2e Emissions** | **19.61 g CO2e** (0.019613 kg) |
| **Emissions Per Sentence** | **~0.098 g CO2e / sentence** |

---

## 💡 Key Research Discoveries

1. **JSON Chain-of-Thought (CoT) Prevents Model Sycophancy:**
   Standard few-shot prompting caused 1.5B–3B quantized models to over-correct and invent non-existent errors on 80%+ of valid sentences. By introducing a mandatory `"grammatical_analysis"` reasoning field *before* the `"has_error"` boolean, false positives on correct sentences dropped to **0%**.

2. **Ultra-Low Energy Footprint:**
   Executing an entire 200-question interactive oral exam offline consumed only **27.49 Watt-hours** of laptop electricity—equivalent to running a standard 30W laptop charger for under 55 minutes, proving localized AI tutoring is thousands of times more energy efficient than cloud API round-trips.

---

## 🛠️ Project Structure

```
BMO/
├── bmo_french_dataset.csv       # Standardized 200-row French A2/B1 benchmark dataset
├── full_benchmark.py            # Main evaluation runner with CodeCarbon & JSON CoT
├── generate_dataset.py          # Dataset generator & balance script (100 error / 100 correct)
├── local_sanity_check.py        # Fast pilot evaluator for prompt tuning
├── download_model.py            # Hugging Face downloader redirecting GGUFs to D:\ drive
├── validate_dataset.py          # CSV format and column integrity check script
├── full_benchmark_results.json  # Complete raw evaluation output, responses, & latencies
├── sanity_check_results.json    # Initial 10-row pilot results
├── emissions.csv                # CodeCarbon power and CO2 telemetry log
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- `llama-cpp-python` (CPU build)
- `codecarbon`

```bash
pip install llama-cpp-python codecarbon huggingface-hub
```

### 2. Download the Model (D Drive Storage Support)
The downloader automatically redirects model files to `D:\BMO-Research\models\` to protect C drive space:

```bash
python download_model.py
```

### 3. Run the 200-Row Benchmark
```bash
python full_benchmark.py
```

---

## 📜 License

MIT License. Free to use for research and educational purposes.
