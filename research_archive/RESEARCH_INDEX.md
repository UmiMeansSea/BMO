# BMO Research Archive Index & Reproducibility Guide

This archive contains the complete empirical data, telemetry logs, high-resolution figures, evaluation code, system environment metadata, and manuscript sources for the **BMO (Benchmarking Model-quantized Offline Agents for Sustainable Language Tutoring)** project.

---

## 📂 Directory Structure Overview

```
research_archive/
├── RESEARCH_INDEX.md                   # Master reproducibility guide and figure mapping
├── system_environment.json             # Hardware, RAM, thread pinning, and LLM build specs
├── requirements_reproducibility.txt    # Exact Python environment freeze (`pip freeze`)
├── benchmark_scripts/                  # Autonomous evaluation and plot rendering scripts
│   ├── full_benchmark.py               # 200-row dataset macro evaluation runner
│   ├── run_blind_set_eval.py           # 50-sentence blind set holdout evaluator
│   ├── green_ai_fpo_benchmark.py       # FPO calculation & Green AI audit script
│   ├── generate_plots.py               # Seaborn/Matplotlib 300 DPI figure renderer
│   ├── generate_paper_docx.py          # Automated research paper compiler
│   └── download_7b_model.py            # Hugging Face weights acquisition script
├── datasets/                           # Evaluation corpora and gold holdout sets
│   ├── bmo_french_dataset.csv          # 200-prompt French morphological dataset
│   ├── past_participle_blind_set.json  # 50-sentence unseen blind holdout set
│   └── past_participle_dev_set.json    # 10-sentence diagnostic dev set
├── figures_and_plots/                  # 300 DPI publication figures
│   ├── bmo_full_benchmark_metrics_7b.png
│   ├── bmo_pareto_efficiency_frontier_7b.png
│   ├── bmo_error_type_breakdown_7b.png
│   ├── fpo_vs_accuracy_7b.png
│   └── energy_vs_models_7b.png
├── manuscripts/                        # Final manuscripts & documentation
│   ├── BMO_Green_AI_Research_Paper.docx
│   └── README.md
└── raw_telemetry/                      # Consolidated evaluation telemetry
    ├── 4_era_consolidated_metrics.csv  # 4-Era comparative metrics CSV
    ├── blind_holdout_results.json      # Blind set evaluation log
    ├── full_benchmark_results.json     # 200-row benchmark evaluation log
    ├── baseline_dev_set_results.json   # 3B Baseline dev set log
    └── cot_dev_set_results.json        # 3B Micro-CoT dev set log
```

---

## 📊 Figure & Table Source Mapping

| Manuscript Artifact | Description | Generating Script | Primary Data Source | Output Location |
|---|---|---|---|---|
| **Table 1** | Multi-Generational Architecture Comparison | `benchmark_scripts/green_ai_fpo_benchmark.py` | `raw_telemetry/4_era_consolidated_metrics.csv` | `manuscripts/BMO_Green_AI_Research_Paper.docx` |
| **Figure 1** | Master Composite Benchmark Summary (7B) | `benchmark_scripts/generate_plots.py` | `raw_telemetry/4_era_consolidated_metrics.csv` | `figures_and_plots/bmo_full_benchmark_metrics_7b.png` |
| **Figure 2** | Efficiency Pareto Frontier (Latency vs. Accuracy) | `benchmark_scripts/generate_plots.py` | `raw_telemetry/4_era_consolidated_metrics.csv` | `figures_and_plots/bmo_pareto_efficiency_frontier_7b.png` |
| **Figure 3** | Pedagogical Error-Type Breakdown | `benchmark_scripts/generate_plots.py` | `raw_telemetry/blind_holdout_results.json` | `figures_and_plots/bmo_error_type_breakdown_7b.png` |
| **Figure 4** | Accuracy vs. Computational Work (FPO) | `benchmark_scripts/green_ai_fpo_benchmark.py` | `raw_telemetry/4_era_consolidated_metrics.csv` | `figures_and_plots/fpo_vs_accuracy_7b.png` |
| **Figure 5** | Runtime Energy Consumption per Model | `benchmark_scripts/green_ai_fpo_benchmark.py` | `raw_telemetry/4_era_consolidated_metrics.csv` | `figures_and_plots/energy_vs_models_7b.png` |

---

## 🔬 Peer Reviewer Reproduction Protocol

To reproduce all empirical telemetry, benchmarks, and publication figures from scratch:

### 1. Environment Setup
```bash
# Clone repository and install exact dependencies
pip install -r research_archive/requirements_reproducibility.txt
```

### 2. Download Model Weights
```bash
python research_archive/benchmark_scripts/download_7b_model.py
```

### 3. Run 50-Sentence Blind Set Evaluation
```bash
python research_archive/benchmark_scripts/run_blind_set_eval.py
```

### 4. Run 200-Turn Macro Sustainability Audit
```bash
python research_archive/benchmark_scripts/full_benchmark.py
```

### 5. Regenerate 300 DPI Publication Plots
```bash
python research_archive/benchmark_scripts/green_ai_fpo_benchmark.py
python research_archive/benchmark_scripts/generate_plots.py
```

### 6. Recompile Research Manuscript
```bash
python research_archive/benchmark_scripts/generate_paper_docx.py
```

---

## 🧮 Mathematical Formulation

Floating Point Operations per inference turn are calculated strictly as:

$$\text{FPO} = 2 \times P \times T$$

Where:
- $P$ = Active model parameters ($3.4 \times 10^9$ for 3B, $7.6 \times 10^9$ for 7B).
- $T$ = Average generated tokens per turn ($T \approx 25$ for Micro-CoT).
- Factor $2$ accounts for 1 multiplication and 1 addition per parameter per token.
