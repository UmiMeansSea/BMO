"""
BMO Research Telemetry Visualization Script
===========================================
Generates publication-quality composite charts (300 DPI) summarizing:
  1. Pedagogical Accuracy (Pass Rate) Progression across configurations
  2. Average CPU Latency per Sentence
  3. Total Carbon Footprint (Edge BMO vs Cloud API)
"""

import matplotlib
matplotlib.use('Agg')  # Headless backend for automated script execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 1. Set Publication-Quality Aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

def generate_benchmark_plots():
    print("=" * 60)
    print("  BMO PUBLICATION PLOT GENERATOR")
    print("=" * 60)

    # Create a 1x3 grid for the master composite figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ---------------------------------------------------------
    # Plot 1: Pedagogical Accuracy (Pass Rate) Progression
    # ---------------------------------------------------------
    models_acc = ["1.5B\n(Zero-Shot)", "1.5B\n(Few-Shot)", "3B\n(Direct)", "3B\n(JSON CoT)"]
    pass_rates = [40, 50, 50, 66]
    
    sns.barplot(x=models_acc, y=pass_rates, ax=axes[0], hue=models_acc, legend=False, palette="Blues")
    axes[0].set_title("Pedagogical Accuracy Progression", fontweight="bold")
    axes[0].set_ylabel("Pass Rate (%)")
    axes[0].set_ylim(0, 100)
    
    # Add percentage labels on top of bars
    for i, v in enumerate(pass_rates):
        axes[0].text(i, v + 2, f"{v}%", ha="center", fontweight="bold", color="#333333")
        
    # ---------------------------------------------------------
    # Plot 2: Inference Latency vs. Model Architecture
    # ---------------------------------------------------------
    models_lat = ["1.5B", "1.5B\n(GBNF)", "3B\n(Direct)", "3B\n(CoT)"]
    latencies = [3.1, 8.1, 10.57, 13.79]
    
    sns.lineplot(x=models_lat, y=latencies, marker="o", linewidth=2.5, 
                 markersize=8, color="#d95f02", ax=axes[1])
    axes[1].set_title("Average CPU Latency per Sentence", fontweight="bold")
    axes[1].set_ylabel("Seconds (s)")
    axes[1].set_ylim(0, 20)
    
    # Annotate data points
    for i, v in enumerate(latencies):
        axes[1].text(i, v + 0.8, f"{v}s", ha="center", color="#d95f02", fontweight="bold")

    # ---------------------------------------------------------
    # Plot 3: Carbon Footprint (Edge vs. Cloud)
    # ---------------------------------------------------------
    architectures = ["Edge Inference\n(BMO 3B)", "Cloud API\n(Estimated)"]
    emissions = [19.61, 343.14]  # Grams of CO2e
    
    # Use green for Edge, purple for Cloud to emphasize environmental thesis
    sns.barplot(x=architectures, y=emissions, ax=axes[2], hue=architectures, legend=False, palette=["#1b9e77", "#7570b3"])
    axes[2].set_title("Total Carbon Emissions (200 Turns)", fontweight="bold")
    axes[2].set_ylabel("Emissions (g CO2e)")
    
    # Use a logarithmic scale to properly display the 94.2% reduction delta
    axes[2].set_yscale("log") 
    
    # Annotate bars
    for i, v in enumerate(emissions):
        axes[2].text(i, v * 1.25, f"{v}g", ha="center", fontweight="bold", color="#333333")

    # Polish and save the composite figure
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / "assets" / "bmo_full_benchmark_metrics.png"
    plt.savefig(output_path)
    print(f"[*] Saved publication figure (300 DPI) to: '{output_path.name}'")
    print("=" * 60)

if __name__ == "__main__":
    generate_benchmark_plots()
