import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

ASSETS_DIR = Path(__file__).parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
ROOT_DIR = Path(__file__).parent.parent

def generate_plots():
    # Chart 1: Master Composite 1x3 Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models_acc = ["1.5B\n(Zero-Shot)", "1.5B\n(Few-Shot)", "3B\n(Direct)", "BMO 3B\n(Micro-CoT)"]
    pass_rates = [40, 50, 50, 76]
    sns.barplot(x=models_acc, y=pass_rates, ax=axes[0], hue=models_acc, legend=False, palette="Blues")
    axes[0].set_title("Pedagogical Accuracy Progression", fontweight="bold")
    axes[0].set_ylabel("Pass Rate (%)")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(pass_rates):
        axes[0].text(i, v + 2, f"{v}%", ha="center", fontweight="bold", color="#333333")
        
    models_lat = ["1.5B", "1.5B\n(GBNF)", "3B\n(Std CoT)", "BMO 3B\n(Micro-CoT)"]
    latencies = [3.1, 8.1, 13.79, 3.53]
    sns.lineplot(x=models_lat, y=latencies, marker="o", linewidth=2.5, 
                 markersize=8, color="#d95f02", ax=axes[1])
    axes[1].set_title("Average CPU Latency per Sentence", fontweight="bold")
    axes[1].set_ylabel("Seconds (s)")
    axes[1].set_ylim(0, 16)
    for i, v in enumerate(latencies):
        axes[1].text(i, v + 0.6, f"{v}s", ha="center", color="#d95f02", fontweight="bold")

    architectures = ["Edge Inference\n(BMO Micro-CoT)", "Cloud API\n(Estimated)"]
    emissions = [7.10, 343.14]
    sns.barplot(x=architectures, y=emissions, ax=axes[2], hue=architectures, legend=False, palette=["#1b9e77", "#7570b3"])
    axes[2].set_title("Total Carbon Emissions (200 Turns)", fontweight="bold")
    axes[2].set_ylabel("Emissions (g CO2e)")
    axes[2].set_yscale("log") 
    for i, v in enumerate(emissions):
        axes[2].text(i, v * 1.25, f"{v}g", ha="center", fontweight="bold", color="#333333")

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "bmo_full_benchmark_metrics.png")
    plt.close()

    # Chart 2: Error-Type Breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = [
        "Future Tense",
        "Gender Agreement",
        "Passé Composé\nvs. Imparfait",
        "Past Participle\nAgreement"
    ]
    overall_acc = [70.0, 68.0, 64.0, 76.0]
    err_detection_acc = [80.0, 40.0, 59.1, 95.0]
    correct_validation_acc = [60.0, 96.0, 67.9, 57.0]

    x = np.arange(len(categories))
    width = 0.25

    rects1 = ax.bar(x - width, err_detection_acc, width, label="Erroneous Detection", color="#e74c3c", alpha=0.85)
    rects2 = ax.bar(x, overall_acc, width, label="Overall Accuracy", color="#2980b9", alpha=0.95)
    rects3 = ax.bar(x + width, correct_validation_acc, width, label="Correct Validation", color="#2ecc71", alpha=0.85)

    ax.set_ylabel("Accuracy / Pass Rate (%)", fontweight="bold")
    ax.set_title("Pedagogical Accuracy by French Grammar Category (CEFR A2/B1)", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(66.0, color="#7f8c8d", linestyle="--", linewidth=1.5, label="Benchmark Threshold (66%)")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color="#1b4f72")
    for rect in rects3:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "bmo_error_type_breakdown.png")
    plt.close()

    # Chart 3: Efficiency Pareto Frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    variants = [
        {"name": "1.5B (Zero-Shot)", "latency": 3.10, "pass_rate": 40.0, "color": "#95a5a6", "marker": "o"},
        {"name": "1.5B (GBNF)", "latency": 8.10, "pass_rate": 50.0, "color": "#3498db", "marker": "s"},
        {"name": "3B (Direct)", "latency": 10.57, "pass_rate": 50.0, "color": "#e67e22", "marker": "^"},
        {"name": "3B (Std CoT)", "latency": 13.79, "pass_rate": 66.0, "color": "#7f8c8d", "marker": "v"},
        {"name": "BMO 3B (Dense Micro-CoT)", "latency": 3.53, "pass_rate": 76.0, "color": "#27ae60", "marker": "*", "size": 250},
        {"name": "Cloud API (GPT-4 Est.)", "latency": 2.50, "pass_rate": 88.0, "color": "#8e44ad", "marker": "D"},
    ]

    frontier_x = [3.10, 3.53]
    frontier_y = [40.0, 76.0]
    ax.plot(frontier_x, frontier_y, "--", color="#2c3e50", alpha=0.6, linewidth=2, label="Edge Efficiency Pareto Frontier")

    for var in variants:
        size = var.get("size", 140)
        ax.scatter(var["latency"], var["pass_rate"], color=var["color"], s=size, 
                   marker=var["marker"], zorder=5, edgecolors="black", linewidth=1.2)
        
        offset_x, offset_y = 0.35, 1.2
        if "Dense Micro-CoT" in var["name"]:
            offset_x, offset_y = -0.5, 2.5
        elif "Cloud API" in var["name"]:
            offset_x, offset_y = 0.35, -2.5
        elif "3B (Direct)" in var["name"]:
            offset_x, offset_y = 0.35, -3.0

        ax.annotate(
            f"{var['name']}\n({var['pass_rate']}%, {var['latency']}s)",
            (var["latency"], var["pass_rate"]),
            xytext=(var["latency"] + offset_x, var["pass_rate"] + offset_y),
            fontsize=10,
            fontweight="bold" if "Dense Micro-CoT" in var["name"] else "normal",
            color=var["color"] if "Dense Micro-CoT" in var["name"] else "#2c3e50"
        )

    bmo_opt = next(v for v in variants if "Dense Micro-CoT" in v["name"])
    ax.annotate(
        "Optimal Edge Operating Point\n(76% Acc @ 3.53s CPU Latency)",
        xy=(bmo_opt["latency"], bmo_opt["pass_rate"]),
        xytext=(bmo_opt["latency"] + 1.2, bmo_opt["pass_rate"] - 12),
        arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=2, headwidth=8),
        fontsize=10,
        fontweight="bold",
        color="#1e8449",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f8f5", edgecolor="#27ae60", alpha=0.9)
    )

    ax.set_title("Efficiency Pareto Frontier: Latency vs. Pedagogical Pass Rate", fontweight="bold", pad=15)
    ax.set_xlabel("Average CPU Latency per Sentence (Seconds)", fontweight="bold")
    ax.set_ylabel("Pedagogical Pass Rate (%)", fontweight="bold")
    ax.set_xlim(0, 16)
    ax.set_ylim(30, 95)

    ax.legend(loc="lower right", frameon=True, facecolor="white")
    plt.tight_layout()

    plt.savefig(ASSETS_DIR / "bmo_pareto_efficiency_frontier.png")
    plt.close()
    print("[*] All publication plots updated with empirical Micro-CoT benchmarks!")

if __name__ == "__main__":
    generate_plots()
