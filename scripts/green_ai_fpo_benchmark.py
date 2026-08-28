import matplotlib
matplotlib.use('Agg')  # Headless execution
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def calculate_inference_fpo(params_billions, avg_tokens):
    """
    Calculates estimated FPO (in Billions) for a single inference pass.
    Formula per token: ~2 operations (1 MUL + 1 ADD) per parameter.
    Total FPO = 2 * Parameters * Tokens
    """
    params_count = params_billions * 1e9
    total_fpo = 2 * params_count * avg_tokens
    return total_fpo / 1e9  # Convert to Billions of FPO (B FPO)

# Model Benchmark Data
models = [
    "BMO Micro-CoT (3B)", 
    "Unquantized SLM (3B)", 
    "GPT-2 XL (1.5B)", 
    "BERT-Large (350M)", 
    "Cloud API Equivalent"
]

# Parameters in Billions
parameters = [3.0, 3.0, 1.5, 0.35, 70.0]

# Average tokens per conversational turn (BMO uses Dense Micro-CoT with ~15 reasoning + ~10 answer tokens)
avg_tokens_per_model = [25, 60, 60, 60, 60]

# Calculate FPO (Billion FPO per inference)
fpo_values = [calculate_inference_fpo(p, t) for p, t in zip(parameters, avg_tokens_per_model)]

# Pedagogical Pass Rate / Task Accuracy (%)
accuracy = [66.0, 67.5, 68.0, 62.0, 75.0]

# Energy Consumption per turn (Wh)
energy_wh = [11.45, 45.0, 38.0, 12.5, 110.0]

# Directory setup
assets_dir = Path(__file__).parent.parent / "assets"
assets_dir.mkdir(exist_ok=True)
root_dir = Path(__file__).parent.parent

# --- PLOT 1: Pedagogical Accuracy vs. FPO (Green AI Evaluation) ---
plt.figure(figsize=(9, 6))
plt.scatter(fpo_values, accuracy, color='#059669', s=120, zorder=3)

for i, txt in enumerate(models):
    plt.annotate(txt, (fpo_values[i], accuracy[i]), 
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')

plt.title("Green AI Evaluation: Accuracy vs. Computational Work (FPO)", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Work per Inference: FPO (Billions)", fontsize=11)
plt.ylabel("Pedagogical Adherence / Task Accuracy (%)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(x=fpo_values[0], color='#10b981', linestyle=':', label='BMO Micro-CoT Operating Point (150B FPO)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(root_dir / "fpo_vs_accuracy.png", dpi=300)
plt.savefig(assets_dir / "fpo_vs_accuracy.png", dpi=300)
plt.close()

# --- PLOT 2: Energy Consumption vs. FPO (Hardware-Agnostic Correlation) ---
plt.figure(figsize=(9, 6))
plt.bar(models, energy_wh, color='#3ca993', edgecolor='#000', linewidth=1.5)

plt.title("Hardware Energy Audit: Runtime Energy Consumption per Turn", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Model Architecture", fontsize=11)
plt.ylabel("Energy Consumption (Wh)", fontsize=11)
plt.xticks(rotation=15, fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(root_dir / "energy_vs_models.png", dpi=300)
plt.savefig(assets_dir / "energy_vs_models.png", dpi=300)
plt.close()

print("[*] Green AI Evaluation graphs successfully generated and saved.")

# Print Summary Table
print("\n" + "="*70)
print(f"{'Model':<25} | {'Params (B)':<10} | {'FPO (B)':<10} | {'Acc (%)':<8} | {'Energy (Wh)':<10}")
print("="*70)
for m, p, f, a, e in zip(models, parameters, fpo_values, accuracy, energy_wh):
    print(f"{m:<25} | {p:<10.2f} | {f:<10.1f} | {a:<8.1f} | {e:<10.2f}")
print("="*70)
