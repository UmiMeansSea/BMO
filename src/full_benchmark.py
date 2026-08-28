import csv
import json
import time
import sys
from pathlib import Path
from llama_cpp import Llama
from codecarbon import EmissionsTracker

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
_POINTER = Path(__file__).parent / ".model_path"
MODEL_PATH = Path(_POINTER.read_text(encoding="utf-8").strip()) if _POINTER.exists() \
                else Path(r"D:\BMO-Research\models\bmo-model-3b-4bit.gguf")

DATASET_PATH = Path(__file__).parent.parent / "data" / "bmo_french_dataset.csv"

# -------------------------------------------------------------------------
# Balanced JSON Chain-of-Thought (CoT) System Prompt
# -------------------------------------------------------------------------
SYSTEM_PROMPT = """You are BMO, a warm, quirky, and supportive French language tutor. You are speaking to an A2/B1 beginner.

Your task is to analyze the user's French sentence, detect any grammatical or conjugation errors, and provide pedagogical feedback. 

Analyze the sentence objectively. Pay careful attention to verbs: habitual past actions or continuous past states require the Imparfait, while single completed past actions require the Passé Composé.

You must ALWAYS respond with a valid JSON object matching this exact schema:
{
  "grammatical_analysis": "In 1 concise sentence, evaluate the subject, verb tenses, and agreement rules.",
  "has_error": true/false,
  "error_type": "Passé composé vs Imparfait" | "Gender agreement" | "Future tense" | "Past Participle Agreement" | "None",
  "feedback": "string"
}

Exemplar 1 (Correct Sentence):
User: "J'ai mangé une pomme ce matin."
Response: {
  "grammatical_analysis": "The subject 'Je' correctly uses the passé composé 'ai mangé' for a completed past action.",
  "has_error": false,
  "error_type": "None",
  "feedback": "C'est délicieux ! Tu as mangé quoi d'autre pour le petit déjeuner ?"
}

Exemplar 2 (Incorrect Sentence - Tense/Habit):
User: "Quand j'étais petit, j'ai joué au foot tous les jours."
Response: {
  "grammatical_analysis": "The phrase 'tous les jours' indicates a habitual past action, which requires the Imparfait 'jouais', not the passé composé 'ai joué'.",
  "has_error": true,
  "error_type": "Passé composé vs Imparfait",
  "feedback": "Nice try! Because this was a daily habit in your childhood, you need the Imparfait here: 'jouais'. Say it with me: 'Quand j'étais petit, je jouais au foot tous les jours.'"
}

Exemplar 3 (Incorrect Sentence - Present vs Past):
User: "Hier, je mange avec mes amis."
Response: {
  "grammatical_analysis": "The sentence starts with 'Hier' (yesterday), but incorrectly uses the present tense 'mange' instead of the passé composé 'ai mangé'.",
  "has_error": true,
  "error_type": "Passé composé vs Imparfait",
  "feedback": "Nice try! You said 'Hier, je mange', but because this happened yesterday, you need the passé composé! Say this back to me: 'Hier, j'ai mangé avec mes amis.'"
}
"""

def load_full_dataset(csv_path: Path):
    prompts = []
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            prompts.append({
                "id": row.get("Prompt_ID", str(idx + 1)),
                "text": row.get("User_Input_French", ""),
                "expected_error": row.get("Error_Type", "None"),
                "is_erroneous": row.get("Is_Erroneous", "FALSE").strip().upper() == "TRUE"
            })
    return prompts

def run_full_benchmark():
    print("=" * 80)
    print("  BMO FULL BENCHMARK EVALUATION (200 ROWS)")
    print("=" * 80)
    print(f"[*] Model Path  : {MODEL_PATH}")
    print(f"[*] Dataset     : {DATASET_PATH.name}")

    if not Path(MODEL_PATH).exists():
        print(f"[!] Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    # Initialize llama.cpp engine with CPU constraints
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        n_threads=4,      # Adjust based on laptop CPU cores
        n_gpu_layers=0,   # 100% CPU inference
        verbose=False
    )

    prompts = load_full_dataset(DATASET_PATH)
    print(f"[*] Loaded {len(prompts)} prompts from {DATASET_PATH.name}\n")

    results = []
    passed_count = 0
    total_latency = 0.0

    print("[*] Starting CodeCarbon EmissionsTracker...")
    tracker = EmissionsTracker(
        project_name="bmo_edge",
        measure_power_secs=1,
        log_level="warning",
        output_dir=str(Path(__file__).parent)
    )
    tracker.start()

    print("=" * 80)
    print(f"{'ID':<6} | {'Target Error':<30} | {'Model Detected':<20} | {'Status':<8} | {'Latency'}")
    print("=" * 80)

    start_eval_time = time.perf_counter()

    for item in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["text"]}
        ]

        start_time = time.perf_counter()
        
        # Enforce JSON-object decoding
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,       # Strict deterministic decoding
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        latency = time.perf_counter() - start_time
        total_latency += latency

        raw_output = response["choices"][0]["message"]["content"]
        
        try:
            parsed = json.loads(raw_output)
            model_has_error = parsed.get("has_error", False)
            model_error_type = parsed.get("error_type", "None")
            feedback = parsed.get("feedback", "")
            analysis = parsed.get("grammatical_analysis", "")
            
            ground_truth_has_error = item["is_erroneous"]
            
            is_pass = (model_has_error == ground_truth_has_error)
            if is_pass:
                passed_count += 1
                status = "PASS"
            else:
                status = "FAIL"

            results.append({
                "id": item["id"],
                "input": item["text"],
                "expected": item["expected_error"],
                "parsed": parsed,
                "status": status,
                "latency": latency
            })

            print(f"{item['id']:<6} | {item['expected_error']:<30} | {str(model_error_type):<20} | {status:<8} | {latency:.2f}s")
            sys.stdout.flush()

        except json.JSONDecodeError:
            print(f"{item['id']:<6} | {item['expected_error']:<30} | {'PARSER ERROR':<20} | {'FAIL':<8} | {latency:.2f}s")

    emissions_kg = tracker.stop()
    total_eval_wall_time = time.perf_counter() - start_eval_time

    # Calculate energy consumed in Watt-hours
    energy_kwh = tracker._total_energy.kWh if hasattr(tracker, '_total_energy') and tracker._total_energy else 0.0
    energy_wh = energy_kwh * 1000.0

    print("=" * 80)
    print("FULL BENCHMARK SUMMARY:")
    print("=" * 80)
    print(f"Total Evaluated        : {len(prompts)} rows")
    print(f"Passed                 : {passed_count}/{len(prompts)} ({(passed_count/len(prompts))*100:.2f}%)")
    print(f"Failed                 : {len(prompts) - passed_count}")
    print(f"Average Latency        : {total_latency / len(prompts):.2f}s per sentence")
    print(f"Total Benchmark Time   : {total_eval_wall_time / 60.0:.2f} minutes")
    print(f"Total Energy Consumed  : {energy_wh:.2f} Wh ({energy_kwh:.6f} kWh)")
    print(f"Total CO2 Emissions    : {emissions_kg * 1000.0:.2f} g CO2e ({emissions_kg:.6f} kg)")
    print("=" * 80)

    # Save benchmark report artifact
    output_path = Path(__file__).parent.parent / "results" / "full_benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_rows": len(prompts),
                "passed": passed_count,
                "failed": len(prompts) - passed_count,
                "pass_rate_pct": round((passed_count/len(prompts))*100, 2),
                "avg_latency_s": round(total_latency / len(prompts), 2),
                "total_time_min": round(total_eval_wall_time / 60.0, 2),
                "energy_consumed_wh": round(energy_wh, 2),
                "emissions_co2_g": round(emissions_kg * 1000.0, 2),
                "model": str(MODEL_PATH.name)
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Benchmark results saved to: {output_path.name}")

if __name__ == "__main__":
    run_full_benchmark()
