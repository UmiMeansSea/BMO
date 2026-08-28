import json
import time
import sys
import re
from pathlib import Path
from llama_cpp import Llama

MODELS_DIR = Path(r"D:\BMO-Research\models")
MODEL_PATH = MODELS_DIR / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
if not MODEL_PATH.exists():
    MODEL_PATH = MODELS_DIR / "qwen2.5-7b-instruct-q4_k_m.gguf"
if not MODEL_PATH.exists():
    _POINTER = Path(__file__).parent.parent / "src" / ".model_path"
    MODEL_PATH = Path(_POINTER.read_text(encoding="utf-8").strip()) if _POINTER.exists() \
                    else MODELS_DIR / "bmo-model-3b-4bit.gguf"

BLIND_SET_PATH = Path(__file__).parent.parent / "data" / "past_participle_blind_set.json"

DENSE_MICRO_COT_SYSTEM_PROMPT = """Tu es BMO, tuteur de français amical pour un débutant A2/B1.

RÈGLES DENSE MICRO-COT & GRAMMAIRE:
Analyse rapidement les règles de grammaire du participe passé (avec être=accord sujet, avec avoir=accord avec COD placé AVANT seulement, invariable avec en ou COD après).

FORMAT DE SORTIE STRICT:
ANALYSE: <AUX:avoir/être/aucun | COD:avant/après/aucun | ACCORD:oui/non/règle>
FR: <Ta réponse ou correction en français>
EN: <The exact English translation>
"""

def main():
    print("=" * 80)
    print("  STEP 3: DENSE MICRO-COT BLIND HOLDOUT EVALUATION (50 CASES)")
    print("=" * 80)
    print(f"Model Path: {MODEL_PATH}")

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=1024,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False
    )

    with open(BLIND_SET_PATH, "r", encoding="utf-8") as f:
        blind_cases = json.load(f)

    blind_results = []
    passed = 0
    total_lat = 0.0

    print(f"\nRunning Dense Micro-CoT evaluation on {len(blind_cases)} blind cases...")
    print("-" * 80)

    for item in blind_cases:
        messages = [
            {"role": "system", "content": DENSE_MICRO_COT_SYSTEM_PROMPT},
            {"role": "user", "content": item["input_sentence"]}
        ]

        t0 = time.perf_counter()
        res = llm.create_chat_completion(
            messages=messages,
            temperature=0.35,
            max_tokens=140,  # Dense Micro-CoT tight token budget
            repeat_penalty=1.18,
            top_p=0.9
        )
        latency = time.perf_counter() - t0
        total_lat += latency

        raw_output = res["choices"][0]["message"]["content"]
        lower_out = raw_output.lower()
        ground_truth_has_err = item["is_erroneous"]

        # Parse error detection
        model_detected_error = any(kw in lower_out for kw in [
            "accord", "faute", "erreur", "répète après moi", "doit s'accorder", 
            "invariable", "remplace par", "écrit avec", "correction", "il faut"
        ])
        if "analyse:" in lower_out:
            analyse_line = lower_out.split("analyse:")[1].split("\n")[0]
            if "non" in analyse_line or "règle" in analyse_line or "invariable" in analyse_line:
                model_detected_error = True

        is_pass = (model_detected_error == ground_truth_has_err)

        if is_pass:
            passed += 1
            status = "[PASS]"
        else:
            status = "[FAIL]"

        blind_results.append({
            "id": item["id"],
            "category": item["category"],
            "input": item["input_sentence"],
            "ground_truth_has_error": ground_truth_has_err,
            "raw_output": raw_output,
            "status": status,
            "latency_s": round(latency, 2)
        })

        print(f"[{item['id']}] {status} | Latency: {latency:.2f}s | {item['category']}")
        sys.stdout.flush()

    total = len(blind_cases)
    avg_latency = total_lat / total
    print("=" * 80)
    print(f"DENSE MICRO-COT BLIND HOLDOUT SUMMARY:")
    print(f"  Total Passed        : {passed}/{total} Passed ({(passed/total)*100:.1f}%)")
    print(f"  Average CPU Latency : {avg_latency:.2f}s per turn")
    print("=" * 80)

    out_path = Path(__file__).parent.parent / "results" / "blind_holdout_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_cases": total,
                "passed": passed,
                "pass_rate_pct": round((passed/total)*100, 2),
                "avg_latency_s": round(avg_latency, 2)
            },
            "results": blind_results
        }, f, indent=2, ensure_ascii=False)
    print(f"Blind holdout results saved to: {out_path.name}")

if __name__ == "__main__":
    main()
