import json
import time
import sys
from pathlib import Path
from llama_cpp import Llama

_POINTER = Path(__file__).parent.parent / "src" / ".model_path"
MODEL_PATH = Path(_POINTER.read_text(encoding="utf-8").strip()) if _POINTER.exists() \
                else Path(r"D:\BMO-Research\models\bmo-model-3b-4bit.gguf")

DEV_SET_PATH = Path(__file__).parent.parent / "data" / "past_participle_dev_set.json"

BASELINE_SYSTEM_PROMPT = """You are BMO, a warm, quirky, and supportive French language tutor. You are speaking to an A2/B1 beginner.

Your task is to analyze the user's French sentence, detect any grammatical or conjugation errors, and provide pedagogical feedback. 

You must ALWAYS respond with a valid JSON object matching this exact schema:
{
  "grammatical_analysis": "In 1 concise sentence, evaluate the subject, verb tenses, and agreement rules.",
  "has_error": true/false,
  "error_type": "Passé composé vs Imparfait" | "Gender agreement" | "Future tense" | "Past Participle Agreement" | "None",
  "feedback": "string"
}
"""

def main():
    print("=" * 80)
    print("  STEP 1: BASELINE DEV SET INFERENCE ON BMO 3B GGUF MODEL")
    print("=" * 80)
    print(f"Model Path: {MODEL_PATH}")

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False
    )

    with open(DEV_SET_PATH, "r", encoding="utf-8") as f:
        dev_cases = json.load(f)

    baseline_results = []
    passed = 0

    print("\nRunning baseline evaluation on 10 Dev Set cases...")
    print("-" * 80)

    for item in dev_cases:
        messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": item["input_sentence"]}
        ]

        t0 = time.perf_counter()
        res = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        latency = time.perf_counter() - t0

        raw_output = res["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(raw_output)
            model_has_err = parsed.get("has_error", False)
            model_err_type = parsed.get("error_type", "None")
            ground_truth_has_err = item["is_erroneous"]

            # Evaluate pass/fail
            is_pass = (model_has_err == ground_truth_has_err)
            if ground_truth_has_err and is_pass:
                # Also verify error_type alignment if erroneous
                if "Past Participle" not in str(model_err_type) and "Gender" not in str(model_err_type):
                    is_pass = False

            if is_pass:
                passed += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"

            baseline_results.append({
                "id": item["id"],
                "category": item["category"],
                "input": item["input_sentence"],
                "ground_truth_has_error": ground_truth_has_err,
                "parsed": parsed,
                "status": status,
                "latency_s": round(latency, 2)
            })

            print(f"[{item['id']}] {status} | Latency: {latency:.2f}s")
            print(f"  Input   : \"{item['input_sentence']}\"")
            print(f"  Detected: has_error={model_has_err}, error_type='{model_err_type}'")
            print(f"  Analysis: {parsed.get('grammatical_analysis', '')}")
            print("-" * 80)
            sys.stdout.flush()

        except json.JSONDecodeError:
            print(f"[{item['id']}] [FAIL - JSON DECODE ERROR] | Latency: {latency:.2f}s")
            baseline_results.append({
                "id": item["id"],
                "status": "[FAIL_DECODE]",
                "raw_output": raw_output
            })

    print(f"\nBaseline Score: {passed}/10 Passed ({(passed/10)*100:.1f}%)")

    # Save output for diagnostic analysis
    out_path = Path(__file__).parent.parent / "results" / "baseline_dev_set_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)
    print(f"Baseline results saved to: {out_path.name}")

if __name__ == "__main__":
    main()
