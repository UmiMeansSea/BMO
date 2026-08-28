import json
import re
from pathlib import Path

# Load Dev Set
DEV_SET_PATH = Path(__file__).parent.parent / "data" / "past_participle_dev_set.json"

with open(DEV_SET_PATH, "r", encoding="utf-8") as f:
    dev_test_cases = json.load(f)

def run_dev_set_evaluation():
    print("=" * 70)
    print("  BMO PAST PARTICIPLE DEV SET HARNESS (10 Diagnostic Cases)")
    print("=" * 70)

    total = len(dev_test_cases)
    print(f"Loaded {total} targeted diagnostic test cases from '{DEV_SET_PATH.name}'.\n")

    for tc in dev_test_cases:
        print(f"[{tc['id']}] Category: {tc['category']}")
        print(f"  Input: \"{tc['input_sentence']}\"")
        if tc['is_erroneous']:
            print(f"  Target Correction: '{tc['expected_correction']}'")
        else:
            print(f"  Target: [Grammatically Correct Baseline]")
        print(f"  Rule: {tc['rule_explanation']}")
        print("-" * 70)

if __name__ == "__main__":
    run_dev_set_evaluation()
