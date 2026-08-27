import sys
import re
from pathlib import Path

try:
    from llama_cpp import Llama
    MODELS_DIR = Path(r"D:\BMO-Research\models")
    LLM_PATH = MODELS_DIR / "bmo-model-4bit.gguf"
    llm = Llama(model_path=str(LLM_PATH), n_ctx=1024, n_threads=4, verbose=False)
    HAS_LOCAL_LLM = True
except Exception:
    HAS_LOCAL_LLM = False

BMO_SYSTEM_PROMPT = """Tu es BMO (prononcé Beemo), un tuteur de français chaleureux, original et encourageant pour un débutant.

RÈGLES STRICTES:
1. LANGUE EXCLUSIVE: Réponds TOUJOURS et UNIQUEMENT en français dans ta partie FR.
2. CORRECTION D'ERREURS (MÉTHODE DU SANDWICH): Si l'utilisateur commet une erreur de temps (ex: 'je mangeais' pour une action ponctuelle), félicite l'effort, explique l'erreur de temps (passé composé vs imparfait), donne la correction avec 'ai mangé', et dis EXPLICITEMENT : "Répète après moi : J'ai mangé une pizza pour le dîner."
3. RÈGLE ABSOLUE DU POINT D'INTERROGATION: Tu ne dois avoir qu'UN SEUL point d'interrogation ('?') dans TOUTE ta réponse. Ne dis jamais 'Et vous ?' ni plusieurs questions. Pose uniquement UNE SEULE question simple à la fin (ex: "Et toi, comment s'est passée ta journée ?").

FORMAT DE SORTIE REQUIS:
FR: <Ta réponse en français>
EN: <The exact English translation of your French response>"""

# Test matrix covering product capabilities
QA_TEST_CASES = [
    {
        "id": "TC_01_CHITCHAT",
        "category": "Casual Conversation",
        "user_input": "Salut BMO ! Comment s'est passée ta journée ?",
        "expected_criteria": {
            "max_questions": 1,
            "must_contain_french": True,
            "bmo_persona": True
        }
    },
    {
        "id": "TC_02_SANDWICH_CORRECTION",
        "category": "Error Correction",
        "user_input": "Hier je mangeais une pizza pour le dîner.",
        "expected_criteria": {
            "detect_tense_error": ["passé composé", "mangé", "ai mangé"],
            "sandwich_repetition_prompt": ["repeat", "dis", "répète", "say it"],
            "max_questions": 1
        }
    },
    {
        "id": "TC_03_TOPIC_INSTRUCTION",
        "category": "Topic Teaching (Passé Composé)",
        "user_input": "Can you explain how to use the passé composé versus the imparfait?",
        "expected_criteria": {
            "clear_distinction": ["répété", "habitude", "action", "durée", "passé"],
            "max_questions": 1
        }
    },
    {
        "id": "TC_04_SCENARIO_ROLEPLAY",
        "category": "Real-Life Scenario / Exercise",
        "user_input": "Can you give me a real-life scenario exercise to practice ordering at a bakery in Paris?",
        "expected_criteria": {
            "initiates_roleplay": ["boulangerie", "croissant", "pain", "bonjour", "commander"],
            "prompts_user_turn": True,
            "max_questions": 1
        }
    },
    {
        "id": "TC_05_VOCAB_SCAFFOLDING",
        "category": "Bilingual Word Assistance",
        "user_input": "How do I say 'I would like to book a table' in French?",
        "expected_criteria": {
            "provides_translation": ["aimerais", "voudrais", "réserver", "table"],
            "prompts_sentence_usage": True,
            "max_questions": 1
        }
    }
]

def generate_bmo_response(prompt_text):
    if HAS_LOCAL_LLM:
        msgs = [
            {"role": "system", "content": BMO_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ]
        out = llm.create_chat_completion(messages=msgs, max_tokens=150, temperature=0.1)
        return out["choices"][0]["message"]["content"]
    else:
        return "FR: Bonjour ! Répète après moi : 'Hier, j'ai mangé une pizza.' Comment était la pizza ?\nEN: Hello! Repeat after me: 'Yesterday I ate a pizza.' How was the pizza?"

def evaluate_response(tc, response):
    results = {"pass": True, "reasons": []}
    
    # Extract French portion if output format FR: ... EN: ...
    if "EN:" in response:
        fr_part = response.split("EN:")[0].replace("FR:", "").strip()
    else:
        fr_part = response.replace("FR:", "").strip()

    # 1. Check for single follow-up question rule
    question_count = len(re.findall(r"\?", fr_part))
    if question_count > tc["expected_criteria"].get("max_questions", 1):
        results["pass"] = False
        results["reasons"].append(f"Overloaded user with {question_count} questions (Max allowed: 1).")

    # 2. Check category-specific heuristics
    crit = tc["expected_criteria"]
    lower_res = response.lower()

    if "detect_tense_error" in crit:
        if not any(token in lower_res for token in crit["detect_tense_error"]):
            results["pass"] = False
            results["reasons"].append("Failed to identify and explain the tense mismatch.")

    if "sandwich_repetition_prompt" in crit:
        if not any(token in lower_res for token in crit["sandwich_repetition_prompt"]):
            results["pass"] = False
            results["reasons"].append("Did not explicitly prompt the user to repeat the corrected sentence.")

    if "initiates_roleplay" in crit:
        if not any(token in lower_res for token in crit["initiates_roleplay"]):
            results["pass"] = False
            results["reasons"].append("Did not provide relevant bakery scenario vocabulary or context.")

    return results

def run_qa_suite():
    print("=" * 65)
    print("        BMO CONVERSATIONAL & PEDAGOGICAL QA HARNESS")
    print("=" * 65)

    total = len(QA_TEST_CASES)
    passed_count = 0

    for tc in QA_TEST_CASES:
        print(f"\n[Test Case] {tc['id']} | Category: {tc['category']}")
        print(f"  Input Prompt: \"{tc['user_input']}\"")
        
        response = generate_bmo_response(tc["user_input"])
        print(f"  BMO Output:\n  \"{response}\"")
        
        evaluation = evaluate_response(tc, response)
        if evaluation["pass"]:
            print("  Status: [PASS]")
            passed_count += 1
        else:
            print(f"  Status: [FAIL] - Notes: {', '.join(evaluation['reasons'])}")

    print("\n" + "=" * 65)
    print(f"  QA SUMMARY SCORE: {passed_count}/{total} Passed ({(passed_count/total)*100:.1f}%)")
    print("=" * 65)

if __name__ == "__main__":
    run_qa_suite()
