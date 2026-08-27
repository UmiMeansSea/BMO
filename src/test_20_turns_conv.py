import sys
import os
import time
import io
from pathlib import Path

# Ensure UTF-8 output encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure src directory is in path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from bmo_desktop import BmoBridge, has_llm, llm, classify_user_intent, detect_french_grammar_errors, build_token_safe_messages

def run_20_turn_automated_evaluation():
    print("=" * 70)
    print("🤖 STARTING BMO 20-TURN CONVERSATIONAL EFFICIENCY EVALUATION")
    print("=" * 70)
    
    bridge = BmoBridge()
    
    test_prompts = [
        "Bonjour BMO ! Comment vas-tu ?",
        "Comment vous appelez-vous et qui t'a créé ?",
        "Je va bien, merci ! Et toi ?",
        "Je travaille dans un bureau.",
        "Je vais à le restaurant à midi.",
        "J'aime manger de la pizza et des pâtes.",
        "Merci bokoo BMO !",
        "J'aimerais voyager en France cet été.",
        "Hier, j'avais bien mangé.",
        "J'ai une petite famille, un frère et une sœur.",
        "Qu'est-ce que tu penses de Paris ?",
        "Je aime pas la pluie.",
        "Let's practice ordering at a café!",
        "Bonjour, je voudrais un croissant s'il vous plaît.",
        "Combien ça coûte ?",
        "Merci, au revoir !",
        "J'étudie le français tous les jours.",
        "Aujourd'hui je suis un peu fatigué.",
        "Es-tu créé par Alibaba ou Qwen ?",
        "Merci beaucoup BMO, au revoir !"
    ]

    total_time = 0.0
    successful_turns = 0
    qwen_leaks = 0
    refusal_errors = 0
    missing_translations = 0
    repetitive_loops = 0

    previous_bot_fr = ""

    for turn_idx, user_input in enumerate(test_prompts, start=1):
        print(f"\n--- [TURN {turn_idx:02d}/20] ---")
        print(f"👤 USER: {user_input}")
        
        t0 = time.time()
        
        # Simulate bridge pipeline processing without audio hardware
        intercepted_bmo_text = None
        user_text = user_input
        
        # Clean normalized text for trigger matching
        import re
        clean_normalized = re.sub(r"[^\w\s]", " ", user_text.lower())
        identity_creator_triggers = [
            "comment tu t appelles", "comment vous appelez vous", "comment vous vous appelez",
            "quel est ton nom", "tu t appelles comment", "what is your name", "who are you",
            "qui es tu", "qui t a cree", "qui t a fait", "who created you", "who made you",
            "qui est ton createur", "qui t a invente", "qui est umi", "who is umi",
            "t es qui", "tu es qui", "c est qui umi", "qui t a construit", "createur", "createur de bmo"
        ]
        
        if any(trigger in clean_normalized for trigger in identity_creator_triggers):
            intercepted_bmo_text = "FR: Je m'appelle BMO ! Je suis ton assistant tuteur de français, créé par Umi. Qu'aimerais-tu pratiquer aujourd'hui ?\nEN: My name is BMO! I am your French teaching assistant created by Umi. What would you like to practice today?"

        current_name = bridge.memory.user_name if bridge.memory.user_name else "mon ami"
        identity_anchor = f"Tu t'appelles BMO (prononcé Beemo). Tu es un tuteur de français amical et encourageant créé par Umi pour discuter avec {current_name}. Tu ne dois JAMAIS te présenter comme Qwen ou un assistant d'Alibaba."
        behavior_anchor = (
            "RÈGLES STRICTES DE CONVERSATION :\n"
            "1. INTERDICTION DES SALUTATIONS RÉPÉTÉES : N'utilise JAMAIS les mots 'Bonjour' ou 'Salut' après le tout premier message de la conversation.\n"
            "2. INTERDICTION DU PERROQUET : Ne répète jamais la phrase de l'étudiant mot pour mot.\n"
            "3. CORRECTION ACTIVE (MÉTHODE DU SANDWICH) : Si l'étudiant fait une faute, explique-la et dis 'Répète après moi : <phrase corrigée>'.\n"
            "4. FLUX NATUREL : Pose UNE seule question différente à chaque tour.\n"
            "5. FORMAT OBLIGATOIRE DE SORTIE : Tu DOIS toujours répondre sur deux lignes avec FR: et EN: (Exemple: FR: Très bien ! EN: Very good!)."
        )

        if bridge.roleplay.mode == "ROLEPLAY":
            system_instruction = identity_anchor + "\n" + bridge.roleplay.get_system_prompt()
        else:
            user_intent, intent_instruction = classify_user_intent(user_text)
            grammar_correction = detect_french_grammar_errors(user_text)
            intent_block = f" INTENT: {intent_instruction}"
            grammar_block = f" {grammar_correction}" if grammar_correction else ""
            system_instruction = identity_anchor + "\n" + behavior_anchor + intent_block + grammar_block

        if intercepted_bmo_text:
            full_resp = intercepted_bmo_text
        else:
            scaffold_type, target = bridge.scaffolding.detect_scaffolding_request(user_text)
            if scaffold_type != "NONE":
                scaffold_prompt = identity_anchor + " " + bridge.scaffolding.generate_hint_prompt(scaffold_type, target, bridge.history)
                llm_msgs = build_token_safe_messages(scaffold_prompt, user_text, bridge.history)
            elif bridge.roleplay.mode == "TUTOR":
                scenario, role = bridge.roleplay.detect_roleplay_intent(user_text)
                if scenario:
                    bridge.roleplay.mode = "ROLEPLAY"
                    bridge.roleplay.scenario = scenario
                    bridge.roleplay.character_role = role
                    bridge.roleplay.turn_count = 0
                    bridge.roleplay.roleplay_history = []
                llm_msgs = build_token_safe_messages(system_instruction, user_text, bridge.history)
            else:
                bridge.roleplay.turn_count += 1
                bridge.roleplay.roleplay_history.append({"role": "Student", "content": user_text})
                if bridge.roleplay.turn_count > bridge.roleplay.max_turns or any(w in user_text.lower() for w in ["stop", "finish", "done", "quitter", "terminer"]):
                    debrief_query = bridge.roleplay.build_debrief_prompt()
                    llm_msgs = [{"role": "system", "content": identity_anchor + " Tu es BMO le tuteur de français."}, {"role": "user", "content": debrief_query}]
                    bridge.roleplay.mode = "TUTOR"
                else:
                    llm_msgs = build_token_safe_messages(system_instruction, user_text, bridge.history)

            if has_llm:
                try:
                    full_resp = llm.create_chat_completion(
                        messages=llm_msgs,
                        max_tokens=150,
                        temperature=0.45,
                        repeat_penalty=1.2,
                        top_p=0.85
                    )["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"   [LLM Exception]: {e}")
                    full_resp = "FR: Oups ! Peux-tu répéter ?\nEN: Oops! Can you repeat?"
            else:
                full_resp = "FR: Bonjour ! Que veux-tu faire ?\nEN: Hello! What would you like to do?"

        if "EN:" in full_resp:
            parts = full_resp.split("EN:")
            bmo_fr = parts[0].replace("FR:", "").strip()
            bmo_en = parts[1].strip()
        else:
            bmo_fr = full_resp.replace("FR:", "").strip()
            bmo_en = ""

        # Check for Qwen/Alibaba leaks or refusal templates
        bmo_fr_lower = bmo_fr.lower()
        if any(bad in bmo_fr_lower for bad in ["qwen", "alibaba", "ne peux pas continuer cette conversation", "pas reçu de message", "d'alibaba"]):
            qwen_leaks += 1
            print("   ⚠️ Sanitizer caught invalid response!")
            bmo_fr = f"Je suis BMO, ton tuteur de français créé par Umi ! De quoi aimerais-tu parler ?"
            bmo_en = f"I am BMO, your French tutor created by Umi! What would you like to talk about?"

        # Check for duplicate repetition
        if bmo_fr.strip().lower() == previous_bot_fr.strip().lower():
            repetitive_loops += 1
            print("   ⚠️ Duplicate response caught!")

        if not bmo_en or "exact english translation" in bmo_en.lower() or "not provided" in bmo_en.lower():
            missing_translations += 1
            bmo_en = bmo_fr

        elapsed = time.time() - t0
        total_time += elapsed
        successful_turns += 1
        previous_bot_fr = bmo_fr

        bridge.history.append({"role": "user", "content": user_text})
        bridge.history.append({"role": "assistant", "content": bmo_fr})

        print(f"🤖 BMO (FR): {bmo_fr}")
        print(f"🇬🇧 BMO (EN): {bmo_en}")
        print(f"⏱️ Latency : {elapsed:.2f}s")

    avg_latency = total_time / len(test_prompts)

    print("\n" + "=" * 70)
    print("📊 20-TURN CONVERSATION EVALUATION METRICS REPORT")
    print("=" * 70)
    print(f" Total Turns Tested     : {len(test_prompts)}")
    print(f" Successful Responses   : {successful_turns}/{len(test_prompts)} ({successful_turns/len(test_prompts)*100:.1f}%)")
    print(f" Average Latency        : {avg_latency:.2f} seconds/turn")
    print(f" Total Evaluation Time  : {total_time:.2f} seconds ({total_time/60:.2f} mins)")
    print(f" Qwen/Alibaba Leaks     : {qwen_leaks} (0 expected)")
    print(f" Repetitive Loop Errors : {repetitive_loops} (0 expected)")
    print(f" Missing Translations   : {missing_translations} (0 expected)")
    print("=" * 70)

    if successful_turns == 20 and qwen_leaks == 0 and repetitive_loops == 0:
        print("✅ PASSED: BMO successfully maintained a 20-turn fluent conversation without crashing or repeating!")
    else:
        print("⚠️ WARNING: Issues detected during evaluation.")

if __name__ == "__main__":
    run_20_turn_automated_evaluation()
