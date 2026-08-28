import sys
import os
import time
import re
import json
import random
import threading
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import sounddevice as sd
import webview

try:
    import whisper
    from llama_cpp import Llama
    from kokoro_onnx import Kokoro
except ImportError as e:
    print(f"[!] Missing dependency: {e}")
    sys.exit(1)

def normalize_bmo_name(text: str) -> str:
    """Normalize any spoken phonetic variation of BMO's name to 'BMO'."""
    pattern = re.compile(r"\b(beemo|bemo|bimo|bi\s+mo|b\.m\.o\.|beemow|bémos|beemoo)\b", re.IGNORECASE)
    return pattern.sub("BMO", text)

# --- MODELS SETUP ---
# NOTE: earlier versions hardcoded exact filenames ("bmo-model-4bit.gguf")
# that didn't always match what's actually on disk (bmo_live.py uses
# "bmo-model-3b-4bit.gguf"). A filename mismatch here makes Llama(...) throw
# FileNotFoundError below, has_llm silently becomes False, and every "reply"
# you see is actually the hardcoded no-LLM fallback string further down.
# This version tries the known names first, then falls back to globbing for
# ANY .gguf file in the models directory so a rename doesn't nuke the app.
BASE_DIR = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
LOCAL_MODELS_DIR = BASE_DIR / "models"
KNOWN_MODEL_NAMES = ["Qwen2.5-7B-Instruct-Q4_K_M.gguf", "qwen2.5-7b-instruct-q4_k_m.gguf", "bmo-model-3b-4bit.gguf", "bmo-model-4bit.gguf"]
FALLBACK_MODELS_DIR = Path(r"D:\BMO-Research\models")

def _resolve_model_path() -> Path:
    for models_dir in (LOCAL_MODELS_DIR, FALLBACK_MODELS_DIR):
        if not models_dir.exists():
            continue
        for name in KNOWN_MODEL_NAMES:
            candidate = models_dir / name
            if candidate.exists():
                return candidate
        # Nothing with a known name — grab any .gguf so a rename doesn't break us
        found = sorted(models_dir.glob("*.gguf"))
        if found:
            return found[0]
    # Nothing found anywhere; return the most likely intended path so the
    # resulting FileNotFoundError message is at least informative.
    return FALLBACK_MODELS_DIR / KNOWN_MODEL_NAMES[0]

LLM_PATH = _resolve_model_path()
MODELS_DIR = LLM_PATH.parent
KOKORO_MODEL = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"

print(f"[*] Initializing BMO Native Desktop Engine using models from: {MODELS_DIR}")
whisper_model = whisper.load_model("small")

LLM_LOAD_ERROR = None
try:
    llm = Llama(model_path=str(LLM_PATH), n_ctx=2048, n_threads=4, verbose=False)
    has_llm = True
    print(f"[OK] LLM loaded from: {LLM_PATH}")
except Exception as e:
    print(f"[!] LLM Load Notice: {e}")
    traceback.print_exc()  # full traceback in console — this is the real cause, check it
    LLM_LOAD_ERROR = f"{type(e).__name__}: {e}"
    has_llm = False

def _patched_create_audio(self, phonemes, voice, speed):
    tokens = np.array(self.tokenizer.tokenize(phonemes[:510]), dtype=np.int64)
    voice_style = voice[len(tokens)]
    tokens_input = [[0, *tokens, 0]]
    inputs = {
        "input_ids": tokens_input,
        "style": np.array(voice_style, dtype=np.float32),
        "speed": np.array([speed], dtype=np.float32),
    }
    audio = self.sess.run(None, inputs)[0]
    return audio, 24000

try:
    Kokoro._create_audio = _patched_create_audio
    kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    print("[OK] Kokoro ONNX initialized!")
except Exception:
    kokoro = None

# --- DYNAMIC ROLEPLAY ENGINE ---
class RoleplayEngine:
    def __init__(self):
        self.mode = "TUTOR"
        self.scenario = None
        self.character_role = None
        self.turn_count = 0
        self.max_turns = 4
        self.roleplay_history = []

    def detect_roleplay_intent(self, user_text):
        triggers = ["roleplay", "scenario", "practice ordering", "at a café", "at a restaurant", 
                    "bakery", "boulangerie", "train station", "hotel", "exercise", "jeu de rôle", "jeu de role", "commander", "commande"]
        lower_text = user_text.lower()
        
        if any(t in lower_text for t in triggers):
            if "café" in lower_text or "coffee" in lower_text or "cafe" in lower_text:
                return "a Parisian Café", "a friendly French barista"
            elif "bakery" in lower_text or "boulangerie" in lower_text or "croissant" in lower_text or "pain" in lower_text:
                return "a traditional French Boulangerie", "a busy local baker"
            elif "train" in lower_text or "gare" in lower_text or "ticket" in lower_text or "billet" in lower_text:
                return "Gare de Lyon Train Station", "an SNCF ticket agent"
            elif "hotel" in lower_text or "room" in lower_text or "chambre" in lower_text:
                return "a Boutique Hotel in Nice", "the hotel receptionist"
            else:
                return "a French Store", "the store shopkeeper"
        return None, None

    def get_system_prompt(self, user_name):
        if self.mode == "ROLEPLAY":
            return f"""Tu es BMO, et tu joues le rôle de {self.character_role} dans ce scénario : {self.scenario}. Tu parles à {user_name}.

RÈGLES DU JEU DE RÔLE :
1. Reste strictement dans ton personnage ({self.character_role}).
2. Parle un français naturel et simple (niveau A2/B1). Fais des réponses courtes (1 à 2 phrases).
3. Termine TOUJOURS par une relance ou question en personnage.

FORMAT REQUIS :
ANALYSE: ACT={self.character_role} | SITUATION={self.scenario}
FR: <Ta réponse en français dans ton rôle>
EN: <The exact English translation>"""
        return ""

    def build_debrief_prompt(self):
        transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.roleplay_history])
        return f"""Le jeu de rôle dans {self.scenario} est terminé.
Voici la conversation :
{transcript}

TÂCHE : Donne un bilan court et encourageant (3 phrases max) :
1. Félicitations pour la participation.
2. Un point de grammaire/vocabulaire observé.
3. Deux expressions utiles à retenir.

FORMAT REQUIS :
ANALYSE: DEBRIEF=terminé
FR: <Ton bilan en français>
EN: <The exact English translation>"""

# --- ADAPTIVE SCAFFOLDING ENGINE ---
class ScaffoldingEngine:
    def __init__(self):
        self.hint_level = 0
        self.last_assisted_target = None

    def detect_scaffolding_request(self, text):
        lower = text.lower().strip()
        translation_match = re.search(r"(?:how do (?:i|you) say|how to say|what is the word for|comment dit-on)\s+['\"]?(.+?)['\"]?\??$", lower)
        if translation_match:
            target = translation_match.group(1).strip()
            return "TRANSLATION", target

        hint_triggers = ["hint", "i'm stuck", "i don't know", "help me", "aide-moi", "give me a clue", "je ne sais pas"]
        if any(t in lower for t in hint_triggers):
            return "HINT", None

        if lower in ["euh", "je...", "uh", "um", "i want", "je veux..."] or len(lower.split()) <= 1:
            return "HESITATION", None

        return "NONE", None

    def generate_hint_prompt(self, request_type, target):
        if request_type == "TRANSLATION":
            return f"""L'étudiant demande comment dire '{target}' en français.
FORMAT REQUIS :
ANALYSE: TYPE=traduction | MOT={target}
FR: <Donne la traduction et une amorce de phrase courte en français>
EN: <The exact English translation>"""
        elif request_type in ["HINT", "HESITATION"]:
            return """L'étudiant hésite ou demande un indice.
FORMAT REQUIS :
ANALYSE: TYPE=indice
FR: <Donne un indice encourageant ou une amorce (ex: 'Tu peux commencer par : Je voudrais...')>
EN: <The exact English translation>"""
        return ""

# --- SESSION MEMORY ENGINE ---
class SessionMemoryEngine:
    def __init__(self, filepath=None):
        bmo_dir = Path.home() / ".bmo"
        bmo_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = bmo_dir / "session_review.json" if filepath is None else Path(filepath)
        self.past_data = self.load_history()
        
        self.user_name = self.past_data.get("user_name", None)
        self.hobbies = self.past_data.get("hobbies", [])
        
        self.current_session = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_turns": 0,
            "roleplays_completed": [],
            "corrections_made": [],
            "new_vocabulary": []
        }

    def load_history(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_name(self, name):
        self.user_name = name
        self.past_data["user_name"] = name
        self.save_raw()

    def detect_and_save_hobby(self, user_text):
        pattern = r"(?:j'aime|j'adore|je joue|ma passion est|mon passe-temps est)\s+(?:à|de|de la|du|le|la|les|un|une)?\s*([a-zA-ZÀ-ÿ\s]+)"
        match = re.search(pattern, user_text, re.IGNORECASE)
        if match:
            hobby = match.group(1).strip().lower()
            if hobby and len(hobby) > 2 and hobby not in self.hobbies:
                self.hobbies.append(hobby)
                self.past_data["hobbies"] = self.hobbies
                self.save_raw()
                return True
        return False

    def log_turn(self, user_text, bmo_text):
        self.current_session["total_turns"] += 1
        self.detect_and_save_hobby(user_text)
        self.log_correction(bmo_text)

    def log_roleplay(self, scenario):
        if scenario not in self.current_session["roleplays_completed"]:
            self.current_session["roleplays_completed"].append(scenario)

    def log_correction(self, bmo_response):
        lower_resp = bmo_response.lower()
        topics = {
            "passé composé": ["passé composé", "past tense", "conjugaison"],
            "imparfait": ["imparfait", "habitual past"],
            "gender agreement": ["masculine", "feminine", "genre", "accord"],
            "participe passé": ["participe passé", "accordé", "invariable", "cod"]
        }
        for topic, keywords in topics.items():
            if any(k in lower_resp for k in keywords) and topic not in self.current_session["corrections_made"]:
                self.current_session["corrections_made"].append(topic)

    def save_raw(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.past_data, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

    def save_session(self, conversation_history, llm_instance=None):
        if llm_instance and len(conversation_history) > 2:
            extraction_prompt = "Examine cette conversation et liste 3 mots de vocabulaire français utiles appris. Format: mot1, mot2, mot3."
            msgs = [{"role": "system", "content": extraction_prompt}] + conversation_history[-6:]
            try:
                vocab_res = llm_instance.create_chat_completion(messages=msgs, max_tokens=30, temperature=0.1)
                vocab_list = vocab_res["choices"][0]["message"]["content"].split(",")
                self.current_session["new_vocabulary"] = [v.strip() for v in vocab_list if v.strip()]
            except Exception as e:
                print(f"Vocab extraction notice: {e}")

        self.past_data["hobbies"] = self.hobbies
        self.past_data["last_session"] = self.current_session
        if "all_time_weak_points" not in self.past_data:
            self.past_data["all_time_weak_points"] = {}
        for topic in self.current_session["corrections_made"]:
            self.past_data["all_time_weak_points"][topic] = self.past_data["all_time_weak_points"].get(topic, 0) + 1

        self.save_raw()

# --- HARDWARE AUDIO RECORDER ---
AUDIO_BUFFER = []
IS_RECORDING = False
BUFFER_LOCK = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    if IS_RECORDING:
        with BUFFER_LOCK:
            AUDIO_BUFFER.append(indata.copy())

try:
    audio_stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=audio_callback)
    audio_stream.start()
    print("[OK] Native Hardware Microphone Stream Active!")
except Exception as e:
    print(f"[CRITICAL] Failed to access microphone: {e}")

def clean_hallucinated_text(text: str) -> str:
    if not text:
        return ""
    sentences = [s.strip() for s in text.replace(".", ".|").replace("!", "!|").replace("?", "?|").split("|") if s.strip()]
    if len(sentences) > 2:
        seen = []
        for s in sentences:
            if s in seen:
                break
            seen.append(s)
        text = " ".join(seen)
    
    words = text.split()
    if len(words) > 6:
        cleaned_words = []
        for i, w in enumerate(words):
            if i >= 4 and words[i-4:i] == words[i-8:i-4]:
                break
            cleaned_words.append(w)
        text = " ".join(cleaned_words)
    return text.strip()

def sanitize_for_tts(text: str) -> str:
    if not text:
        return ""
    clean = re.sub(r'[\*\_\#`~\[\]\(\)]', '', text)
    clean = re.sub(r'[\U00010000-\U0010ffff]', '', clean, flags=re.UNICODE)
    return clean.strip()

def prune_history_by_length(history, max_chars=1200):
    total = 0
    pruned = []
    for msg in reversed(history):
        msg_len = len(msg.get("content", ""))
        if total + msg_len > max_chars:
            break
        pruned.insert(0, msg)
        total += msg_len
    return pruned

# --- CORE BMO BRIDGE CONTROLLER ---
class BmoBridge:
    def __init__(self):
        self.history = []
        self.memory = SessionMemoryEngine()
        self.roleplay = RoleplayEngine()
        self.scaffolding = ScaffoldingEngine()

    def toggle_record(self):
        global IS_RECORDING, AUDIO_BUFFER
        if not IS_RECORDING:
            AUDIO_BUFFER.clear()
            IS_RECORDING = True
            return {"status": "listening"}
        else:
            IS_RECORDING = False
            threading.Thread(target=self.process_pipeline).start()
            return {"status": "thinking"}

    def process_pipeline(self):
        try:
            with BUFFER_LOCK:
                if not AUDIO_BUFFER:
                    print("[WARN] Audio buffer empty.")
                    window.evaluate_js("window.setBMOState('idle');")
                    return

                audio_data = np.concatenate(AUDIO_BUFFER, axis=0).flatten()
                AUDIO_BUFFER.clear()

            # 1. RMS VAD Guard
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 0.005 or len(audio_data) < 16000 * 0.3:
                print(f"[Audio Guard] Dropped audio (RMS={rms:.5f}, Length={len(audio_data)/16000:.2f}s).")
                window.evaluate_js("window.setBMOState('idle');")
                return

            # 2. Whisper ASR with Priming
            current_name = self.memory.user_name if self.memory.user_name else "Umi"
            domain_primer = f"BMO, {current_name}, français, passé composé, imparfait, futur simple, grammaire, bonjour."
            result = whisper_model.transcribe(
                audio_data, 
                fp16=False, 
                language="fr",
                initial_prompt=domain_primer,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False
            )
            raw_text = result.get("text", "").strip() or "Bonjour BMO!"
            raw_text = clean_hallucinated_text(raw_text)
            user_text = normalize_bmo_name(raw_text)
            print(f"[ASR Transcript] {user_text}")

            # 3. Exit Intent Detection
            exit_triggers = ["goodbye", "au revoir", "see you later", "stop session", "à bientôt", "a bientot"]
            if any(t in user_text.lower() for t in exit_triggers):
                bmo_fr = f"Au revoir {current_name} ! J'ai enregistré nos progrès dans ton journal. À bientôt !"
                bmo_en = f"Goodbye {current_name}! I saved our progress in your journal. See you soon!"
                self.memory.save_session(self.history, llm if has_llm else None)
                
                self.history.append({"role": "user", "content": user_text})
                window.evaluate_js(f"window.appendChatMessage('user', {repr(user_text)});")
                self.history.append({"role": "assistant", "content": bmo_fr})
                window.evaluate_js(f"window.appendChatMessage('bot', {repr(bmo_fr)}, {repr(bmo_en)});")
                
                self._speak(bmo_fr, speed=0.85)
                return

            # 4. Name Interception & Identity Questions
            intercepted_bmo_text = None
            lower_user_text = user_text.lower()
            bmo_name_triggers = ["comment tu t'appelles", "comment vous appelez vous", "quel est ton nom", "tu t'appelles comment", "what is your name"]
            
            if any(trigger in lower_user_text for trigger in bmo_name_triggers):
                intercepted_bmo_text = f"FR: Je m'appelle BMO ! Je suis ton tuteur de français. Qu'aimerais-tu pratiquer aujourd'hui, {current_name} ?\nEN: My name is BMO! I am your French tutor. What would you like to practice today, {current_name}?"
            elif hasattr(self.memory, 'pending_name_change') and self.memory.pending_name_change:
                if "yes" in lower_user_text or "oui" in lower_user_text:
                    self.memory.save_name(self.memory.pending_name_change)
                    current_name = self.memory.user_name
                    intercepted_bmo_text = f"FR: D'accord, {current_name} ! C'est noté. Que veux-tu faire aujourd'hui ?\nEN: Okay, {current_name}! Duly noted. What would you like to do today?"
                else:
                    intercepted_bmo_text = f"FR: D'accord, pas de souci. Je continuerai à t'appeler {current_name}.\nEN: Okay, no problem. I'll continue calling you {current_name}."
                self.memory.pending_name_change = None
            else:
                clean_text = user_text.replace(".", "").replace("!", "").replace("?", "").lower()
                name_match = re.search(r"(?:je\s*m'appelle|j'mapelle|j'm'appelle|m'appelle)\s+([a-zA-ZÀ-ÿ\s]+)", clean_text, re.IGNORECASE)
                if name_match:
                    new_name = name_match.group(1).strip().title()
                    if self.memory.user_name and self.memory.user_name.lower() != new_name.lower():
                        intercepted_bmo_text = f"FR: Tu as changé de nom ? Veux-tu que je t'appelle {new_name} à partir de maintenant ?\nEN: Did you change your name? Do you want me to call you {new_name} from now on?"
                        self.memory.pending_name_change = new_name
                    else:
                        self.memory.save_name(new_name)
                        current_name = new_name
                        intercepted_bmo_text = f"FR: Bonjour {new_name} ! C'est noté. Comment vas-tu ?\nEN: Hello {new_name}! Noted. How are you?"

            # 5. Build Dense Micro-CoT System Prompt
            hobby_context = f"\nCentres d'intérêt de l'étudiant: {', '.join(self.memory.hobbies)}." if self.memory.hobbies else ""
            
            # --- DENSE MICRO-COT PROMPT ---
            system_instruction = (
                f"Tu es BMO, tuteur de français amical pour {current_name}. "
                "RÈGLES STRICTES :\n"
                "1. Parle EXCLUSIVEMENT en français simple (A2/B1). N'utilise jamais l'anglais dans FR.\n"
                "2. INTERDICTION DES SALUTATIONS RÉPÉTÉES : Ne redis jamais 'Bonjour' après le premier message.\n"
                "3. PAS DE PERROQUET : Ne répète jamais la phrase de l'étudiant mot pour mot.\n"
                "4. DENSE MICRO-COT & GRAMMAIRE : Avant de répondre, analyse rapidement les règles de grammaire (participe passé: avec être=accord sujet, avec avoir=accord avec COD placé AVANT seulement, invariable avec en ou COD après). Corrige toute erreur selon la méthode sandwich.\n"
                "5. Termine TOUJOURS par une seule question simple pour faire avancer la conversation.\n\n"
                "FORMAT DE SORTIE STRICT :\n"
                "ANALYSE: <AUX:avoir/être/aucun | COD:avant/après/aucun | ACCORD:oui/non/règle>\n"
                "FR: <Ta réponse ou correction en français>\n"
                "EN: <The exact English translation>"
            ) + hobby_context

            current_tts_speed = 0.85

            if intercepted_bmo_text:
                full_resp = intercepted_bmo_text
            else:
                scaffold_type, target = self.scaffolding.detect_scaffolding_request(user_text)
                recent_history = prune_history_by_length(self.history, max_chars=1000)

                if scaffold_type != "NONE":
                    current_tts_speed = 0.70
                    scaffold_prompt = system_instruction + "\n" + self.scaffolding.generate_hint_prompt(scaffold_type, target)
                    llm_msgs = [{"role": "system", "content": scaffold_prompt}] + recent_history + [{"role": "user", "content": user_text}]
                elif self.roleplay.mode == "ROLEPLAY":
                    self.roleplay.turn_count += 1
                    self.roleplay.roleplay_history.append({"role": "Student", "content": user_text})
                    if self.roleplay.turn_count > self.roleplay.max_turns or any(w in user_text.lower() for w in ["stop", "finish", "done", "quitter", "terminer"]):
                        current_tts_speed = 0.75
                        debrief_query = self.roleplay.build_debrief_prompt()
                        llm_msgs = [{"role": "system", "content": system_instruction}, {"role": "user", "content": debrief_query}]
                        self.roleplay.mode = "TUTOR"
                    else:
                        llm_msgs = [{"role": "system", "content": self.roleplay.get_system_prompt(current_name)}] + recent_history + [{"role": "user", "content": user_text}]
                else:
                    scenario, role = self.roleplay.detect_roleplay_intent(user_text)
                    if scenario:
                        self.roleplay.mode = "ROLEPLAY"
                        self.roleplay.scenario = scenario
                        self.roleplay.character_role = role
                        self.roleplay.turn_count = 0
                        self.roleplay.roleplay_history = []
                        self.memory.log_roleplay(scenario)
                        llm_msgs = [{"role": "system", "content": self.roleplay.get_system_prompt(current_name)}] + recent_history + [{"role": "user", "content": user_text}]
                    else:
                        llm_msgs = [{"role": "system", "content": system_instruction}] + recent_history + [{"role": "user", "content": user_text}]

                # 6. LLM Generation with Micro-CoT
                print("[*] Generating LLM response (Micro-CoT active)...")
                if has_llm:
                    try:
                        full_resp = llm.create_chat_completion(
                            messages=llm_msgs, 
                            max_tokens=200,              
                            temperature=0.55,
                            repeat_penalty=1.30,
                            frequency_penalty=0.30,
                            top_p=0.85
                        )["choices"][0]["message"]["content"]
                    except Exception as e:
                        print(f"[LLM Error]: {e}")
                        full_resp = f"ANALYSE: ERR=fallback\nFR: Peux-tu répéter cela s'il te plaît, {current_name} ?\nEN: Can you repeat that please, {current_name}?"
                else:
                    full_resp = f"ANALYSE: ERR=nollm\nFR: Bonjour {current_name} ! Que veux-tu faire ?\nEN: Hello {current_name}! What do you want to do?"

            # 7. Parse Micro-CoT and Extract UI/TTS Clean Text
            if "ANALYSE:" in full_resp:
                print(f"[Micro-CoT Debug] {full_resp.split('ANALYSE:')[1].split('FR:')[0].strip()}")

            if "EN:" in full_resp:
                parts = full_resp.split("EN:")
                bmo_fr = parts[0]
                if "FR:" in bmo_fr:
                    bmo_fr = bmo_fr.split("FR:")[1].strip()
                else:
                    bmo_fr = re.sub(r'ANALYSE:.*', '', bmo_fr).strip()
                bmo_en = parts[1].strip()
            elif "FR:" in full_resp:
                bmo_fr = full_resp.split("FR:")[1].strip()
                bmo_en = ""
            else:
                bmo_fr = re.sub(r'ANALYSE:.*', '', full_resp).strip()
                bmo_en = ""

            # --- Anti-Loop Guardrail ---
            # Check against the last THREE bot messages, not just one — a two-line
            # oscillation (A, B, A, B, ...) still counts as stuck, and only checking
            # the single previous message lets it slip through every other turn.
            recent_bot_msgs = [m["content"].strip().lower() for m in reversed(self.history) if m["role"] == "assistant"][:3]
            loop_triggers = ["je vais bien, merci", "comment ça va pour toi", "comment allez-vous"]
            is_looping = (
                any(t in bmo_fr.lower() for t in loop_triggers)
                or bmo_fr.strip().lower() in recent_bot_msgs
            )
            if is_looping:
                print(f"[Anti-Loop] Caught loop phrase. Forcing contextual pivot.")
                if "travail" in user_text.lower() or "travaille" in user_text.lower():
                    pivot_pool = [
                        (f"C'est intéressant, {current_name} ! Quel est ton métier ?", f"That's interesting, {current_name}! What is your job?"),
                        (f"Ah oui ? Et ça se passe bien pour toi en ce moment ?", f"Oh yeah? And is it going well for you right now?"),
                    ]
                else:
                    pivot_pool = [
                        (f"Je comprends bien, {current_name} ! Raconte-moi ce que tu fais d'autre aujourd'hui.", f"I understand well, {current_name}! Tell me what else you are doing today."),
                        (f"D'accord ! Et sinon, qu'est-ce que tu as prévu pour la suite ?", f"Okay! And otherwise, what do you have planned next?"),
                        (f"Je vois ! Parle-moi d'autre chose, {current_name} — tes loisirs, par exemple ?", f"I see! Tell me about something else, {current_name} — your hobbies, for example?"),
                    ]
                # Avoid picking a pivot that's itself already in the recent messages
                choices = [p for p in pivot_pool if p[0].strip().lower() not in recent_bot_msgs] or pivot_pool
                chosen_fr, chosen_en = random.choice(choices)
                bmo_fr = chosen_fr
                bmo_en = chosen_en

            if not bmo_en or bmo_en.strip().lower() == bmo_fr.strip().lower():
                bmo_en = "Translation unavailable"

            self.memory.log_turn(user_text, bmo_fr)

            bmo_fr_ui = bmo_fr.replace("Beemo", "BMO")
            bmo_en_ui = bmo_en.replace("Beemo", "BMO")

            print(f"[BMO Response] FR: {bmo_fr_ui} | EN: {bmo_en_ui}")

            # Append to history & Web UI
            self.history.append({"role": "user", "content": user_text})
            window.evaluate_js(f"window.appendChatMessage('user', {repr(user_text)});")
            
            self.history.append({"role": "assistant", "content": bmo_fr_ui})
            window.evaluate_js(f"window.appendChatMessage('bot', {repr(bmo_fr_ui)}, {repr(bmo_en_ui)});")

            # 8. Asynchronous Audio Synthesis
            self._speak(bmo_fr_ui, speed=current_tts_speed)

        except Exception as e:
            print(f"[Pipeline Error]: {e}")
            window.evaluate_js("window.setBMOState('idle');")

    def _speak(self, text, speed=0.85):
        window.evaluate_js("window.setBMOState('speaking');")
        spoken_text = sanitize_for_tts(text.replace("BMO", "Beemo"))
        
        if kokoro:
            try:
                audio, sr_out = kokoro.create(spoken_text, voice="ff_siwis", speed=speed, lang="fr-fr")
            except Exception:
                audio, sr_out = kokoro.create(spoken_text, voice="af_bella", speed=speed)
            
            audio_flat = audio.squeeze()
            pitch_factor = 1.2599
            new_len = int(len(audio_flat) / pitch_factor)
            samples_shifted = scipy.signal.resample(audio_flat, new_len).astype(np.float32)
            sd.play(samples_shifted, sr_out)
            sd.wait()
        else:
            from gtts import gTTS
            import soundfile as sf
            tts = gTTS(text=text, lang='fr', slow=False)
            tts.save("temp_bmo_gtts.mp3")
            samples_raw, sr_out = sf.read("temp_bmo_gtts.mp3")
            if samples_raw.ndim > 1:
                samples_raw = samples_raw.mean(axis=1)
            pitch_factor = 1.2599
            new_len = int(len(samples_raw) / pitch_factor)
            samples_shifted = scipy.signal.resample(samples_raw, new_len).astype(np.float32)
            sd.play(samples_shifted, sr_out)
            sd.wait()

        window.evaluate_js("window.setBMOState('idle');")

bridge = BmoBridge()

# --- HTML / CSS / JS FRONTEND ---
html_content = """
<!DOCTYPE html>
<html>
<head>
<style>
body { background-color: #122821; font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; overflow: hidden; }
#main-layout { display: flex; gap: 30px; align-items: center; }

/* BMO Chassis */
#bmo-container { background-color: #3ca993; width: 400px; height: 520px; border: 4px solid #000; border-radius: 20px; position: relative; }
.bmo-screen { background-color: #b7efcc; width: 350px; height: 220px; border: 4px solid #000; border-radius: 15px; position: absolute; top: 20px; left: 21px; overflow: hidden; transition: background 0.2s; }

.bmo-eye { background: #000; width: 16px; height: 16px; border-radius: 50%; position: absolute; top: 35%; animation: blink 4s infinite; }
.bmo-eye.left { left: 25%; } .bmo-eye.right { right: 25%; }
@keyframes blink { 0%, 96%, 98%, 100% { transform: scaleY(1); } 97% { transform: scaleY(0.1); } }

.bmo-mouth { position: absolute; top: 50%; left: 50%; transform: translateX(-50%); width: 65px; height: 32px; border: 4px solid #000; border-top: transparent; border-left: transparent; border-right: transparent; border-radius: 0 0 50px 50px; }
.bmo-mouth.speaking { animation: talk-anim 0.22s infinite alternate ease-in-out; background: #112a20; border: 3px solid #000; }
@keyframes talk-anim { 0% { height: 12px; width: 35px; top: 58%; } 100% { height: 38px; width: 52px; top: 50%; } }

.bmo-waveform { display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 220px; height: 80px; align-items: center; justify-content: space-between; }
.bmo-screen.listening { background-color: #0b1f17; }
.bmo-screen.listening .bmo-eye, .bmo-screen.listening .bmo-mouth { display: none; }
.bmo-screen.listening .bmo-waveform { display: flex; }
.wave-bar { width: 10px; background: #33ff99; border-radius: 5px; animation: wave-anim 0.6s infinite alternate ease-in-out; }
.wave-bar:nth-child(odd) { height: 30px; animation-delay: 0.1s; } .wave-bar:nth-child(even) { height: 60px; animation-delay: 0.3s; }
@keyframes wave-anim { 0% { transform: scaleY(0.3); } 100% { transform: scaleY(1.3); background: #66ffff; } }

.bmo-thinking-box { display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); flex-direction: column; align-items: center; }
.bmo-screen.thinking { background-color: #1a4237; }
.bmo-screen.thinking .bmo-eye, .bmo-screen.thinking .bmo-mouth, .bmo-screen.thinking .bmo-waveform { display: none; }
.bmo-screen.thinking .bmo-thinking-box { display: flex; color: #ffcc00; font-family: monospace; font-weight: bold; }

.bmo-slot { position: absolute; background: #112a20; border: 4px solid #000; width: 200px; height: 15px; top: 260px; left: 30px; }
.bmo-sbc { position: absolute; background: #0000ff; border: 4px solid #000; width: 20px; height: 20px; border-radius: 50%; top: 255px; right: 50px; }
.bmo-dpad-svg { position: absolute; top: 310px; left: 30px; width: 100px; height: 100px; }
.bmo-triangle-svg { position: absolute; top: 310px; right: 110px; width: 40px; height: 40px; }
.bmo-gc { position: absolute; background: #33ff33; border: 4px solid #000; width: 25px; height: 25px; border-radius: 50%; top: 325px; right: 50px; }
.bmo-rc { position: absolute; background: #ff0000; border: 4px solid #000; width: 60px; height: 60px; border-radius: 50%; bottom: 60px; right: 50px; cursor: pointer; transition: transform 0.1s; }
.bmo-rc:active { transform: scale(0.92); }
.bmo-pill { position: absolute; background: #0000ff; border: 4px solid #000; width: 45px; height: 15px; border-radius: 15px; bottom: 25px; } .bmo-pill.p1 { left: 40px; } .bmo-pill.p2 { left: 105px; }

/* Chat Box */
#chat-container { width: 400px; height: 520px; background-color: #f4fce8; border: 4px solid #000; border-radius: 20px; display: flex; flex-direction: column; padding: 15px; box-sizing: border-box; overflow-y: auto; gap: 12px; }
.message { padding: 12px; border-radius: 15px; font-size: 15px; max-width: 82%; line-height: 1.4; color: #1a1a1a; position: relative; box-sizing: border-box; }
.message.user { background-color: #fff9d2; border: 2px solid #fbe490; align-self: flex-end; border-bottom-right-radius: 2px; }
.message.bot { background-color: #d7f4a5; border: 2px solid #bce27f; align-self: flex-start; border-bottom-left-radius: 2px; padding-bottom: 30px; }

.translate-btn-small { 
    position: absolute; 
    bottom: 6px; 
    right: 8px; 
    background: linear-gradient(180deg, #ffffff 0%, #ecfdf5 100%); 
    border: 2px solid #059669; 
    color: #065f46; 
    border-radius: 12px; 
    padding: 3px 9px; 
    font-size: 11px; 
    font-weight: 800; 
    cursor: pointer; 
    transition: all 0.15s ease-in-out; 
    box-shadow: 0 2px 0 #047857; 
}
.translate-btn-small:hover { 
    background: linear-gradient(180deg, #fef08a 0%, #fde047 100%); 
    border-color: #ca8a04; 
    color: #713f12; 
    box-shadow: 0 3px 0 #854d0e; 
    transform: translateY(-1px); 
}
.translate-btn-small:active {
    transform: translateY(1px);
    box-shadow: 0 0 0 #ca8a04;
}
.translation-box { 
    display: none; 
    margin-top: 8px; 
    padding: 8px 12px; 
    background: #eefdf4; 
    border: 2px solid #a7f3d0; 
    border-radius: 10px; 
    font-size: 13px; 
    color: #065f46; 
    font-style: italic; 
    line-height: 1.35; 
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.05); 
}
</style>
</head>
<body>

<div id="main-layout">
    <!-- BMO Chassis -->
    <div id="bmo-container">
        <div id="bmo-screen" class="bmo-screen">
            <div class="bmo-eye left"></div><div class="bmo-eye right"></div>
            <div id="bmo-mouth" class="bmo-mouth"></div>
            <div class="bmo-waveform">
                <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
            </div>
            <div class="bmo-thinking-box">
                <p>🤖 BMO is thinking...</p>
            </div>
        </div>
        <div class="bmo-slot"></div><div class="bmo-sbc"></div>
        <svg class="bmo-dpad-svg" viewBox="0 0 100 100"><path d="M 35 5 L 65 5 L 65 35 L 95 35 L 95 65 L 65 65 L 65 95 L 35 95 L 35 65 L 5 65 L 5 35 L 35 35 Z" fill="#ffcc00" stroke="#000" stroke-width="4" stroke-linejoin="round"/></svg>
        <svg class="bmo-triangle-svg" viewBox="0 0 100 100"><polygon points="50,10 90,90 10,90" fill="#00ccff" stroke="#000" stroke-width="6" stroke-linejoin="round"/></svg>
        <div class="bmo-gc"></div>
        <div class="bmo-rc" onclick="onRedButtonClicked()"></div>
        <div class="bmo-pill p1"></div><div class="bmo-pill p2"></div>
    </div>

    <!-- Pastel Chat History -->
    <div id="chat-container">
        <div class="message bot">
            <div>Bonjour ! Je suis prêt à t'aider avec ton français.</div>
            <button class="translate-btn-small" onclick="toggleTranslation(this)">🌐 Translate</button>
            <div class="translation-box">Hello! I'm ready to help you with your French.</div>
        </div>
    </div>
</div>

<script>
function playBeep(freq) {
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.frequency.setValueAtTime(freq, ctx.currentTime);
        gain.gain.setValueAtTime(0.1, ctx.currentTime);
        osc.connect(gain); gain.connect(ctx.destination);
        osc.start(); osc.stop(ctx.currentTime + 0.15);
    } catch(e) {}
}

async function onRedButtonClicked() {
    let res = await window.pywebview.api.toggle_record();
    if (res.status === 'listening') {
        playBeep(880);
        setBMOState('listening');
    } else if (res.status === 'thinking') {
        playBeep(440);
        setBMOState('thinking');
    }
}

function setBMOState(state) {
    const screen = document.getElementById('bmo-screen');
    const mouth = document.getElementById('bmo-mouth');
    screen.className = 'bmo-screen';
    mouth.classList.remove('speaking');

    if (state === 'listening') screen.classList.add('listening');
    else if (state === 'thinking') screen.classList.add('thinking');
    else if (state === 'speaking') mouth.classList.add('speaking');
}

function toggleTranslation(btn) {
    const box = btn.nextElementSibling;
    if (box.style.display === 'none' || !box.style.display) {
        box.style.display = 'block';
        btn.innerText = ' Hide';
    } else {
        box.style.display = 'none';
        btn.innerText = ' Translate';
    }
}

function appendChatMessage(sender, text, translation = '') {
    const container = document.getElementById('chat-container');
    if (sender === 'user') {
        const div = document.createElement('div');
        div.className = 'message user';
        div.innerText = text;
        container.appendChild(div);
    } else {
        const div = document.createElement('div');
        div.className = 'message bot';
        
        const textDiv = document.createElement('div');
        textDiv.innerText = text;
        div.appendChild(textDiv);

        if (!translation || translation.trim() === text.trim()) {
            translation = "(English translation unavailable)";
        }
        
        const btn = document.createElement('button');
        btn.className = 'translate-btn-small';
        btn.innerText = '🌐 Translate';
        btn.onclick = function() { toggleTranslation(btn); };
        
        const transBox = document.createElement('div');
        transBox.className = 'translation-box';
        transBox.innerText = translation;
        
        div.appendChild(btn);
        div.appendChild(transBox);
        container.appendChild(div);
    }
    container.scrollTop = container.scrollHeight;
}
</script>
</body>
</html>
"""

def _on_window_loaded():
    if not has_llm:
        warning_fr = (
            f"⚠️ Le modèle IA ne s'est pas chargé (chemin essayé : {LLM_PATH}). "
            f"BMO répond en mode dégradé avec des phrases fixes jusqu'à ce que ce soit corrigé."
        )
        warning_en = f"Model load failed ({LLM_LOAD_ERROR}). Check the console for the full traceback."
        window.evaluate_js(f"window.appendChatMessage('bot', {repr(warning_fr)}, {repr(warning_en)});")

if __name__ == "__main__":
    window = webview.create_window("BMO Live Edge Tutor", html=html_content, js_api=bridge, width=860, height=600, background_color='#122821')
    window.events.loaded += _on_window_loaded
    webview.start()