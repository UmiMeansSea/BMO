import sys
import os
import time
import re
import json
import threading
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
BASE_DIR = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
LOCAL_MODELS_DIR = BASE_DIR / "models"

if LOCAL_MODELS_DIR.exists() and (LOCAL_MODELS_DIR / "bmo-model-4bit.gguf").exists():
    MODELS_DIR = LOCAL_MODELS_DIR
else:
    MODELS_DIR = Path(r"D:\BMO-Research\models")

LLM_PATH = MODELS_DIR / "bmo-model-4bit.gguf"
KOKORO_MODEL = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"

print(f"[*] Initializing BMO Native Desktop Engine using models from: {MODELS_DIR}")
whisper_model = whisper.load_model("small")

try:
    llm = Llama(model_path=str(LLM_PATH), n_ctx=512, n_threads=4, verbose=False)
    has_llm = True
except Exception:
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

BMO_SYSTEM_PROMPT = """Tu es BMO (prononcé Beemo), un tuteur de français chaleureux, original et encourageant pour un débutant.

RÈGLES STRICTES:
1. RECONNAISSANCE DE TON NOM: Ton nom est BMO (prononcé Beemo). L'utilisateur peut t'appeler BMO, Beemo, Bemo ou Bi Mo. Quand l'utilisateur prononce ton nom ou te salue avec ton nom (ex: "BMO", "Salut BMO", "Hey Beemo"), reconnais immédiatement qu'il s'adresse à toi avec enthousiasme (ex: "Oui ! C'est moi BMO !").
2. LANGUE EXCLUSIVE: Réponds TOUJOURS et UNIQUEMENT en français dans ta partie FR.
3. CORRECTION D'ERREURS (MÉTHODE DU SANDWICH): Si l'utilisateur fait une erreur de conjugaison ou de grammaire (ex: 'je mangeais' au lieu de 'j'ai mangé'), félicite l'effort, explique la faute de temps (passé composé vs imparfait), donne la bonne phrase avec 'ai mangé', et dis EXPLICITEMENT : "Répète après moi : <bonne phrase>".
4. RÈGLE ABSOLUE DU POINT D'INTERROGATION: Tu ne dois avoir qu'UN SEUL point d'interrogation ('?') dans TOUTE ta réponse. Ne dis jamais 'Et vous ?' ni plusieurs questions. Pose uniquement UNE SEULE question simple à la fin.

FORMAT DE SORTIE REQUIS:
Tu DOIS TOUJOURS fournir ta réponse sous cette forme exacte avec deux lignes :
FR: <Ta réponse en français>
EN: <The exact English translation of your French response>"""

# --- DYNAMIC ROLEPLAY ENGINE ---
class RoleplayEngine:
    """Manages conversational modes, in-character personas, and multi-turn debriefs."""
    
    def __init__(self):
        self.mode = "TUTOR"  # Modes: "TUTOR" | "ROLEPLAY"
        self.scenario = None
        self.character_role = None
        self.turn_count = 0
        self.max_turns = 4
        self.roleplay_history = []

    def detect_roleplay_intent(self, user_text):
        """Identifies scenario keywords and extracts the target setting."""
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

    def get_system_prompt(self):
        """Dynamically yields the appropriate system instructions based on active state."""
        if self.mode == "ROLEPLAY":
            return f"""Tu es BMO, et tu joues le rôle de {self.character_role} dans ce scénario : {self.scenario}.

RÈGLES DU JEU DE RÔLE:
1. Reste strictement dans ton personnage ({self.character_role}). Ne sors pas de ton rôle pour l'instant.
2. Parle un français naturel et simple (niveau A2/B1). Fais des réponses courtes (1 à 2 phrases).
3. Fais avancer la situation naturellement (demande sa commande, son choix, le paiement, etc.).
4. Termine TOUJOURS par UNE seule question ou relance en personnage.

FORMAT DE SORTIE REQUIS:
Tu DOIS TOUJOURS fournir ta réponse sous cette forme exacte avec deux lignes :
FR: <Ta réponse en français dans ton rôle>
EN: <The exact English translation of your French roleplay line>"""
        else:
            return BMO_SYSTEM_PROMPT

    def build_debrief_prompt(self):
        """Generates a structured prompt to wrap up the roleplay session."""
        transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.roleplay_history])
        return f"""Le jeu de rôle dans {self.scenario} est terminé.
Voici la conversation :
{transcript}

TÂCHE: Donne un bilan court et encourageant en français :
1. Un compliment chaleureux sur sa participation au jeu de rôle.
2. Un conseil sur une faute de grammaire ou de prononciation observée.
3. Le rappel de 2 phrases essentielles en français utilisées dans ce scénario.
Fais un bilan court (3 à 4 phrases maximum).

FORMAT DE SORTIE REQUIS:
FR: <Ton bilan en français>
EN: <The exact English translation of your French debrief>"""

# --- ADAPTIVE SCAFFOLDING ENGINE ---
class ScaffoldingEngine:
    """Manages multi-tiered hints, hesitation detection, and dynamic audio pacing."""

    def __init__(self):
        self.hint_level = 0
        self.last_assisted_target = None

    def detect_scaffolding_request(self, text):
        """Detects if the user is asking for assistance, vocabulary, or is hesitating."""
        lower = text.lower().strip()
        
        # 1. Direct translation request
        translation_match = re.search(r"(?:how do (?:i|you) say|how to say|what is the word for|comment dit-on)\s+['\"]?(.+?)['\"]?\??$", lower)
        if translation_match:
            target = translation_match.group(1).strip()
            return "TRANSLATION", target

        # 2. Explicit hint / hesitation keywords
        hint_triggers = ["hint", "i'm stuck", "i don't know", "help me", "aide-moi", "give me a clue", "je ne sais pas"]
        if any(t in lower for t in hint_triggers):
            return "HINT", None

        # 3. Trailing or hesitant input
        if lower in ["euh", "je...", "uh", "um", "i want", "je veux..."] or len(lower.split()) <= 1:
            return "HESITATION", None

        return "NONE", None

    def generate_hint_prompt(self, request_type, target, conversation_history):
        """Builds system instructions to enforce tiered assistance."""
        if request_type == "TRANSLATION":
            return f"""L'utilisateur demande comment dire '{target}' en français.
RÈGLES:
1. Donne la traduction française de '{target}'.
2. Donne une amorce de phrase courte.
3. Demande à l'utilisateur de compléter sa phrase.

FORMAT DE SORTIE REQUIS:
FR: <Ta réponse et amorce en français>
EN: <The exact English translation of your French response>"""

        elif request_type in ["HINT", "HESITATION"]:
            return """L'utilisateur hésite ou demande un indice.
RÈGLES:
1. Donne un indice encourageant ou une amorce de phrase en français (ex: 'Essaie de commencer par : Je voudrais...').
2. Ne donne pas toute la phrase immédiatement.
3. Demande-lui gentiment de compléter son idée.

FORMAT DE SORTIE REQUIS:
FR: <Ton indice et amorce en français>
EN: <The exact English translation of your French response>"""

        return None

# --- SESSION MEMORY ENGINE ---
class SessionMemoryEngine:
    """Handles local storage of student weak points, hobbies, user name, and session history in session_review.json."""
    
    def __init__(self, filepath="session_review.json"):
        self.filepath = Path(filepath)
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
        """Loads previous session data if it exists."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def detect_and_save_hobby(self, user_text):
        """Silently listens for hobbies/passions and saves them to the ledger."""
        pattern = r"(?:j'aime|j'adore|je joue|ma passion est|mon passe-temps est)\s+(?:à|de|de la|du|le|la|les|un|une)?\s*([a-zA-ZÀ-ÿ\s]+)"
        match = re.search(pattern, user_text, re.IGNORECASE)
        
        if match:
            hobby = match.group(1).strip().lower()
            if hobby and len(hobby) > 2 and hobby not in self.hobbies:
                self.hobbies.append(hobby)
                self.past_data["hobbies"] = self.hobbies
                self.save_raw()
                print(f"[Memory Engine] New hobby logged: {hobby}")
                return True
        return False

    def get_warmup_prompt(self):
        """Injects past weak points and student persona into BMO's startup prompt."""
        hobby_str = f" The student's hobbies and interests include: {', '.join(self.hobbies)}." if self.hobbies else ""
        
        if not self.past_data or "last_session" not in self.past_data:
            return f"Welcome the student warmly in French.{hobby_str} Ask how their day is going."
        
        last_corrections = self.past_data["last_session"].get("corrections_made", [])
        if last_corrections:
            topics = ", ".join(last_corrections[:2])
            return f"Welcome the student warmly in French.{hobby_str} In the last session, they struggled with: {topics}. Casually ask a simple question in French to test one of these concepts."
        
        return f"Welcome the student warmly in French.{hobby_str} Ask what they would like to practice today."

    def log_turn(self, user_text, bmo_text):
        self.current_session["total_turns"] += 1
        self.detect_and_save_hobby(user_text)
        self.log_correction(bmo_text)

    def log_roleplay(self, scenario):
        if scenario not in self.current_session["roleplays_completed"]:
            self.current_session["roleplays_completed"].append(scenario)

    def log_correction(self, bmo_response):
        """Simple heuristic to log grammar topics if BMO makes corrections."""
        lower_resp = bmo_response.lower()
        topics = {
            "passé composé": ["passé composé", "past tense", "conjugaison"],
            "imparfait": ["imparfait", "habitual past"],
            "gender agreement": ["masculine", "feminine", "genre", "accord"],
            "future tense": ["future tense", "futur"]
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
        """Triggers an LLM call to extract vocab on exit and saves the JSON file."""
        print("[Memory Engine] Saving session data...")
        
        # 1. Extract vocabulary using the LLM if available
        if llm_instance and len(conversation_history) > 2:
            extraction_prompt = "Review this transcript and list 3 useful French vocabulary words or phrases the student learned or should remember. Output ONLY a comma-separated list."
            msgs = [{"role": "system", "content": extraction_prompt}] + conversation_history[-6:]
            try:
                vocab_res = llm_instance.create_chat_completion(messages=msgs, max_tokens=30, temperature=0.1)
                vocab_list = vocab_res["choices"][0]["message"]["content"].split(",")
                self.current_session["new_vocabulary"] = [v.strip() for v in vocab_list if v.strip()]
            except Exception as e:
                print(f"Vocab extraction failed: {e}")

        # Fallback if no LLM vocab extraction
        if not self.current_session["new_vocabulary"] and conversation_history:
            for turn in reversed(conversation_history):
                if turn.get("role") == "assistant":
                    words = [w.strip() for w in re.findall(r"\b[a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]{4,}\b", turn.get("content", ""))]
                    self.current_session["new_vocabulary"] = list(set(words[:3]))
                    break

        # 2. Update and save local JSON ledger
        self.past_data["hobbies"] = self.hobbies
        self.past_data["last_session"] = self.current_session
        if "all_time_weak_points" not in self.past_data:
            self.past_data["all_time_weak_points"] = {}
            
        for topic in self.current_session["corrections_made"]:
            self.past_data["all_time_weak_points"][topic] = self.past_data["all_time_weak_points"].get(topic, 0) + 1

        self.save_raw()
        print("[Memory Engine] Session saved successfully to session_review.json.")

# --- HARDWARE AUDIO RECORDER ---
AUDIO_BUFFER = []
IS_RECORDING = False

def hardware_audio_callback(indata, frames, time_info, status):
    if IS_RECORDING:
        AUDIO_BUFFER.append(indata.copy())

try:
    sd_stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=hardware_audio_callback)
    sd_stream.start()
    print("[OK] Native Hardware Microphone Stream Active!")
except Exception as e:
    print(f"[CRITICAL] Failed to access microphone: {e}")

# --- PYTHON API EXPOSED TO FRONTEND ---
class BmoBridge:
    def __init__(self):
        self.history = []
        self.roleplay = RoleplayEngine()
        self.scaffolding = ScaffoldingEngine()
        self.memory = SessionMemoryEngine()
        
        # Inject warm-up instruction based on past session history
def clean_hallucinated_text(text: str) -> str:
    """Detects and removes Whisper repetition hallucination loops on trailing audio."""
    if not text:
        return ""
    # Strip repeating sentence/phrase loops
    sentences = [s.strip() for s in text.replace(".", ".|").replace("!", "!|").replace("?", "?|").split("|") if s.strip()]
    if len(sentences) > 2:
        seen = []
        for s in sentences:
            if s in seen:
                break
            seen.append(s)
        text = " ".join(seen)
    
    # Strip repeating 2-4 word phrases
    words = text.split()
    if len(words) > 6:
        cleaned_words = []
        for i, w in enumerate(words):
            if i >= 4 and words[i-4:i] == words[i-8:i-4]:
                break
            cleaned_words.append(w)
        text = " ".join(cleaned_words)
    return text.strip()

class BmoBridge:
    def __init__(self):
        self.history = []
        self.memory = SessionMemoryEngine()
        self.roleplay = RoleplayEngine()
        self.scaffolding = ScaffoldingEngine()
        
        warmup_instruction = self.memory.get_warmup_prompt()
        self.warmup_prompt = f"{BMO_SYSTEM_PROMPT}\n\nCONTEXTE D'ACCUEIL :\n{warmup_instruction}"

    def toggle_record(self):
        global IS_RECORDING, AUDIO_BUFFER
        if not IS_RECORDING:
            AUDIO_BUFFER.clear()
            IS_RECORDING = True
            return {"status": "listening"}
        else:
            IS_RECORDING = False
            # Run processing in a background thread so UI doesn't freeze
            threading.Thread(target=self.process_pipeline).start()
            return {"status": "thinking"}

    def process_pipeline(self):
        try:
            if not AUDIO_BUFFER:
                window.evaluate_js("window.setBMOState('idle');")
                return

            audio_data = np.concatenate(AUDIO_BUFFER, axis=0).flatten()
            AUDIO_BUFFER.clear()

            # 1. Fast French-Only Whisper ASR (Optimized beam_size=1 for CPU)
            result = whisper_model.transcribe(
                audio_data, 
                fp16=False, 
                language="fr",
                initial_prompt="Le nom du robot est BMO (prononcé Beemo). Transcription exacte du français parlé.",
                beam_size=1,          # Fast CPU decoding (1x beam)
                best_of=1,            # 5x speedup on CPU
                temperature=0.0,      # Deterministic decoding
                condition_on_previous_text=False
            )
            raw_text = result.get("text", "").strip() or "Bonjour BMO!"
            raw_text = clean_hallucinated_text(raw_text)
            user_text = normalize_bmo_name(raw_text)

            # Detect exit command to save session
            exit_triggers = ["goodbye", "au revoir", "see you later", "stop session", "à bientôt", "a bientot"]
            if any(t in user_text.lower() for t in exit_triggers):
                bmo_fr = "Au revoir ! J'ai enregistré tous nos progrès dans ton journal. À la prochaine !"
                bmo_en = "Goodbye! I saved all our progress in your journal. See you next time!"
                self.memory.save_session(self.history, llm if has_llm else None)
                
                bmo_fr = bmo_fr.replace("BMO", "Beemo")
                bmo_en = bmo_en.replace("BMO", "Beemo")
                
                self.history.append({"role": "user", "content": user_text})
                window.evaluate_js(f"window.appendChatMessage('user', {repr(user_text)});")
                self.history.append({"role": "assistant", "content": bmo_fr})
                window.evaluate_js(f"window.appendChatMessage('bot', {repr(bmo_fr)}, {repr(bmo_en)});")
                
                window.evaluate_js("window.setBMOState('speaking');")
                if kokoro:
                    try:
                        audio, sr_out = kokoro.create(bmo_fr, voice="ff_siwis", speed=0.85, lang="fr-fr")
                    except Exception:
                        audio, sr_out = kokoro.create(bmo_fr, voice="af_bella", speed=0.85)
                    audio_flat = audio.squeeze()
                    pitch_factor = 1.2599
                    new_len = int(len(audio_flat) / pitch_factor)
                    samples_shifted = scipy.signal.resample(audio_flat, new_len).astype(np.float32)
                    sd.play(samples_shifted, sr_out)
                    sd.wait()
                window.evaluate_js("window.setBMOState('idle');")
                return

            # 1. Listen for new hobbies passively & build Persona Context
            self.memory.detect_and_save_hobby(user_text)
            current_name = self.memory.user_name if self.memory.user_name else "a student"
            hobby_context = ""
            if self.memory.hobbies:
                hobby_str = ", ".join(self.memory.hobbies)
                hobby_context = f"\nThe student's hobbies and interests include: {hobby_str}. Naturally weave these topics into examples or conversation when relevant."

            persona_prefix = f"The user's name is {current_name}.{hobby_context}\n\n"

            # 2. Analyze Scaffolding & Roleplay Triggers
            scaffold_type, target = self.scaffolding.detect_scaffolding_request(user_text)
            current_tts_speed = 0.85

            if scaffold_type != "NONE":
                # Hint Mode Triggered -> Slower, clear articulation (0.70x)
                current_tts_speed = 0.70
                scaffold_prompt = persona_prefix + self.scaffolding.generate_hint_prompt(scaffold_type, target, self.history)
                llm_msgs = [{"role": "system", "content": scaffold_prompt}] + self.history[-3:] + [{"role": "user", "content": user_text}]
            elif self.roleplay.mode == "TUTOR":
                scenario, role = self.roleplay.detect_roleplay_intent(user_text)
                if scenario:
                    self.roleplay.mode = "ROLEPLAY"
                    self.roleplay.scenario = scenario
                    self.roleplay.character_role = role
                    self.roleplay.turn_count = 0
                    self.roleplay.roleplay_history = []
                    self.memory.log_roleplay(scenario)
                
                system_instruction = persona_prefix + self.roleplay.get_system_prompt()
                llm_msgs = [{"role": "system", "content": system_instruction}] + self.history[-4:] + [{"role": "user", "content": user_text}]
            else:
                # Active Roleplay
                self.roleplay.turn_count += 1
                self.roleplay.roleplay_history.append({"role": "Student", "content": user_text})
                
                if self.roleplay.turn_count > self.roleplay.max_turns or any(w in user_text.lower() for w in ["stop", "finish", "done", "quitter", "terminer"]):
                    current_tts_speed = 0.75
                    debrief_query = self.roleplay.build_debrief_prompt()
                    llm_msgs = [{"role": "system", "content": persona_prefix + "Tu es BMO le tuteur de français."}, {"role": "user", "content": debrief_query}]
                    self.roleplay.mode = "TUTOR"
                else:
                    system_instruction = persona_prefix + self.roleplay.get_system_prompt()
                    llm_msgs = [{"role": "system", "content": system_instruction}] + self.history[-4:] + [{"role": "user", "content": user_text}]

            # 3. LLM Generation (French Response + English Translation)
            if has_llm:
                full_resp = llm.create_chat_completion(
                    messages=llm_msgs, 
                    max_tokens=220,
                    temperature=0.3,
                    top_p=0.9
                )["choices"][0]["message"]["content"]
            else:
                full_resp = f"FR: J'ai entendu : '{user_text}'. Comment ça va ?\nEN: I heard: '{user_text}'. How are you?"

            if "EN:" in full_resp:
                parts = full_resp.split("EN:")
                bmo_fr = parts[0].replace("FR:", "").strip()
                bmo_en = parts[1].strip()
            else:
                bmo_fr = full_resp.replace("FR:", "").strip()
                bmo_en = ""

            # Secondary translation fallback if LLM omitted EN tag
            if not bmo_en and has_llm and bmo_fr:
                try:
                    trans_resp = llm.create_chat_completion(
                        messages=[{"role": "user", "content": f"Translate this French sentence into English: {bmo_fr}"}],
                        max_tokens=60,
                        temperature=0.0
                    )
                    bmo_en = trans_resp["choices"][0]["message"]["content"].strip()
                except Exception:
                    bmo_en = bmo_fr

            if self.roleplay.mode == "ROLEPLAY":
                self.roleplay.roleplay_history.append({"role": "BMO", "content": bmo_fr})

            self.memory.log_turn(user_text, bmo_fr)

            bmo_fr = bmo_fr.replace("BMO", "Beemo")
            bmo_en = bmo_en.replace("BMO", "Beemo")

            self.history.append({"role": "user", "content": user_text})
            window.evaluate_js(f"window.appendChatMessage('user', {repr(user_text)});")
            
            self.history.append({"role": "assistant", "content": bmo_fr})
            window.evaluate_js(f"window.appendChatMessage('bot', {repr(bmo_fr)}, {repr(bmo_en)});")

            # 4. Kokoro / gTTS Adaptive Audio Synthesis
            window.evaluate_js("window.setBMOState('speaking');")
            if kokoro:
                try:
                    audio, sr_out = kokoro.create(bmo_fr, voice="ff_siwis", speed=current_tts_speed, lang="fr-fr")
                except Exception:
                    audio, sr_out = kokoro.create(bmo_fr, voice="af_bella", speed=current_tts_speed)
                
                audio_flat = audio.squeeze()
                pitch_factor = 1.2599
                new_len = int(len(audio_flat) / pitch_factor)
                samples_shifted = scipy.signal.resample(audio_flat, new_len).astype(np.float32)
                
                sd.play(samples_shifted, sr_out)
                sd.wait()
            else:
                from gtts import gTTS
                tts = gTTS(text=bmo_fr, lang='fr', slow=True)
                tts.save("temp_bmo_gtts.mp3")
                import soundfile as sf
                samples_raw, sr_out = sf.read("temp_bmo_gtts.mp3")
                if samples_raw.ndim > 1:
                    samples_raw = samples_raw.mean(axis=1)
                pitch_factor = 1.2599
                new_len = int(len(samples_raw) / pitch_factor)
                samples_shifted = scipy.signal.resample(samples_raw, new_len).astype(np.float32)
                sd.play(samples_shifted, sr_out)
                sd.wait()

            window.evaluate_js("window.setBMOState('idle');")

        except Exception as e:
            print(f"[Pipeline Error]: {e}")
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
        btn.innerText = '🙈 Hide';
    } else {
        box.style.display = 'none';
        btn.innerText = '🌐 Translate';
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

        if (!translation) translation = text;
        
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

if __name__ == "__main__":
    window = webview.create_window("BMO Live Edge Tutor", html=html_content, js_api=bridge, width=860, height=600, background_color='#122821')
    webview.start()
