import sys
import os
import time
import re
import json
import random
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

    def get_system_prompt(self):
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

    def generate_hint_prompt(self, request_type, target, conversation_history):
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

    def get_warmup_prompt(self):
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
        if llm_instance and len(conversation_history) > 2:
            extraction_prompt = "Review this transcript and list 3 useful French vocabulary words or phrases the student learned or should remember. Output ONLY a comma-separated list."
            msgs = [{"role": "system", "content": extraction_prompt}] + conversation_history[-6:]
            try:
                vocab_res = llm_instance.create_chat_completion(messages=msgs, max_tokens=30, temperature=0.1)
                vocab_list = vocab_res["choices"][0]["message"]["content"].split(",")
                self.current_session["new_vocabulary"] = [v.strip() for v in vocab_list if v.strip()]
            except Exception as e:
                print(f"Vocab extraction failed: {e}")

        if not self.current_session["new_vocabulary"] and conversation_history:
            for turn in reversed(conversation_history):
                if turn.get("role") == "assistant":
                    words = [w.strip() for w in re.findall(r"\b[a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]{4,}\b", turn.get("content", ""))]
                    self.current_session["new_vocabulary"] = list(set(words[:3]))
                    break

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
    audio_stream = sd.InputStream(
        samplerate=16000, 
        channels=1, 
        dtype='float32', 
        callback=audio_callback
    )
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

# --- PYTHON-SIDE INTENT CLASSIFIER (PRE-LLM) ---
def classify_user_intent(user_text):
    """Classifies user intent via keyword matching. Returns (intent_label, instruction_for_llm)."""
    lower = user_text.lower()
    intent_map = [
        ("WORK", ["travail", "travaille", "bureau", "collègue", "collegue", "métier", "metier", "boulot", "entreprise", "patron"],
         "L'étudiant parle de son travail. Réagis à ce sujet et pose une question sur son métier ou sa journée de travail."),
        ("HOBBY", ["jouer", "musique", "sport", "lire", "film", "série", "jeu", "dessiner", "peindre", "danse", "guitare", "piano", "foot", "basket"],
         "L'étudiant parle de ses loisirs ou passe-temps. Explore ce sujet avec enthousiasme."),
        ("FOOD", ["manger", "mange", "mangé", "cuisine", "restaurant", "faim", "dîner", "déjeuner", "petit-déjeuner", "gâteau", "pizza", "poulet", "salade"],
         "L'étudiant parle de nourriture ou de repas. Discute de ce thème culinaire."),
        ("FAMILY", ["famille", "frère", "frere", "sœur", "soeur", "maman", "papa", "parents", "enfants", "fils", "fille", "mari", "femme", "cousin"],
         "L'étudiant parle de sa famille. Pose une question chaleureuse à ce sujet."),
        ("SCHOOL", ["école", "ecole", "étudier", "etudier", "cours", "professeur", "examen", "devoir", "classe", "université", "apprendre"],
         "L'étudiant parle de ses études ou de l'école. Pose une question sur ce sujet."),
        ("TRAVEL", ["voyage", "voyager", "vacances", "pays", "ville", "avion", "train", "hôtel", "plage", "visiter"],
         "L'étudiant parle de voyage. Explore cette destination ou cette expérience."),
        ("FEELING", ["bien", "mal", "fatigué", "fatigue", "content", "contente", "triste", "heureux", "heureuse", "ennuyé", "stressé", "malade"],
         "L'étudiant parle de son état émotionnel ou physique. Réagis avec empathie et fais avancer la conversation."),
        ("GREETING", ["bonjour", "salut", "coucou", "hey", "bonsoir", "hello"],
         "L'étudiant te salue. Réponds brièvement et lance un sujet de conversation intéressant."),
        ("QUESTION", ["qu'est-ce", "comment", "pourquoi", "est-ce que", "quand", "où", "combien", "quel", "quelle"],
         "L'étudiant pose une question. Réponds-y directement en français simple."),
    ]
    for label, keywords, instruction in intent_map:
        if any(kw in lower for kw in keywords):
            return label, instruction
    return "GENERAL", "Réagis au contenu de ce que dit l'étudiant et pose une nouvelle question pour faire avancer la conversation."

# --- PYTHON-SIDE FRENCH GRAMMAR CORRECTION ENGINE (PRE-LLM) ---
def detect_french_grammar_errors(user_text):
    """Detects common A2/B1 French grammar errors via regex. Returns correction instruction or None."""
    lower = user_text.lower().strip()
    grammar_rules = [
        (r"\bje\s+va\b", "je vais", "conjugaison du verbe 'aller' au présent : je vais, tu vas, il va"),
        (r"\bj'avais\s+bien\b", "je vais bien", "confusion de temps : 'j'avais' est l'imparfait, il faut utiliser le présent 'je vais bien'"),
        (r"\bje\s+suis\s+allé\s+à\s+le\b", "je suis allé au", "contraction obligatoire : à + le = au"),
        (r"\bà\s+le\s", "au ", "contraction obligatoire : à + le = au"),
        (r"\bde\s+le\s", "du ", "contraction obligatoire : de + le = du"),
        (r"\bje\s+(?:mange|parle|travaille|joue|aime|regarde|écoute|habite|danse|chante)\s+pas\b", None, "négation incomplète : il manque 'ne' → 'je ne ... pas'"),
        (r"\bil\s+a\s+manger\b", "il a mangé", "confusion infinitif/participe passé : après 'avoir', on utilise le participe passé 'mangé'"),
        (r"\bil\s+a\s+parler\b", "il a parlé", "confusion infinitif/participe passé : après 'avoir', on utilise le participe passé 'parlé'"),
        (r"\bil\s+a\s+travailler\b", "il a travaillé", "confusion infinitif/participe passé : après 'avoir', on utilise le participe passé 'travaillé'"),
        (r"\bje\s+est\b", "je suis", "conjugaison du verbe 'être' : je suis (pas 'je est')"),
        (r"\btu\s+est\b", "tu es", "conjugaison du verbe 'être' : tu es (pas 'tu est')"),
        (r"\bje\s+a\b", "j'ai", "conjugaison du verbe 'avoir' : j'ai (pas 'je a')"),
        (r"\btu\s+a\b", "tu as", "conjugaison du verbe 'avoir' : tu as (pas 'tu a')"),
    ]
    for pattern, correction, explanation in grammar_rules:
        match = re.search(pattern, lower)
        if match:
            original = match.group(0)
            if correction:
                return f"CORRECTION REQUISE: L'étudiant a dit \"{original}\". La forme correcte est \"{correction}\" ({explanation}). Tu DOIS corriger cette erreur avec la Méthode du Sandwich : 1) félicite l'effort, 2) explique la correction, 3) dis 'Répète après moi : <phrase corrigée>'."
            else:
                return f"CORRECTION REQUISE: L'étudiant a fait une erreur — {explanation}. Tu DOIS corriger cette erreur avec la Méthode du Sandwich."
    return None

# --- RESPONSE DIVERSITY POOL (POST-LLM FALLBACK) ---
PIVOT_RESPONSES = {
    "WORK": [
        "Ah, tu travailles ! Quel genre de travail fais-tu, {name} ?",
        "C'est bien de travailler ! Tu aimes ton travail ?",
        "Tu travailles beaucoup ? Raconte-moi, {name} !",
        "Intéressant ! Tu travailles depuis longtemps, {name} ?",
    ],
    "HOBBY": [
        "Super, {name} ! C'est quoi ton passe-temps préféré ?",
        "Ah, c'est amusant ! Tu fais ça souvent, {name} ?",
        "J'adore ça ! Depuis quand tu fais cette activité ?",
    ],
    "FOOD": [
        "Miam ! Quel est ton plat préféré, {name} ?",
        "Tu cuisines souvent, {name} ?",
        "C'est délicieux ! Tu manges ça souvent ?",
    ],
    "FAMILY": [
        "C'est chouette ! Tu as une grande famille, {name} ?",
        "Ah, la famille ! Vous êtes proches ?",
        "Raconte-moi plus sur ta famille, {name} !",
    ],
    "SCHOOL": [
        "L'école, c'est important ! Tu aimes tes cours, {name} ?",
        "Qu'est-ce que tu étudies en ce moment ?",
        "Tu as un cours préféré, {name} ?",
    ],
    "TRAVEL": [
        "Le voyage, c'est super ! Tu es déjà allé où, {name} ?",
        "Tu aimes voyager ? Quel pays veux-tu visiter ?",
        "Raconte-moi ton meilleur souvenir de voyage !",
    ],
    "FEELING": [
        "Je comprends, {name}. Qu'est-ce que tu fais aujourd'hui ?",
        "D'accord ! Qu'as-tu prévu pour la journée, {name} ?",
        "Je vois, {name}. Parlons de quelque chose d'amusant !",
    ],
    "GREETING": [
        "Qu'est-ce que tu veux pratiquer aujourd'hui, {name} ?",
        "Content de te voir, {name} ! On parle de quoi ?",
        "Qu'as-tu fait de beau récemment, {name} ?",
    ],
    "QUESTION": [
        "Bonne question, {name} ! Laisse-moi t'expliquer.",
        "Hmm, c'est intéressant comme question !",
    ],
    "GENERAL": [
        "Intéressant, {name} ! Dis-moi en plus !",
        "Je vois ! Qu'est-ce que tu voudrais pratiquer, {name} ?",
        "D'accord, {name}. Parlons d'un nouveau sujet !",
        "C'est bien, {name} ! Continue, je t'écoute.",
    ],
}

def sanitize_history(history):
    """Removes duplicate/looping assistant messages from history."""
    loop_phrases = ["je vais bien, merci", "comment ça va", "comment allez-vous", "comment vas-tu"]
    sanitized = []
    seen_assistant_contents = set()
    for msg in history:
        if msg["role"] == "assistant":
            content_lower = msg["content"].lower().strip()
            if content_lower in seen_assistant_contents:
                continue
            if any(phrase in content_lower for phrase in loop_phrases) and len(seen_assistant_contents) > 0:
                continue
            seen_assistant_contents.add(content_lower)
        sanitized.append(msg)
    return sanitized

def count_tokens(text):
    """Count tokens using the loaded LLM tokenizer. Falls back to char estimate if LLM unavailable."""
    if has_llm and llm:
        try:
            return len(llm.tokenize(text.encode('utf-8', errors='replace')))
        except Exception:
            pass
    return len(text) // 3  # Conservative fallback: ~3 chars per token for French

MAX_OUTPUT_TOKENS = 150
SAFETY_MARGIN = 20
N_CTX = 512
INPUT_TOKEN_BUDGET = N_CTX - MAX_OUTPUT_TOKENS - SAFETY_MARGIN  # = 342 tokens for input

def build_token_safe_messages(system_instruction, user_text, history):
    """Builds a message list guaranteed to fit within the LLM's context window.
    
    Token budget allocation:
    1. System instruction (fixed, always included)
    2. Current user message (fixed, always included)
    3. History (variable, fills remaining budget most-recent-first)
    """
    # Measure fixed costs
    system_tokens = count_tokens(system_instruction)
    user_tokens = count_tokens(user_text)
    fixed_cost = system_tokens + user_tokens + 10  # +10 for chat template overhead
    
    # Calculate remaining budget for history
    history_budget = max(0, INPUT_TOKEN_BUDGET - fixed_cost)
    
    # Sanitize history first (remove loops/duplicates)
    clean_history = sanitize_history(history)
    
    # Fill history from most recent, respecting token budget
    selected_history = []
    used_tokens = 0
    for msg in reversed(clean_history):
        msg_tokens = count_tokens(msg["content"]) + 4  # +4 for role/template overhead
        if used_tokens + msg_tokens > history_budget:
            break
        selected_history.insert(0, msg)
        used_tokens += msg_tokens
    
    total_input = fixed_cost + used_tokens
    print(f"[Token Budget] system={system_tokens} user={user_tokens} history={used_tokens} total_input={total_input}/{INPUT_TOKEN_BUDGET}")
    
    # Assemble final message list
    return selected_history + [{"role": "user", "content": user_text}, {"role": "system", "content": system_instruction}]

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

            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 0.005:
                print(f"[Audio VAD Guard] Ignored silent/noise trigger (RMS={rms:.5f} < 0.005).")
                window.evaluate_js("window.setBMOState('idle');")
                return

            if len(audio_data) < 16000 * 0.3:
                print("[WARN] Audio recording too short (< 0.3s). Ignoring.")
                window.evaluate_js("window.setBMOState('idle');")
                return

            result = whisper_model.transcribe(
                audio_data, 
                fp16=False, 
                language="fr",
                initial_prompt="Le nom du robot est BMO (prononcé Beemo). Transcription exacte du français parlé.",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False
            )
            raw_text = result.get("text", "").strip() or "Bonjour BMO!"
            raw_text = clean_hallucinated_text(raw_text)
            user_text = normalize_bmo_name(raw_text)

            exit_triggers = ["goodbye", "au revoir", "see you later", "stop session", "à bientôt", "a bientot"]
            if any(t in user_text.lower() for t in exit_triggers):
                bmo_fr = "Au revoir ! J'ai enregistré tous nos progrès dans ton journal. À la prochaine !"
                bmo_en = "Goodbye! I saved all our progress in your journal. See you next time!"
                self.memory.save_session(self.history, llm if has_llm else None)
                
                bmo_fr_ui = bmo_fr.replace("BMO", "Beemo")
                bmo_en_ui = bmo_en.replace("BMO", "Beemo")
                
                self.history.append({"role": "user", "content": user_text})
                window.evaluate_js(f"window.appendChatMessage('user', {repr(user_text)});")
                self.history.append({"role": "assistant", "content": bmo_fr_ui})
                window.evaluate_js(f"window.appendChatMessage('bot', {repr(bmo_fr_ui)}, {repr(bmo_en_ui)});")
                
                window.evaluate_js("window.setBMOState('speaking');")
                if kokoro:
                    try:
                        audio, sr_out = kokoro.create(sanitize_for_tts(bmo_fr), voice="ff_siwis", speed=0.85, lang="fr-fr")
                    except Exception:
                        audio, sr_out = kokoro.create(sanitize_for_tts(bmo_fr), voice="af_bella", speed=0.85)
                    audio_flat = audio.squeeze()
                    samples_shifted = scipy.signal.resample(audio_flat, int(len(audio_flat) / 1.2599)).astype(np.float32)
                    sd.play(samples_shifted, sr_out)
                    sd.wait()
                window.evaluate_js("window.setBMOState('idle');")
                return

            self.memory.detect_and_save_hobby(user_text)
            
            intercepted_bmo_text = None
            lower_user_text = user_text.lower().replace("?", "").replace("!", "").replace(".", "").strip()

            # Check if user is asking for BMO's name or identity
            bmo_name_triggers = ["comment tu t'appelles", "comment vous appelez vous", "comment vous vous appelez", "quel est ton nom", "tu t'appelles comment", "what is your name", "who are you", "qui es-tu", "qui es tu"]
            if any(trigger in lower_user_text for trigger in bmo_name_triggers):
                intercepted_bmo_text = "FR: Je m'appelle BMO ! Je suis ton tuteur de français. Qu'aimerais-tu pratiquer aujourd'hui ?\nEN: My name is BMO! I am your French tutor. What would you like to practice today?"
            elif hasattr(self.memory, 'pending_name_change') and self.memory.pending_name_change:
                lower_ans = user_text.lower()
                if "yes" in lower_ans or "oui" in lower_ans:
                    self.memory.save_name(self.memory.pending_name_change)
                    intercepted_bmo_text = f"FR: D'accord, {self.memory.user_name} ! C'est noté. Que veux-tu faire aujourd'hui ?\nEN: Okay, {self.memory.user_name}! Duly noted. What would you like to do today?"
                else:
                    intercepted_bmo_text = f"FR: D'accord, pas de souci. Je continuerai à t'appeler {self.memory.user_name}.\nEN: Okay, no problem. I'll continue calling you {self.memory.user_name}."
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
                        intercepted_bmo_text = f"FR: Bonjour {new_name} ! C'est noté.\nEN: Hello {new_name}! Noted."

            current_name = self.memory.user_name if self.memory.user_name else "mon ami"
            hobby_context = ""
            if self.memory.hobbies:
                hobby_str = ", ".join(self.memory.hobbies)
                hobby_context = f"\nThe student's hobbies and interests include: {hobby_str}. Naturally weave these topics into examples or conversation when relevant."

            identity_anchor = f"Tu t'appelles BMO (prononcé Beemo). Tu es un tuteur de français amical et encourageant qui discute avec {current_name}."
            behavior_anchor = (
                "RÈGLES STRICTES DE CONVERSATION :\n"
                "1. INTERDICTION DES SALUTATIONS RÉPÉTÉES : N'utilise JAMAIS les mots 'Bonjour' ou 'Salut' après le tout premier message de la conversation. Commence directement ta réponse.\n"
                "2. INTERDICTION DU PERROQUET : Ne répète jamais la phrase de l'étudiant mot pour mot dans ta réponse.\n"
                "3. CORRECTION ACTIVE (MÉTHODE DU SANDWICH) : Si l'étudiant fait une faute (par exemple 'Je va bien' au lieu de 'Je vais bien'), félicite l'effort et corrige-la poliment en expliquant brièvement la conjugaison.\n"
                "4. FLUX NATUREL & PROGRESSION : Réagis au sens exact de ce que dit l'étudiant et fais avancer la discussion. Pose une NOUVELLE question différente à chaque tour (ne redemande jamais 'Comment allez-vous ?').\n"
                "5. FORMAT & LANGUE : Parle EXCLUSIVEMENT en français. 1 ou 2 phrases maximum en français simple (présent, passé composé, imparfait, ou futur)."
            )
            
            if self.roleplay.mode == "ROLEPLAY":
                system_instruction = identity_anchor + "\n" + self.roleplay.get_system_prompt() + hobby_context
            else:
                # --- INJECT INTENT + GRAMMAR CONTEXT INTO SYSTEM PROMPT ---
                user_intent, intent_instruction = classify_user_intent(user_text)
                grammar_correction = detect_french_grammar_errors(user_text)
                
                intent_block = f" INTENT: {intent_instruction}"
                grammar_block = f" {grammar_correction}" if grammar_correction else ""
                
                system_instruction = identity_anchor + "\n" + behavior_anchor + intent_block + grammar_block

            current_tts_speed = 0.85

            if intercepted_bmo_text:
                full_resp = intercepted_bmo_text
            else:
                scaffold_type, target = self.scaffolding.detect_scaffolding_request(user_text)

                if scaffold_type != "NONE":
                    current_tts_speed = 0.70
                    scaffold_prompt = identity_anchor + " " + self.scaffolding.generate_hint_prompt(scaffold_type, target, self.history)
                    llm_msgs = build_token_safe_messages(scaffold_prompt, user_text, self.history)
                elif self.roleplay.mode == "TUTOR":
                    scenario, role = self.roleplay.detect_roleplay_intent(user_text)
                    if scenario:
                        self.roleplay.mode = "ROLEPLAY"
                        self.roleplay.scenario = scenario
                        self.roleplay.character_role = role
                        self.roleplay.turn_count = 0
                        self.roleplay.roleplay_history = []
                        self.memory.log_roleplay(scenario)
                    
                    llm_msgs = build_token_safe_messages(system_instruction, user_text, self.history)
                else:
                    self.roleplay.turn_count += 1
                    self.roleplay.roleplay_history.append({"role": "Student", "content": user_text})
                    
                    if self.roleplay.turn_count > self.roleplay.max_turns or any(w in user_text.lower() for w in ["stop", "finish", "done", "quitter", "terminer"]):
                        current_tts_speed = 0.75
                        debrief_query = self.roleplay.build_debrief_prompt()
                        llm_msgs = [{"role": "system", "content": identity_anchor + " Tu es BMO le tuteur de français."}, {"role": "user", "content": debrief_query}]
                        self.roleplay.mode = "TUTOR"
                    else:
                        llm_msgs = build_token_safe_messages(system_instruction, user_text, self.history)

                if has_llm:
                    try:
                        full_resp = llm.create_chat_completion(
                            messages=llm_msgs, 
                            max_tokens=MAX_OUTPUT_TOKENS,
                            temperature=0.45,
                            repeat_penalty=1.2,
                            top_p=0.85
                        )["choices"][0]["message"]["content"]
                    except Exception as e:
                        print(f"[LLM Error Safe Fallback]: {e}")
                        full_resp = f"FR: Oups, {current_name} ! Peux-tu répéter s'il te plaît ?\nEN: Oops, {current_name}! Can you repeat that please?"
                else:
                    full_resp = f"FR: Bonjour {current_name} ! Que veux-tu faire aujourd'hui ?\nEN: Hello {current_name}! What would you like to do today?"

            if "EN:" in full_resp:
                parts = full_resp.split("EN:")
                bmo_fr = parts[0].replace("FR:", "").strip()
                bmo_en = parts[1].strip()
            else:
                bmo_fr = full_resp.replace("FR:", "").strip()
                bmo_en = ""

            # --- PYTHON ANTI-LOOP GUARDRAIL (CIRCUIT BREAKER) WITH DIVERSITY POOL ---
            last_bot_msg = next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), "")
            loop_triggers = ["je vais bien, merci", "comment ça va pour toi", "comment allez-vous", "comment vas-tu", "comment ça va"]
            
            is_looping = any(trigger in bmo_fr.lower() for trigger in loop_triggers) and len(self.history) >= 2
            is_duplicate = bmo_fr.strip().lower() == last_bot_msg.strip().lower()

            if is_looping or is_duplicate:
                # Use intent-aware diversity pool instead of hardcoded fallback
                detected_intent = classify_user_intent(user_text)[0] if not intercepted_bmo_text else "GENERAL"
                pool = PIVOT_RESPONSES.get(detected_intent, PIVOT_RESPONSES["GENERAL"])
                bmo_fr = random.choice(pool).format(name=current_name)
                print(f"[Loop Interceptor] Caught loop → Pivoted with intent={detected_intent}: '{bmo_fr}')")
            # ----------------------------------------------------

            if not bmo_en and has_llm and bmo_fr and not intercepted_bmo_text:
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

            bmo_fr_ui = bmo_fr.replace("Beemo", "BMO")
            bmo_en_ui = bmo_en.replace("Beemo", "BMO")

            self.history.append({"role": "user", "content": user_text})
            window.evaluate_js(f"window.appendChatMessage('user', {repr(user_text)});")
            
            self.history.append({"role": "assistant", "content": bmo_fr_ui})
            window.evaluate_js(f"window.appendChatMessage('bot', {repr(bmo_fr_ui)}, {repr(bmo_en_ui)});")

            bmo_fr_spoken = sanitize_for_tts(bmo_fr_ui.replace("BMO", "Beemo"))

            window.evaluate_js("window.setBMOState('speaking');")
            if kokoro:
                try:
                    audio, sr_out = kokoro.create(bmo_fr_spoken, voice="ff_siwis", speed=current_tts_speed, lang="fr-fr")
                except Exception:
                    audio, sr_out = kokoro.create(bmo_fr_spoken, voice="af_bella", speed=current_tts_speed)
                
                audio_flat = audio.squeeze()
                pitch_factor = 1.2599
                new_len = int(len(audio_flat) / pitch_factor)
                samples_shifted = scipy.signal.resample(audio_flat, new_len).astype(np.float32)
                sd.play(samples_shifted, sr_out)
                sd.wait()
            else:
                from gtts import gTTS
                tts = gTTS(text=bmo_fr, lang='fr', slow=False)
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

html_content = """
<!DOCTYPE html>
<html>
<head>
<style>
body { background-color: #122821; font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; overflow: hidden; }
#main-layout { display: flex; gap: 30px; align-items: center; }

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