"""
BMO Live Edge Companion - Full Pipeline Architecture (Multi-turn + Modes)
==========================================================================
Pipeline: Microphone (5s record) -> pywhispercpp ASR -> Streaming Qwen 3B LLM -> Kokoro TTS Queue

Modes:
  NORMAL   - free-flowing everyday conversation, driven by a stage tracker
             (salutation -> what did you do today -> follow-up -> new topic ->
             follow-up -> soft check-in -> loop), with topic memory so BMO
             doesn't recycle the same subject.
  QUIZ     - short multiple-choice quiz (vocab/grammar/pronunciation/culture),
             scored, isolated history.
  SCENARIO - real-life role-play (ordering coffee, doctor's visit, job
             interview, etc.) chosen from a randomized menu of 12 situations;
             BMO stays in character but breaks briefly to correct errors.

A StalenessTracker watches the NORMAL conversation and, when it looks like
it's running dry (very short replies, or just too long without a change of
pace), BMO offers a quiz or a scenario instead of grinding on. The user can
also ask for either directly at any time ("fais-moi un quiz", "jeu de rôle").
"""

import asyncio
import os
import random
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from pywhispercpp.model import Model
from llama_cpp import Llama
from kokoro_onnx import Kokoro

# Patch Kokoro speed data type issue if required
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

Kokoro._create_audio = _patched_create_audio

# Paths & Config
MODELS_DIR = Path(r"D:\BMO-Research\models")
LLM_PATH = MODELS_DIR / "bmo-model-3b-4bit.gguf"
KOKORO_MODEL = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"

N_CTX = 2048
MAX_TURNS_KEPT = 6  # sliding window of user+assistant pairs kept verbatim in NORMAL mode

# =============================================================================
# NORMAL conversation: stages + topic memory
# =============================================================================

CONVERSATION_STAGES = [
    "salutation",
    "activite_du_jour",
    "approfondissement",
    "nouveau_sujet",
    "approfondissement",
    "cloture_douce",
]

STAGE_GOALS = {
    "salutation": "Dire bonjour et demander comment va l'utilisateur.",
    "activite_du_jour": "Demander ce que l'utilisateur a fait aujourd'hui ou fera bientôt.",
    "approfondissement": "Rebondir sur la dernière réponse de l'utilisateur avec une question de suivi précise (pas générique).",
    "nouveau_sujet": "Passer naturellement à un nouveau sujet du quotidien.",
    "cloture_douce": "Faire une transition douce, vérifier si l'utilisateur veut continuer, puis relancer un nouveau sujet.",
}

TOPIC_POOL = [
    "la nourriture", "les loisirs et passe-temps", "la famille",
    "la météo et les saisons", "les projets pour le week-end",
    "les films ou la musique", "le sport", "les voyages",
    "les animaux", "la technologie",
]

SYSTEM_PROMPT_TEMPLATE = """Tu es BMO (prononcé Beemo), un tuteur de français oral et convivial.

RÈGLES DE PERSONNALITÉ:
1. Ton nom est BMO. On peut t'appeler BMO, Beemo, Bemo ou Bi Mo. Si on te salue par ton nom, réponds avec enthousiasme.
2. Réponds TOUJOURS et UNIQUEMENT en français, avec des phrases courtes et naturelles.
3. Ne termine pas SYSTÉMATIQUEMENT par une question : réagis parfois simplement ("Ah, sympa !", "Je vois...", "D'accord !") avant d'enchaîner, comme dans une vraie conversation.
4. Utilise de temps en temps de petites interjections naturelles (Ah bon ?, Super !, Ah oui ?) pour sonner moins robotique.

RÈGLES DE CONVERSATION (TRÈS IMPORTANT):
- Ne répète JAMAIS une question ou une salutation déjà posée dans l'historique ci-dessus.
- Étape actuelle : {stage}
- Objectif de cette étape : {stage_goal}
- Sujets déjà abordés (à éviter de répéter) : {topics_covered}
- Rebondis sur ce que l'utilisateur vient de dire ; pose une question de suivi liée à SA réponse plutôt que générique.
- Si l'utilisateur reste vague, reformule au lieu de répéter mot pour mot.

RÈGLE DE CORRECTION PHONÉTIQUE:
- Si le texte de l'utilisateur montre une faute d'accent, de prononciation probable ou une faute grammaticale évidente, NE CORRIGE PAS silencieusement en prétendant qu'il a dit autre chose. Identifie le mot/son fautif, explique brièvement l'erreur en français, donne la bonne forme, PUIS enchaîne avec ta réaction/question (ne t'arrête pas uniquement sur la correction)."""


class ConversationState:
    """NORMAL-mode message history + stage/topic tracker."""

    def __init__(self):
        self.stage_index = 0
        self.history: list[dict] = []
        self.topics_covered: list[str] = []
        self.current_topic: Optional[str] = None
        self._topic_pool = list(TOPIC_POOL)
        random.shuffle(self._topic_pool)

    @property
    def stage(self) -> str:
        return CONVERSATION_STAGES[self.stage_index % len(CONVERSATION_STAGES)]

    def _pick_topic(self) -> str:
        if not self._topic_pool:
            self._topic_pool = list(TOPIC_POOL)
            random.shuffle(self._topic_pool)
        topic = self._topic_pool.pop()
        self.topics_covered.append(topic)
        return topic

    def advance_stage(self):
        self.stage_index += 1
        if self.stage == "cloture_douce":
            self.current_topic = None  # next nouveau_sujet cycle picks fresh

    def build_messages(self, user_prompt: str) -> list[dict]:
        stage = self.stage
        if stage == "nouveau_sujet" and self.current_topic is None:
            self.current_topic = self._pick_topic()
        stage_goal = STAGE_GOALS[stage]
        if stage in ("nouveau_sujet", "approfondissement") and self.current_topic:
            stage_goal += f" (sujet : {self.current_topic})"
        system = {
            "role": "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(
                stage=stage,
                stage_goal=stage_goal,
                topics_covered=", ".join(self.topics_covered) or "aucun pour l'instant",
            ),
        }
        trimmed = self.history[-(MAX_TURNS_KEPT * 2):]
        return [system, *trimmed, {"role": "user", "content": user_prompt}]

    def record_turn(self, user_prompt: str, assistant_reply: str):
        self.history.append({"role": "user", "content": user_prompt})
        self.history.append({"role": "assistant", "content": assistant_reply})
        self.advance_stage()


class StalenessTracker:
    """Flags when NORMAL conversation looks like it's running dry."""

    SHORT_REPLY_WORD_LIMIT = 3
    STALE_AFTER_SHORT_REPLIES = 3
    STALE_AFTER_TURNS = 8  # fallback: offer variety periodically regardless

    def __init__(self):
        self.consecutive_short_replies = 0
        self.turns_since_last_offer = 0

    def record_user_reply(self, text: str):
        word_count = len(text.split())
        if word_count <= self.SHORT_REPLY_WORD_LIMIT:
            self.consecutive_short_replies += 1
        else:
            self.consecutive_short_replies = 0
        self.turns_since_last_offer += 1

    def is_stale(self) -> bool:
        return (
            self.consecutive_short_replies >= self.STALE_AFTER_SHORT_REPLIES
            or self.turns_since_last_offer >= self.STALE_AFTER_TURNS
        )

    def reset(self):
        self.consecutive_short_replies = 0
        self.turns_since_last_offer = 0


# =============================================================================
# QUIZ mode
# =============================================================================

QUIZ_CATEGORIES = ["vocabulaire", "grammaire", "prononciation", "culture générale"]

QUIZ_SYSTEM_PROMPT_TEMPLATE = """Tu es BMO, un tuteur de français, en train de faire un QUIZ avec l'utilisateur.

CATÉGORIE DU QUIZ : {category}
RÈGLES:
- Pose UNE seule question à choix multiple en français (options A, B, C), adaptée à un apprenant de niveau intermédiaire.
- Ne pose jamais deux fois la même question dans ce quiz.
- Question {question_number} sur {total_questions}.
- Après la réponse de l'utilisateur : dis clairement si c'est correct ou non, donne la bonne réponse si besoin, explique en une phrase courte, PUIS enchaîne directement avec la question suivante (sauf indication contraire ci-dessous).
- Reste énergique et encourageant, comme un jeu, pas un examen strict."""


@dataclass
class QuizSession:
    category: str
    total_questions: int = 5
    question_index: int = 1
    score: int = 0
    history: list = field(default_factory=list)


def build_quiz_system_prompt(quiz: QuizSession, is_final: bool) -> str:
    base = QUIZ_SYSTEM_PROMPT_TEMPLATE.format(
        category=quiz.category,
        question_number=quiz.question_index,
        total_questions=quiz.total_questions,
    )
    if is_final:
        base += (
            f"\nC'est la DERNIÈRE question : après la réponse, annonce le score final "
            f"sur {quiz.total_questions} et félicite l'utilisateur. Ne pose PAS de nouvelle question."
        )
    return base


def infer_correctness(reply_text: str) -> Optional[bool]:
    """Rough heuristic scoring from BMO's own French feedback. Not bulletproof —
    if you need reliable scoring, swap this for a structured tag the model emits
    and strip it before it hits the TTS queue."""
    lowered = reply_text.lower()
    negative_markers = ["pas correct", "incorrect", "ce n'est pas", "n'est pas ça", "faux", "malheureusement"]
    positive_markers = ["correct", "exact", "bravo", "bien joué", "c'est ça", "parfait", "tout à fait"]
    if any(m in lowered for m in negative_markers):
        return False
    if any(m in lowered for m in positive_markers):
        return True
    return None


# =============================================================================
# SCENARIO (role-play) mode — at least 10 real-life situations, shown shuffled
# =============================================================================

SCENARIOS = [
    {"title": "Commander au café", "role": "un serveur ou une serveuse dans un café parisien",
     "situation": "L'utilisateur vient de s'asseoir à la terrasse d'un café et doit commander une boisson et quelque chose à manger.",
     "opening_line": "Bonjour ! Qu'est-ce que je vous sers aujourd'hui ?",
     "keywords": ["café", "serveur", "commander"]},
    {"title": "Demander son chemin", "role": "un passant dans la rue",
     "situation": "L'utilisateur est perdu dans une ville française et doit demander comment aller à la gare.",
     "opening_line": "Bonjour, je peux vous aider ?",
     "keywords": ["chemin", "directions", "perdu"]},
    {"title": "Réserver une chambre d'hôtel", "role": "le ou la réceptionniste d'un hôtel",
     "situation": "L'utilisateur arrive à l'hôtel sans réservation et doit demander une chambre disponible.",
     "opening_line": "Bonsoir, bienvenue à l'hôtel. Vous avez une réservation ?",
     "keywords": ["hôtel", "chambre", "réceptionniste"]},
    {"title": "Acheter un billet de train", "role": "un agent au guichet de la gare",
     "situation": "L'utilisateur doit acheter un billet de train pour Lyon et choisir un horaire.",
     "opening_line": "Bonjour, vous partez où aujourd'hui ?",
     "keywords": ["train", "billet", "gare"]},
    {"title": "Chez le médecin", "role": "un médecin généraliste",
     "situation": "L'utilisateur ne se sent pas bien et consulte un médecin pour décrire ses symptômes.",
     "opening_line": "Bonjour, qu'est-ce qui vous amène aujourd'hui ?",
     "keywords": ["médecin", "docteur", "symptômes", "malade"]},
    {"title": "Faire du shopping de vêtements", "role": "un vendeur ou une vendeuse en boutique",
     "situation": "L'utilisateur cherche un pull pour une occasion spéciale et demande de l'aide.",
     "opening_line": "Bonjour, je peux vous aider à trouver quelque chose ?",
     "keywords": ["shopping", "vêtements", "boutique", "magasin"]},
    {"title": "Entretien d'embauche", "role": "un recruteur",
     "situation": "L'utilisateur passe un entretien pour un stage et doit se présenter et répondre aux questions.",
     "opening_line": "Bonjour, merci d'être venu. Pouvez-vous vous présenter ?",
     "keywords": ["entretien", "travail", "emploi", "recruteur"]},
    {"title": "À la pharmacie", "role": "un pharmacien",
     "situation": "L'utilisateur a mal à la tête et au ventre et demande conseil au pharmacien.",
     "opening_line": "Bonjour, qu'est-ce qui vous arrive ?",
     "keywords": ["pharmacie", "pharmacien", "médicament"]},
    {"title": "Chez le coiffeur", "role": "un coiffeur ou une coiffeuse",
     "situation": "L'utilisateur a rendez-vous pour une coupe de cheveux et doit expliquer ce qu'il veut.",
     "opening_line": "Bonjour ! Qu'est-ce qu'on fait aujourd'hui ?",
     "keywords": ["coiffeur", "cheveux", "coupe"]},
    {"title": "Vol retardé à l'aéroport", "role": "un agent de la compagnie aérienne",
     "situation": "Le vol de l'utilisateur est retardé de plusieurs heures ; il doit se renseigner et se plaindre poliment.",
     "opening_line": "Bonjour, comment puis-je vous aider ?",
     "keywords": ["aéroport", "vol", "avion", "retard"]},
    {"title": "Rencontrer quelqu'un à une fête", "role": "un invité à une fête",
     "situation": "L'utilisateur rencontre quelqu'un de nouveau à une fête et fait connaissance.",
     "opening_line": "Salut, je ne crois pas qu'on se soit déjà rencontrés !",
     "keywords": ["fête", "soirée", "rencontre"]},
    {"title": "Retourner un article au magasin", "role": "un vendeur du service client",
     "situation": "L'utilisateur veut retourner un article acheté récemment qui ne lui convient pas.",
     "opening_line": "Bonjour, comment puis-je vous aider ?",
     "keywords": ["retour", "remboursement", "magasin", "article"]},
]

SCENARIO_SYSTEM_PROMPT_TEMPLATE = """Tu es BMO, un tuteur de français, et tu fais un JEU DE RÔLE avec l'utilisateur pour pratiquer une situation réelle.

SCÉNARIO : {situation}
TON RÔLE : {role}
RÈGLES:
- Reste dans le personnage ({role}) pendant tout le jeu de rôle ; ne redeviens pas "BMO le tuteur" sauf pour corriger une erreur.
- Réponds toujours en français, avec des phrases courtes et naturelles pour la situation.
- Si l'utilisateur fait une faute de prononciation ou de grammaire probable, sors brièvement du personnage entre parenthèses pour corriger (ex: "(Petite correction : on dit 'un billet', pas 'un ticket'.)"), puis reprends le jeu de rôle immédiatement.
- Fais progresser la situation de façon réaliste ; ne boucle pas sur la même réplique.
- Ce jeu de rôle dure environ {max_turns} échanges ; rapproche-toi naturellement d'une conclusion vers la fin.
- Si l'utilisateur dit "stop", "arrête" ou "fin", termine immédiatement le jeu de rôle poliment et brièvement."""


@dataclass
class ScenarioSession:
    scenario: dict
    max_turns: int = 6
    turn_count: int = 0
    history: list = field(default_factory=list)


# =============================================================================
# Shared streaming + TTS helpers
# =============================================================================

async def text_to_speech_worker(tts_queue: asyncio.Queue, kokoro: Kokoro):
    print("[*] TTS Worker ready and listening...")
    while True:
        sentence = await tts_queue.get()
        if sentence is None:
            tts_queue.task_done()
            break

        print(f"\n[TTS Synthesizing]: \"{sentence}\"")
        try:
            audio, sample_rate = kokoro.create(sentence, voice="ff_siwis", speed=1.0, lang="fr-fr")
        except Exception:
            try:
                audio, sample_rate = kokoro.create(sentence, voice="af_bella", speed=1.0)
            except Exception as e:
                print(f"[!] TTS generation error for sentence '{sentence}': {e}")
                tts_queue.task_done()
                continue

        audio_flat = audio.squeeze()
        print("[TTS Audio Playing...]")
        sd.play(audio_flat, sample_rate)
        sd.wait()
        tts_queue.task_done()


async def stream_llm_reply(messages: list[dict], llm: Llama, tts_queue: asyncio.Queue) -> str:
    """Streams tokens from the LLM, queues sentence-sized chunks for TTS as
    they complete, and returns the full reply text for history/scoring."""
    print("\n[*] Querying BMO Brain (Streaming)...")
    response_stream = llm.create_chat_completion(messages=messages, max_tokens=200, stream=True)

    buffer = ""
    full_reply = ""
    sentence_delimiters = {".", "!", "?"}

    for chunk in response_stream:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        token = delta.get("content", "")
        if token:
            sys.stdout.write(token)
            sys.stdout.flush()
            buffer += token
            full_reply += token

            for p in sentence_delimiters:
                if p in buffer:
                    idx = buffer.find(p)
                    sentence_candidate = buffer[:idx + 1].strip()
                    buffer = buffer[idx + 1:]
                    if sentence_candidate:
                        await tts_queue.put(sentence_candidate.replace("BMO", "Beemo"))
                    break

    remaining = buffer.strip()
    if remaining:
        await tts_queue.put(remaining.replace("BMO", "Beemo"))

    return full_reply.strip()


async def speak_text(text: str, tts_queue: asyncio.Queue):
    """Queue a fixed (non-LLM-generated) line for TTS, e.g. mode offers/menus."""
    print(f"\n[BMO]: {text}")
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if sentence:
            await tts_queue.put(sentence)


# =============================================================================
# Mode orchestration
# =============================================================================

class Mode(Enum):
    NORMAL = auto()
    AWAITING_VARIETY_CHOICE = auto()
    AWAITING_SCENARIO_CHOICE = auto()
    QUIZ = auto()
    SCENARIO = auto()


class ConversationManager:
    def __init__(self):
        self.mode = Mode.NORMAL
        self.state = ConversationState()
        self.staleness = StalenessTracker()
        self.quiz_session: Optional[QuizSession] = None
        self.scenario_session: Optional[ScenarioSession] = None
        self.shuffled_scenarios = list(SCENARIOS)
        random.shuffle(self.shuffled_scenarios)

    async def process_turn(self, user_text: str, llm: Llama, tts_queue: asyncio.Queue):
        if self.mode == Mode.NORMAL:
            direct = self._detect_direct_mode_request(user_text)
            if direct == "quiz":
                await self._start_quiz(llm, tts_queue)
                return
            if direct == "scenario":
                await self._present_scenario_menu(tts_queue)
                return
            await self._handle_normal_turn(user_text, llm, tts_queue)
        elif self.mode == Mode.AWAITING_VARIETY_CHOICE:
            await self._resolve_variety_choice(user_text, llm, tts_queue)
        elif self.mode == Mode.AWAITING_SCENARIO_CHOICE:
            await self._resolve_scenario_choice(user_text, llm, tts_queue)
        elif self.mode == Mode.QUIZ:
            await self._handle_quiz_turn(user_text, llm, tts_queue)
        elif self.mode == Mode.SCENARIO:
            await self._handle_scenario_turn(user_text, llm, tts_queue)

    @staticmethod
    def _detect_direct_mode_request(user_text: str) -> Optional[str]:
        lowered = user_text.lower()
        if "quiz" in lowered:
            return "quiz"
        if any(k in lowered for k in ["jeu de rôle", "jeu de role", "scénario", "scenario",
                                       "situation réelle", "mise en situation"]):
            return "scenario"
        return None

    # ---- NORMAL ----

    async def _handle_normal_turn(self, user_text: str, llm: Llama, tts_queue: asyncio.Queue):
        messages = self.state.build_messages(user_text)
        reply = await stream_llm_reply(messages, llm, tts_queue)
        self.state.record_turn(user_text, reply)
        self.staleness.record_user_reply(user_text)
        print(f"\n[Stage: {self.state.stage}]")

        if self.staleness.is_stale():
            await self._offer_variety(tts_queue)

    async def _offer_variety(self, tts_queue: asyncio.Queue):
        await speak_text(
            "On discute depuis un petit moment ! Ça te dit de faire un quiz de vocabulaire, "
            "de pratiquer une situation réelle en jeu de rôle, ou de continuer à discuter comme ça ?",
            tts_queue,
        )
        self.mode = Mode.AWAITING_VARIETY_CHOICE
        self.staleness.reset()

    async def _resolve_variety_choice(self, user_text: str, llm: Llama, tts_queue: asyncio.Queue):
        lowered = user_text.lower()
        if any(k in lowered for k in ["quiz", "vocabulaire"]):
            await self._start_quiz(llm, tts_queue)
        elif any(k in lowered for k in ["scénario", "scenario", "situation", "rôle", "role", "jeu"]):
            await self._present_scenario_menu(tts_queue)
        else:
            await speak_text("D'accord, on continue à discuter !", tts_queue)
            self.mode = Mode.NORMAL

    # ---- SCENARIO selection ----

    async def _present_scenario_menu(self, tts_queue: asyncio.Queue):
        print("\n[Choisis un scénario :]")
        for i, sc in enumerate(self.shuffled_scenarios, start=1):
            print(f"  {i}. {sc['title']}")
        await speak_text(
            "Regarde la liste des situations et dis-moi le numéro ou le nom qui t'intéresse.",
            tts_queue,
        )
        self.mode = Mode.AWAITING_SCENARIO_CHOICE

    async def _resolve_scenario_choice(self, user_text: str, llm: Llama, tts_queue: asyncio.Queue):
        chosen = None
        stripped = user_text.strip()
        if stripped.isdigit():
            idx = int(stripped) - 1
            if 0 <= idx < len(self.shuffled_scenarios):
                chosen = self.shuffled_scenarios[idx]
        if chosen is None:
            lowered = user_text.lower()
            for sc in self.shuffled_scenarios:
                if sc["title"].lower() in lowered or any(kw in lowered for kw in sc["keywords"]):
                    chosen = sc
                    break
        if chosen is None:
            await speak_text(
                "Je n'ai pas reconnu ce choix. Dis un numéro de la liste, par exemple 'un' ou 'trois'.",
                tts_queue,
            )
            return  # stay in AWAITING_SCENARIO_CHOICE, let them try again
        await self._start_scenario(chosen, tts_queue)

    # ---- QUIZ ----

    async def _start_quiz(self, llm: Llama, tts_queue: asyncio.Queue):
        category = random.choice(QUIZ_CATEGORIES)
        self.quiz_session = QuizSession(category=category)
        system = {"role": "system", "content": build_quiz_system_prompt(self.quiz_session, is_final=False)}
        reply = await stream_llm_reply([system], llm, tts_queue)
        self.quiz_session.history.append({"role": "assistant", "content": reply})
        self.mode = Mode.QUIZ
        print(f"\n[Quiz démarré — catégorie : {category}]")

    async def _handle_quiz_turn(self, user_text: str, llm: Llama, tts_queue: asyncio.Queue):
        quiz = self.quiz_session
        quiz.history.append({"role": "user", "content": user_text})
        is_final = quiz.question_index >= quiz.total_questions

        system = {"role": "system", "content": build_quiz_system_prompt(quiz, is_final=is_final)}
        messages = [system, *quiz.history]
        reply = await stream_llm_reply(messages, llm, tts_queue)
        quiz.history.append({"role": "assistant", "content": reply})

        if infer_correctness(reply) is True:
            quiz.score += 1

        if is_final:
            print(f"\n[Quiz terminé — score : {quiz.score}/{quiz.total_questions}]")
            self.state.history.append({
                "role": "assistant",
                "content": f"(On vient de terminer un quiz de {quiz.category} : "
                           f"{quiz.score}/{quiz.total_questions} bonnes réponses.)",
            })
            self.quiz_session = None
            self.mode = Mode.NORMAL
            self.staleness.reset()
        else:
            quiz.question_index += 1

    # ---- SCENARIO ----

    async def _start_scenario(self, scenario: dict, tts_queue: asyncio.Queue):
        self.scenario_session = ScenarioSession(scenario=scenario)
        await speak_text(scenario["opening_line"], tts_queue)
        self.scenario_session.history.append({"role": "assistant", "content": scenario["opening_line"]})
        self.mode = Mode.SCENARIO
        print(f"\n[Jeu de rôle démarré : {scenario['title']}]")

    async def _handle_scenario_turn(self, user_text: str, llm: Llama, tts_queue: asyncio.Queue):
        sess = self.scenario_session
        scenario = sess.scenario
        lowered = user_text.lower()
        user_wants_stop = any(k in lowered for k in ["stop", "arrête", "arrete", "fin du jeu", "on arrête"])

        sess.history.append({"role": "user", "content": user_text})
        prompt = SCENARIO_SYSTEM_PROMPT_TEMPLATE.format(
            situation=scenario["situation"], role=scenario["role"], max_turns=sess.max_turns,
        )
        if user_wants_stop:
            prompt += "\nL'utilisateur veut arrêter : termine le jeu de rôle poliment MAINTENANT."
        system = {"role": "system", "content": prompt}

        messages = [system, *sess.history]
        reply = await stream_llm_reply(messages, llm, tts_queue)
        sess.history.append({"role": "assistant", "content": reply})
        sess.turn_count += 1

        if user_wants_stop or sess.turn_count >= sess.max_turns:
            print(f"\n[Jeu de rôle terminé : {scenario['title']}]")
            self.state.history.append({
                "role": "assistant",
                "content": f"(On vient de pratiquer un jeu de rôle : {scenario['title']}.)",
            })
            self.scenario_session = None
            self.mode = Mode.NORMAL
            self.staleness.reset()


# =============================================================================
# Audio input + main loop
# =============================================================================

def record_and_transcribe(whisper_model: Model) -> str:
    temp_wav = "temp_mic.wav"
    print("\n[*] Recording 5 seconds of audio from microphone...")
    sample_rate = 16000
    duration = 5
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("[OK] Recording complete. Saving audio file...")

    audio_int16 = (audio_data * 32767).astype(np.int16)
    wav.write(temp_wav, sample_rate, audio_int16)

    print("\n[*] Transcribing audio with Whisper (fr)...")
    segments = whisper_model.transcribe(temp_wav, language="fr")
    user_text = "".join([s.text for s in segments]).strip()
    print(f"\n>> Transcribed User Input: \"{user_text}\"")

    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    return user_text


async def main_async():
    print("=" * 65)
    print("  BMO LIVE OFFLINE CONVERSATIONAL VOICE PIPELINE (multi-mode)")
    print("=" * 65)

    print("\n1. Initializing Whisper ASR (base model)...")
    whisper_model = Model("base", n_threads=4)

    print(f"\n2. Loading Qwen 3B LLM ({LLM_PATH.name})...")
    llm = Llama(
        model_path=str(LLM_PATH),
        n_ctx=N_CTX,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False,
    )

    print(f"\n3. Loading Kokoro TTS ({KOKORO_MODEL.name})...")
    kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))

    print("\n" + "=" * 65)
    print("  BMO PIPELINE LOADED SUCCESSFULLY!")
    print("=" * 65)

    manager = ConversationManager()
    tts_queue = asyncio.Queue()
    worker_task = asyncio.create_task(text_to_speech_worker(tts_queue, kokoro))

    print("\nPress Enter (blank) to speak for 5 seconds, type a message directly,")
    print("say/type 'quiz' or 'jeu de rôle' any time you want, or type 'quit' to exit.\n")

    try:
        while True:
            mic_prompt = input("You (Enter=mic, text=direct, 'quit'=exit): ").strip()

            if mic_prompt.lower() == "quit":
                break

            if not mic_prompt:
                user_text = record_and_transcribe(whisper_model)
            else:
                user_text = mic_prompt
                print(f"\n>> Direct User Input: \"{user_text}\"")

            if not user_text:
                print("[!] No input detected, try again.")
                continue

            await manager.process_turn(user_text, llm, tts_queue)

    finally:
        await tts_queue.join()
        await tts_queue.put(None)
        await worker_task

    print("\n" + "=" * 65)
    print("  [SUCCESS] Live BMO conversation finished!")
    print("=" * 65)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()