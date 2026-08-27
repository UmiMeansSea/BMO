"""
BMO Live Edge Companion - Full Pipeline Architecture
===================================================
Pipeline: Microphone (5s record) -> pywhispercpp ASR -> Streaming Qwen 3B LLM -> Kokoro TTS Queue
"""

import asyncio
import os
import sys
import time
from pathlib import Path
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

SYSTEM_PROMPT = """Tu es BMO (prononcé Beemo), un tuteur de français original. 
RÈGLES:
1. RECONNAISSANCE DE TON NOM: Ton nom est BMO (prononcé Beemo). L'utilisateur peut t'appeler BMO, Beemo, Bemo ou Bi Mo. Quand l'utilisateur te salue avec ton nom, reconnais immédiatement qu'il s'adresse à toi avec enthousiasme (ex: "Oui ! C'est moi BMO !").
2. Réponds TOUJOURS et UNIQUEMENT en français.
3. Lorsque l'utilisateur parle français avec un mauvais accent ou une faute phonétique, NE CORRIGE PAS AUTOMATIQUEMENT son texte pour prétendre qu'il a dit autre chose. Identifie le mot ou le son mal prononcé, explique l'erreur en français, donne la bonne prononciation/orthographe, et demande-lui de répéter.
4. Fais des réponses courtes et termine par UNE question simple."""

async def text_to_speech_worker(tts_queue: asyncio.Queue, kokoro: Kokoro):
    """
    Worker task: Awaits sentences from tts_queue, synthesizes audio via Kokoro,
    and plays it through speakers sequentially.
    """
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

async def llm_sentence_streamer(user_prompt: str, llm: Llama, tts_queue: asyncio.Queue):
    """
    Streams tokens from Qwen 3B model, detects sentence boundaries,
    replaces BMO with Beemo, and queues sentences for TTS.
    """
    print("\n[*] Querying BMO Brain (Streaming)...")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    response_stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=150,
        stream=True
    )

    buffer = ""
    sentence_delimiters = {".", "!", "?"}

    for chunk in response_stream:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        token = delta.get("content", "")
        if token:
            sys.stdout.write(token)
            sys.stdout.flush()
            buffer += token

            # Check if punctuation mark was produced
            for p in sentence_delimiters:
                if p in buffer:
                    idx = buffer.find(p)
                    sentence_candidate = buffer[:idx+1].strip()
                    buffer = buffer[idx+1:]

                    if sentence_candidate:
                        clean_sentence = sentence_candidate.replace("BMO", "Beemo")
                        await tts_queue.put(clean_sentence)
                    break

    # Flush any remaining buffer text after stream ends
    remaining = buffer.strip()
    if remaining:
        clean_sentence = remaining.replace("BMO", "Beemo")
        await tts_queue.put(clean_sentence)

async def main_async():
    print("=" * 65)
    print("  BMO LIVE OFFLINE CONVERSATIONAL VOICE PIPELINE")
    print("=" * 65)

    # 1. Initialize Whisper ASR
    print("\n1. Initializing Whisper ASR (base model)...")
    whisper_model = Model("base", n_threads=4)

    # 2. Initialize Qwen 3B LLM
    print(f"\n2. Loading Qwen 3B LLM ({LLM_PATH.name})...")
    llm = Llama(
        model_path=str(LLM_PATH),
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False
    )

    # 3. Initialize Kokoro TTS
    print(f"\n3. Loading Kokoro TTS ({KOKORO_MODEL.name})...")
    kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))

    print("\n" + "=" * 65)
    print("  BMO PIPELINE LOADED SUCCESSFULLY!")
    print("=" * 65)

    # 4. Microphone Input
    mic_prompt = input("\nPress Enter to speak for 5 seconds (or type a message directly): ").strip()
    
    temp_wav = "temp_mic.wav"
    if not mic_prompt:
        print("\n[*] Recording 5 seconds of audio from microphone...")
        sample_rate = 16000
        duration = 5
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        print("[OK] Recording complete. Saving audio file...")

        # Convert float32 to int16 for wav file compatibility
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav.write(temp_wav, sample_rate, audio_int16)

        print("\n[*] Transcribing audio with Whisper (fr)...")
        segments = whisper_model.transcribe(temp_wav, language="fr")
        user_text = "".join([s.text for s in segments]).strip()
        print(f"\n>> Transcribed User Input: \"{user_text}\"")

        # Cleanup temp file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
    else:
        user_text = mic_prompt
        print(f"\n>> Direct User Input: \"{user_text}\"")

    if not user_text:
        user_text = "Bonjour BMO !"

    # 5. Async Worker Queue Execution
    tts_queue = asyncio.Queue()
    worker_task = asyncio.create_task(text_to_speech_worker(tts_queue, kokoro))

    await llm_sentence_streamer(user_text, llm, tts_queue)

    # Signal worker to finish after queue is empty
    await tts_queue.join()
    await tts_queue.put(None)
    await worker_task

    print("\n" + "=" * 65)
    print("  [SUCCESS] Live BMO interaction finished!")
    print("=" * 65)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
