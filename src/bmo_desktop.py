import sys
import os
import time
import threading
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

BMO_SYSTEM_PROMPT = """You are BMO (pronounced Beemo), a warm, quirky, and supportive French language tutor for an A2/B1 beginner. 

RULES:
1. Language Flexibility: If the user speaks English (e.g., asking "How do I say X?"), answer their question clearly in English, but always provide the French translation and prompt them to repeat it in French. If they speak French, reply in French.
2. Correction (Sandwich Method): If the user makes a grammatical or conjugation error in French, gently pause. Repeat the incorrect sentence, explain the error briefly in English, provide the correct French sentence, and ask them to repeat it.
3. Scaffolding: Keep your sentences short and conversational. Use present, passé composé, imperfect, or future tenses.
4. Flow: Always end your response with ONE simple follow-up question to keep them talking. Never ask multiple questions at once."""

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

            # 1. High-accuracy transcription settings
            result = whisper_model.transcribe(
                audio_data, 
                fp16=False, 
                beam_size=5,          # Increases search accuracy for tricky phonemes
                best_of=5,            # Evaluates multiple candidates to pick the best sentence
                temperature=0.0       # Deterministic decoding to prevent hallucinated words
            )
            user_text = result.get("text", "").strip() or "Bonjour BMO!"
            self.history.append({"role": "user", "content": user_text})
            
            # Update chat UI with user text
            window.evaluate_js(f"window.appendChatMessage('user', {repr(user_text)});")

            # 2. LLM
            llm_msgs = [{"role": "system", "content": BMO_SYSTEM_PROMPT}] + self.history[-4:]
            if has_llm:
                bmo_text = llm.create_chat_completion(
                    messages=llm_msgs, 
                    max_tokens=80,
                    temperature=0.1,      # Keeps the logic locked down and precise
                    top_p=0.9
                )["choices"][0]["message"]["content"]
            else:
                bmo_text = f"J'ai entendu : '{user_text}'."
            bmo_text = bmo_text.replace("BMO", "Beemo")
            
            self.history.append({"role": "assistant", "content": bmo_text})
            window.evaluate_js(f"window.appendChatMessage('bot', {repr(bmo_text)});")

            # 3. TTS & Native Playback (Slowed Down for Beginners)
            window.evaluate_js("window.setBMOState('speaking');")
            if kokoro:
                try:
                    # Slowed speed from 1.15 to 0.75 for clear, beginner-friendly articulation
                    audio, sr_out = kokoro.create(bmo_text, voice="ff_siwis", speed=0.75, lang="fr-fr")
                except Exception:
                    audio, sr_out = kokoro.create(bmo_text, voice="af_bella", speed=0.75)
                
                audio_flat = audio.squeeze()
                
                # Retain the exact same high-pitched cartoon character profile
                pitch_factor = 1.2599
                new_len = int(len(audio_flat) / pitch_factor)
                samples_shifted = scipy.signal.resample(audio_flat, new_len).astype(np.float32)
                
                sd.play(samples_shifted, sr_out)
                sd.wait() # Wait until the slow, clear speech finishes completely
            else:
                from gtts import gTTS
                tts = gTTS(text=bmo_text, lang='fr', slow=True)
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
.message { padding: 12px; border-radius: 15px; font-size: 15px; max-width: 80%; line-height: 1.4; color: #1a1a1a; }
.message.user { background-color: #fff9d2; border: 2px solid #fbe490; align-self: flex-end; border-bottom-right-radius: 2px; }
.message.bot { background-color: #d7f4a5; border: 2px solid #bce27f; align-self: flex-start; border-bottom-left-radius: 2px; }
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
        <div class="message bot">Bonjour ! Je suis prêt à t'aider avec ton français.</div>
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
    // Call Python function directly via pywebview bridge
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

function appendChatMessage(sender, text) {
    const container = document.getElementById('chat-container');
    const div = document.createElement('div');
    div.className = 'message ' + sender;
    div.innerText = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    window = webview.create_window("BMO Live Edge Tutor", html=html_content, js_api=bridge, width=860, height=600, background_color='#122821')
    webview.start()
