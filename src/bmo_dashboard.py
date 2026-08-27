"""
BMO Gradio Dashboard - Complete Integrated System (Pure Python Audio Pipeline)
==============================================================================
Combines:
1. Animated State Machine UI (Idle, Listening, Thinking, Speaking).
2. Pure Python Audio Pipeline (scipy WAV read/resample -> Whisper/LLM/Kokoro).
3. DSP Cartoon Voice Timbre (+4.0 semitone pitch shift & 1.15x speed).
"""

import sys
import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import librosa

try:
    import gradio as gr
    import whisper
    from llama_cpp import Llama
    from kokoro_onnx import Kokoro
except ImportError as e:
    print(f"[!] Missing dependency: {e}")
    sys.exit(1)

# Patch Kokoro speed data type issue if needed
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

MODELS_DIR = Path(r"D:\BMO-Research\models")
LLM_PATH = MODELS_DIR / "bmo-model-4bit.gguf"
KOKORO_MODEL = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"

print("[*] Initializing complete integrated BMO system...")
print("  1/3 Loading Whisper ASR (tiny)...")
whisper_model = whisper.load_model("tiny")

print(f"  2/3 Loading Qwen LLM ({LLM_PATH.name})...")
try:
    llm = Llama(model_path=str(LLM_PATH), n_ctx=256, n_batch=32, n_threads=4, n_gpu_layers=0, verbose=False)
    has_llm = True
except Exception as e:
    print(f"[!] LLM load notice: {e}")
    has_llm = False

print(f"  3/3 Loading Kokoro TTS ({KOKORO_MODEL.name})...")
try:
    kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    has_kokoro = True
except Exception as e:
    print(f"[!] Kokoro load notice: {e}")
    has_kokoro = False

print("[OK] Integrated BMO pipeline initialized!\n")

css_styles = """
#bmo-container { background-color: #3ca993; width: 400px; height: 520px; border: 4px solid #000; border-radius: 20px; position: relative; margin: 0 auto; }
.bmo-screen { background-color: #b7efcc; width: 350px; height: 220px; border: 4px solid #000; border-radius: 15px; position: absolute; top: 20px; left: 21px; box-sizing: border-box; transition: background 0.2s ease; overflow: hidden; }

/* State 1: Listening */
.bmo-screen.listening { background-color: #112a20; }

/* State 2: Thinking */
.bmo-screen.thinking { background-color: #2c5e50; }

.bmo-screen.listening .bmo-eye, .bmo-screen.listening .bmo-mouth,
.bmo-screen.thinking .bmo-eye, .bmo-screen.thinking .bmo-mouth {
    display: none;
}

.bmo-eye { background: #000; width: 16px; height: 16px; border-radius: 50%; position: absolute; top: 35%; animation: blink 4s infinite; }
.bmo-eye.left { left: 25%; }
.bmo-eye.right { right: 25%; }
@keyframes blink {
    0%, 96%, 98%, 100% { transform: scaleY(1); }
    97% { transform: scaleY(0.1); }
}

.bmo-mouth {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translateX(-50%);
    width: 65px;
    height: 32px;
    border: 4px solid #000;
    border-top: transparent;
    border-left: transparent;
    border-right: transparent;
    border-radius: 0 0 50px 50px;
    transition: all 0.15s ease;
}

.bmo-mouth.speaking {
    animation: talk-anim 0.25s infinite alternate ease-in-out;
    background: #112a20;
    border: 3px solid #000;
}

@keyframes talk-anim {
    0% { height: 12px; width: 35px; border-radius: 20px; top: 58%; }
    50% { height: 28px; width: 48px; border-radius: 50% 50% 45% 45%; top: 52%; }
    100% { height: 38px; width: 52px; border-radius: 40% 40% 50% 50%; top: 50%; }
}

.bmo-slot { position: absolute; background: #112a20; border: 4px solid #000; width: 200px; height: 15px; top: 260px; left: 30px; }
.bmo-sbc { position: absolute; background: #0000ff; border: 4px solid #000; width: 20px; height: 20px; border-radius: 50%; top: 255px; right: 50px; }
.bmo-dpad-svg { position: absolute; top: 310px; left: 30px; width: 100px; height: 100px; }
.bmo-triangle-svg { position: absolute; top: 310px; right: 110px; width: 40px; height: 40px; }
.bmo-gc { position: absolute; background: #33ff33; border: 4px solid #000; width: 25px; height: 25px; border-radius: 50%; top: 325px; right: 50px; }
.bmo-rc { position: absolute; background: #ff0000; border: 4px solid #000; width: 60px; height: 60px; border-radius: 50%; bottom: 60px; right: 50px; cursor: pointer; transition: transform 0.1s; }
.bmo-rc:active { transform: scale(0.92); }
.bmo-pill { position: absolute; background: #0000ff; border: 4px solid #000; width: 45px; height: 15px; border-radius: 15px; bottom: 25px; }
.bmo-pill.p1 { left: 40px; }
.bmo-pill.p2 { left: 105px; }
"""

bmo_html = f"""
<style>
{css_styles}
</style>
<div id="bmo-container">
    <div id="bmo-screen" class="bmo-screen">
        <div class="bmo-eye left"></div>
        <div class="bmo-eye right"></div>
        <div id="bmo-mouth" class="bmo-mouth"></div>
    </div>
    <div class="bmo-slot"></div><div class="bmo-sbc"></div>
    <svg class="bmo-dpad-svg" viewBox="0 0 100 100"><path d="M 35 5 L 65 5 L 65 35 L 95 35 L 95 65 L 65 65 L 65 95 L 35 95 L 35 65 L 5 65 L 5 35 L 35 35 Z" fill="#ffcc00" stroke="#000" stroke-width="4" stroke-linejoin="round"/></svg>
    <svg class="bmo-triangle-svg" viewBox="0 0 100 100"><polygon points="50,10 90,90 10,90" fill="#00ccff" stroke="#000" stroke-width="6" stroke-linejoin="round"/></svg>
    <div class="bmo-gc"></div>
    <div class="bmo-rc" onclick="window.setBMOState('listening')"></div>
    <div class="bmo-pill p1"></div><div class="bmo-pill p2"></div>
</div>
<script>
    window.setBMOState = function(state) {{
        const screen = document.getElementById('bmo-screen');
        const mouth = document.getElementById('bmo-mouth');
        if (!screen || !mouth) return;

        screen.className = 'bmo-screen';
        mouth.classList.remove('speaking');

        if (state === 'listening') {{
            screen.classList.add('listening');
        }} else if (state === 'thinking') {{
            screen.classList.add('thinking');
        }} else if (state === 'speaking') {{
            mouth.classList.add('speaking');
        }}
    }};
</script>
"""

def bmo_pipeline(audio_data):
    if audio_data is None:
        return None, "Silence detected.", "idle"

    print("\n[*] Processing incoming audio stream...")
    
    # Extract audio array directly from Gradio tuple or read WAV file via scipy
    if isinstance(audio_data, tuple):
        sr, arr = audio_data
    else:
        try:
            sr, arr = wav.read(audio_data)
        except Exception:
            sr, arr = 44100, np.zeros(44100, dtype=np.float32)

    # Convert to 1D float32 array normalized to [-1.0, 1.0]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if np.max(np.abs(arr)) > 1.0:
        arr = arr / 32768.0

    # Resample float32 audio to 16,000 Hz directly in Python (bypassing ffmpeg sub-process calls)
    if sr != 16000:
        num_samples = int(len(arr) * 16000 / sr)
        arr_16k = scipy.signal.resample(arr, num_samples).astype(np.float32)
    else:
        arr_16k = arr

    # 1. Ears: Transcribe directly from numpy array via PyTorch Whisper ASR
    result = whisper_model.transcribe(arr_16k, language="fr")
    user_text = result.get("text", "").strip()
    if not user_text:
        user_text = "Bonjour !"
    print(f"  [Ears] Transcribed: \"{user_text}\"")

    # 2. Brain: Generate response with Qwen LLM
    if has_llm:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are BMO (pronounced Beemo), a quirky French language tutor. Keep responses short and conversational, correcting errors gently."},
                {"role": "user", "content": user_text}
            ],
            max_tokens=80
        )
        bmo_text = response["choices"][0]["message"]["content"]
    else:
        bmo_text = f"Salut ! J'ai entendu : '{user_text}'. C'est du très bon français !"
    
    bmo_text = bmo_text.replace("BMO", "Beemo")
    print(f"  [Brain] BMO generated: \"{bmo_text}\"")

    # 3. Voice & DSP: Synthesize with Kokoro (1.15x speed) & Pitch Shift (+4.0 semitones)
    sr_out = 24000
    if has_kokoro:
        try:
            samples, sr_out = kokoro.create(bmo_text, voice="ff_siwis", speed=1.15, lang="fr-fr")
            samples = samples.squeeze().astype(np.float32)
        except Exception:
            t = np.linspace(0, 1.5, int(24000 * 1.5), False)
            samples = np.sin(2 * np.pi * 520 * t).astype(np.float32) * 0.3
    else:
        t = np.linspace(0, 1.5, int(24000 * 1.5), False)
        samples = np.sin(2 * np.pi * 520 * t).astype(np.float32) * 0.3

    print("  [Voice DSP] Pitch shifting +4.0 semitones with librosa...")
    samples_shifted = librosa.effects.pitch_shift(y=samples, sr=sr_out, n_steps=4.0)

    out_wav = "bmo_live_response.wav"
    samples_int16 = (samples_shifted * 32767).astype(np.int16)
    wav.write(out_wav, sr_out, samples_int16)

    return out_wav, f"User: {user_text}\nBMO: {bmo_text}", "speaking"

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🤖 BMO Live Edge Tutor - Integrated Dashboard</h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(bmo_html)
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone"], type="numpy", label="Speak to BMO")
            audio_out = gr.Audio(label="BMO Response", autoplay=True)
            txt_log = gr.Textbox(label="Conversation Log", lines=4)
            bmo_state = gr.Textbox(visible=False)
            btn = gr.Button("Talk to BMO", variant="primary")
            
            btn.click(
                fn=bmo_pipeline, 
                inputs=[audio_in], 
                outputs=[audio_out, txt_log, bmo_state]
            )

    bmo_state.change(
        fn=None,
        inputs=[bmo_state],
        js="(state) => window.setBMOState(state)"
    )

if __name__ == "__main__":
    print("[*] Launching Pure-Python BMO Dashboard on http://127.0.0.1:7882 ...")
    demo.launch(server_name="127.0.0.1", server_port=7882)
