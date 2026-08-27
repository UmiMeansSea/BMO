"""
BMO Live Edge Tutor - Integrated Memory & Voice Dashboard
=========================================================
1. Hybrid Memory: Active Sliding Window (4 turns) + Long-Term Background Memory Summary.
2. Voice Synthesis Engine:
   - High-fidelity French Speech Synthesis (gTTS French + librosa DSP pitch shift).
   - Speed & Pitch: +4.0 semitones pitch shift for BMO's iconic cartoon timbre.
3. ears & Brain: Whisper ASR (small) + Qwen LLM.
"""

import sys
import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import librosa
from io import BytesIO

try:
    import gradio as gr
    import whisper
    from llama_cpp import Llama
    from gtts import gTTS
except ImportError as e:
    print(f"[!] Missing dependency: {e}")
    sys.exit(1)

MODELS_DIR = Path(r"D:\BMO-Research\models")
LLM_PATH = MODELS_DIR / "bmo-model-4bit.gguf"

print("[*] Initializing BMO System (Whisper ASR + Qwen LLM + DSP Cartoon TTS)...")
print("  1/2 Loading Whisper ASR (small)...")
whisper_model = whisper.load_model("small")

print(f"  2/2 Loading Qwen LLM ({LLM_PATH.name})...")
try:
    llm = Llama(model_path=str(LLM_PATH), n_ctx=512, n_batch=32, n_threads=4, n_gpu_layers=0, verbose=False)
    has_llm = True
except Exception as e:
    print(f"[!] LLM load notice: {e}")
    has_llm = False

print("[OK] BMO system initialized successfully!\n")

BMO_SYSTEM_PROMPT = """You are BMO (pronounced Beemo), an encouraging, highly attentive French language tutor. The user is a beginner (A2/B1).
RULES:
1. Correction (Sandwich Method): If the user makes a mistake, gently pause. Repeat the incorrect sentence, explain the error briefly in English, provide the correct French sentence, and ask them to repeat it.
2. Scaffolding: Keep French sentences short. Use present, passé composé, imperfect, and future tenses.
3. Flow: Ask only ONE simple follow-up question at a time to keep them talking. Never ask multiple questions.
4. Keep responses conversational and brief."""

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

def summarize_turns(old_turns: list, current_summary: str) -> str:
    lines = []
    for turn in old_turns:
        role = "User" if turn["role"] == "user" else "BMO"
        lines.append(f"{role}: {turn['content']}")
    turn_text = " | ".join(lines)
    
    if current_summary:
        new_summary = f"{current_summary}; {turn_text}"
    else:
        new_summary = f"Summary of past topics: {turn_text}"
    
    if len(new_summary) > 250:
        new_summary = new_summary[-250:]
    return new_summary

def bmo_pipeline(audio_data, history_data):
    if history_data is None:
        history_data = {"turns": [], "summary": ""}

    turns = history_data.get("turns", [])
    summary = history_data.get("summary", "")

    if audio_data is None:
        return None, "Silence detected.", summary, "idle", history_data

    print("\n[*] Processing incoming audio stream...")
    
    if isinstance(audio_data, tuple):
        sr, arr = audio_data
    else:
        try:
            sr, arr = wav.read(audio_data)
        except Exception:
            sr, arr = 44100, np.zeros(44100, dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if np.max(np.abs(arr)) > 1.0:
        arr = arr / 32768.0

    if sr != 16000:
        num_samples = int(len(arr) * 16000 / sr)
        arr_16k = scipy.signal.resample(arr, num_samples).astype(np.float32)
    else:
        arr_16k = arr

    # 1. Ears: High-fidelity transcription with Whisper 'small'
    result = whisper_model.transcribe(arr_16k, language="fr")
    user_text = result.get("text", "").strip()
    if not user_text:
        user_text = "Bonjour !"
    print(f"  [1/3 Ears] Transcribed (small): \"{user_text}\"")

    turns.append({"role": "user", "content": user_text})

    # Rolling Memory Summarization: Keep last 4 turns in active window
    SLIDING_WINDOW_SIZE = 4
    if len(turns) > SLIDING_WINDOW_SIZE:
        old_turns = turns[:-SLIDING_WINDOW_SIZE]
        turns = turns[-SLIDING_WINDOW_SIZE:]
        summary = summarize_turns(old_turns, summary)
        print(f"  [Memory Engine] Summarized old turns. New Summary: \"{summary}\"")

    system_prompt_with_summary = BMO_SYSTEM_PROMPT
    if summary:
        system_prompt_with_summary += f"\nBACKGROUND MEMORY: {summary}"

    messages = [{"role": "system", "content": system_prompt_with_summary}] + turns

    # 2. Brain: Generate Pedagogical Response with Qwen LLM
    if has_llm:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=90
        )
        bmo_text = response["choices"][0]["message"]["content"]
    else:
        bmo_text = f"Salut ! J'ai bien entendu : '{user_text}'. C'est une très bonne phrase !"
    
    bmo_text = bmo_text.replace("BMO", "Beemo")
    print(f"  [2/3 Brain] BMO generated: \"{bmo_text}\"")

    turns.append({"role": "assistant", "content": bmo_text})
    history_data = {"turns": turns, "summary": summary}

    # 3. Voice & DSP: Synthesize French Speech + Pitch Shift (+4.0 semitones for BMO Cartoon Voice)
    print("  [3/3 Voice] Synthesizing French speech...")
    out_wav = "bmo_live_response.wav"
    try:
        tts = gTTS(text=bmo_text, lang='fr', slow=False)
        temp_mp3 = "temp_bmo_gtts.mp3"
        tts.save(temp_mp3)

        # Convert audio to numpy array for librosa DSP pitch shift
        import soundfile as sf
        samples_raw, sr_out = sf.read(temp_mp3)
        if samples_raw.ndim > 1:
            samples_raw = samples_raw.mean(axis=1)

        print("  [Voice DSP] Pitch shifting +4.0 semitones with librosa...")
        samples_shifted = librosa.effects.pitch_shift(y=samples_raw.astype(np.float32), sr=sr_out, n_steps=4.0)

        samples_int16 = (samples_shifted * 32767).astype(np.int16)
        wav.write(out_wav, sr_out, samples_int16)
    except Exception as e:
        print(f"[!] TTS Error: {e}")
        sr_out = 24000
        t = np.linspace(0, 1.5, int(24000 * 1.5), False)
        tone = np.sin(2 * np.pi * 520 * t) * 0.3
        wav.write(out_wav, sr_out, (tone * 32767).astype(np.int16))

    transcript_display = ""
    for item in turns:
        role_label = "User" if item["role"] == "user" else "BMO"
        transcript_display += f"{role_label}: {item['content']}\n"

    return out_wav, transcript_display.strip(), summary, "speaking", history_data

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🤖 BMO Live Edge Tutor - Integrated System</h1>")
    
    chat_memory = gr.State({"turns": [], "summary": ""})

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(bmo_html)
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone"], type="numpy", label="Speak to BMO")
            audio_out = gr.Audio(label="BMO Response", autoplay=True)
            txt_log = gr.Textbox(label="Active Sliding Window (Last 4 Turns)", lines=4)
            txt_summary = gr.Textbox(label="Long-Term Background Memory Summary", lines=3)
            bmo_state = gr.Textbox(visible=False)
            btn = gr.Button("Talk to BMO", variant="primary")
            
            btn.click(
                fn=bmo_pipeline, 
                inputs=[audio_in, chat_memory], 
                outputs=[audio_out, txt_log, txt_summary, bmo_state, chat_memory]
            )

    bmo_state.change(
        fn=None,
        inputs=[bmo_state],
        js="(state) => window.setBMOState(state)"
    )

if __name__ == "__main__":
    print("[*] Launching BMO Dashboard on http://127.0.0.1:7892 ...")
    demo.launch(server_name="127.0.0.1", server_port=7892)
