import sys
import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import sounddevice as sd
import traceback

try:
    import gradio as gr
    import whisper
    from llama_cpp import Llama
    from kokoro_onnx import Kokoro
except ImportError as e:
    print(f"[!] Missing dependency: {e}")
    sys.exit(1)

# Patch Kokoro speed data type issue
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
    from kokoro_onnx import Kokoro
    Kokoro._create_audio = _patched_create_audio
except ImportError:
    pass

MODELS_DIR = Path(r"D:\BMO-Research\models")
LLM_PATH = MODELS_DIR / "bmo-model-4bit.gguf"
KOKORO_MODEL = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"

print("[*] Initializing BMO Hardware-Native System...")
whisper_model = whisper.load_model("small")

try:
    llm = Llama(model_path=str(LLM_PATH), n_ctx=512, n_threads=4, verbose=False)
    has_llm = True
except Exception as e:
    print(f"[!] LLM load notice: {e}")
    has_llm = False

try:
    kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    print("[OK] Kokoro ONNX loaded successfully!")
except Exception as e:
    print(f"[!] Kokoro init notice: {e}")
    kokoro = None

# --- GLOBAL HARDWARE MICROPHONE BUFFER ---
AUDIO_BUFFER = []
IS_RECORDING = False

def hardware_audio_callback(indata, frames, time, status):
    if IS_RECORDING:
        AUDIO_BUFFER.append(indata.copy())

try:
    sd_stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=hardware_audio_callback)
    sd_stream.start()
    print("[OK] Direct Hardware Microphone Stream Active!")
except Exception as e:
    print(f"[CRITICAL] Failed to access microphone: {e}")

BMO_SYSTEM_PROMPT = "You are BMO (pronounced Beemo), an encouraging French tutor. Keep sentences short. Ask only ONE simple follow-up question. Use the sandwich method for corrections."

css_styles = """
#bmo-container { background-color: #3ca993; width: 400px; height: 520px; border: 4px solid #000; border-radius: 20px; position: relative; margin: 0 auto; }
.bmo-screen { background-color: #b7efcc; width: 350px; height: 220px; border: 4px solid #000; border-radius: 15px; position: absolute; top: 20px; left: 21px; box-sizing: border-box; transition: background 0.2s ease; overflow: hidden; }
.bmo-eye { background: #000; width: 16px; height: 16px; border-radius: 50%; position: absolute; top: 35%; animation: blink 4s infinite; }
.bmo-eye.left { left: 25%; } .bmo-eye.right { right: 25%; }
@keyframes blink { 0%, 96%, 98%, 100% { transform: scaleY(1); } 97% { transform: scaleY(0.1); } }
.bmo-mouth { position: absolute; top: 50%; left: 50%; transform: translateX(-50%); width: 65px; height: 32px; border: 4px solid #000; border-top: transparent; border-left: transparent; border-right: transparent; border-radius: 0 0 50px 50px; transition: all 0.15s ease; }

/* States */
.bmo-waveform { display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 220px; height: 80px; align-items: center; justify-content: space-between; }
.bmo-screen.listening { background-color: #0b1f17; }
.bmo-screen.listening .bmo-eye, .bmo-screen.listening .bmo-mouth { display: none; }
.bmo-screen.listening .bmo-waveform { display: flex; }
.wave-bar { width: 10px; background: #33ff99; border-radius: 5px; animation: wave-anim 0.6s infinite alternate ease-in-out; }
.wave-bar:nth-child(1) { height: 30px; animation-delay: 0.1s; } .wave-bar:nth-child(2) { height: 60px; animation-delay: 0.3s; }
.wave-bar:nth-child(3) { height: 40px; animation-delay: 0.2s; } .wave-bar:nth-child(4) { height: 75px; animation-delay: 0.4s; }
.wave-bar:nth-child(5) { height: 50px; animation-delay: 0.15s; } .wave-bar:nth-child(6) { height: 65px; animation-delay: 0.35s; }
.wave-bar:nth-child(7) { height: 35px; animation-delay: 0.25s; }
@keyframes wave-anim { 0% { transform: scaleY(0.3); background: #33ff99; } 100% { transform: scaleY(1.3); background: #66ffff; } }

.bmo-thinking-box { display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); flex-direction: column; align-items: center; justify-content: center; width: 85%; }
.bmo-thinking-grid { display: grid; width: 70px; height: 70px; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 10px; }
.bmo-screen.thinking { background-color: #1a4237; }
.bmo-screen.thinking .bmo-eye, .bmo-screen.thinking .bmo-mouth, .bmo-screen.thinking .bmo-waveform { display: none; }
.bmo-screen.thinking .bmo-thinking-box { display: flex; }
.grid-dot { background: #ffcc00; border-radius: 4px; animation: dot-pulse 0.8s infinite alternate; }
.grid-dot:nth-child(odd) { animation-delay: 0.2s; } .grid-dot:nth-child(even) { animation-delay: 0.5s; }
@keyframes dot-pulse { 0% { opacity: 0.2; transform: scale(0.8); } 100% { opacity: 1; transform: scale(1.1); background: #ff5500; } }

.bmo-progress-container { width: 100%; background: #0c211b; border: 2px solid #000; border-radius: 10px; height: 16px; position: relative; overflow: hidden; }
.bmo-progress-fill { background: linear-gradient(90deg, #33ff99, #66ffff); width: 0%; height: 100%; transition: width 0.2s ease-in-out; }
.bmo-progress-text { font-family: monospace; font-size: 13px; font-weight: bold; color: #ffcc00; margin-top: 5px; text-shadow: 1px 1px 0 #000; }

.bmo-mouth.speaking { animation: talk-anim 0.22s infinite alternate ease-in-out; background: #112a20; border: 3px solid #000; }
@keyframes talk-anim { 0% { height: 12px; width: 35px; border-radius: 20px; top: 58%; } 50% { height: 28px; width: 48px; border-radius: 50% 50% 45% 45%; top: 52%; } 100% { height: 38px; width: 52px; border-radius: 40% 40% 50% 50%; top: 50%; } }

/* Hardware Chassis */
.bmo-slot { position: absolute; background: #112a20; border: 4px solid #000; width: 200px; height: 15px; top: 260px; left: 30px; }
.bmo-sbc { position: absolute; background: #0000ff; border: 4px solid #000; width: 20px; height: 20px; border-radius: 50%; top: 255px; right: 50px; }
.bmo-dpad-svg { position: absolute; top: 310px; left: 30px; width: 100px; height: 100px; }
.bmo-triangle-svg { position: absolute; top: 310px; right: 110px; width: 40px; height: 40px; }
.bmo-gc { position: absolute; background: #33ff33; border: 4px solid #000; width: 25px; height: 25px; border-radius: 50%; top: 325px; right: 50px; }
.bmo-rc { position: absolute; background: #ff0000; border: 4px solid #000; width: 60px; height: 60px; border-radius: 50%; bottom: 60px; right: 50px; cursor: pointer; transition: transform 0.1s; }
.bmo-rc:active { transform: scale(0.92); }
.bmo-pill { position: absolute; background: #0000ff; border: 4px solid #000; width: 45px; height: 15px; border-radius: 15px; bottom: 25px; } .bmo-pill.p1 { left: 40px; } .bmo-pill.p2 { left: 105px; }

/* Chat UI */
.cloak-audio { position: absolute !important; top: -9999px !important; left: -9999px !important; opacity: 0; pointer-events: none; height: 0px !important; }
.cute-chat, .cute-chat .wrap, .cute-chat .bubble-wrap { background-color: #f4fce8 !important; border-radius: 20px !important; }
.cute-chat .message * { color: #1a1a1a !important; font-weight: 500 !important; font-size: 16px !important; margin: 0 !important; }
.cute-chat .message.user { background-color: #fff9d2 !important; border: 2px solid #fbe490 !important; border-radius: 20px 20px 0 20px !important; padding: 12px !important; }
.cute-chat .message.bot { background-color: #d7f4a5 !important; border: 2px solid #bce27f !important; border-radius: 20px 20px 20px 0 !important; padding: 12px !important; }
"""

head_js = """
<script>
    window.isBmoRecording = false;

    window.playBmoBeep = function(freq) {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine'; osc.frequency.setValueAtTime(freq, ctx.currentTime);
            gain.gain.setValueAtTime(0.1, ctx.currentTime);
            osc.connect(gain); gain.connect(ctx.destination);
            osc.start(); osc.stop(ctx.currentTime + 0.15);
        } catch(e) {}
    };

    window.toggleBmoRecord = function() {
        // Safely grab the button directly by its Gradio elem_id
        let triggerBtn = document.getElementById('bmo-trigger');
        
        // Fallback in case a future Gradio update wraps the button in a div
        if (triggerBtn && triggerBtn.tagName !== 'BUTTON') {
            triggerBtn = triggerBtn.querySelector('button');
        }
        
        if (!triggerBtn) return console.error("Hardware trigger missing! Check Gradio UI rendering.");

        window.isBmoRecording = !window.isBmoRecording;
        if (window.isBmoRecording) {
            window.playBmoBeep(880);
            window.setBMOState('listening');
        } else {
            window.playBmoBeep(440);
            window.setBMOProgress(20, 'System: Reading Hardware Buffer...');
            window.setBMOState('thinking');
        }
        // Send signal to Python Backend!
        triggerBtn.click(); 
    };

    window.setBMOProgress = function(percent, text) {
        const fill = document.getElementById('bmo-progress-fill');
        const label = document.getElementById('bmo-progress-text');
        if (fill) fill.style.width = percent + '%';
        if (label) label.innerText = percent + '% - ' + text;
    };

    window.setBMOState = function(state) {
        const screen = document.getElementById('bmo-screen');
        const mouth = document.getElementById('bmo-mouth');
        if (!screen || !mouth) return;
        screen.className = 'bmo-screen'; mouth.classList.remove('speaking');
        if (state === 'listening') { screen.classList.add('listening'); } 
        else if (state === 'thinking') { screen.classList.add('thinking'); } 
        else if (state === 'speaking') { mouth.classList.add('speaking'); }
    };
</script>
"""

bmo_html = f"""
<style>{css_styles}</style>
<div id="bmo-container">
    <div id="bmo-screen" class="bmo-screen">
        <div class="bmo-eye left"></div><div class="bmo-eye right"></div><div id="bmo-mouth" class="bmo-mouth"></div>
        <div class="bmo-waveform"><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div></div>
        <div class="bmo-thinking-box">
            <div class="bmo-thinking-grid"><div class="grid-dot"></div><div class="grid-dot"></div><div class="grid-dot"></div><div class="grid-dot"></div><div class="grid-dot"></div><div class="grid-dot"></div><div class="grid-dot"></div><div class="grid-dot"></div><div class="grid-dot"></div></div>
            <div class="bmo-progress-container"><div id="bmo-progress-fill" class="bmo-progress-fill"></div></div>
            <div id="bmo-progress-text" class="bmo-progress-text">0% - Initializing</div>
        </div>
    </div>
    <div class="bmo-slot"></div><div class="bmo-sbc"></div>
    <svg class="bmo-dpad-svg" viewBox="0 0 100 100"><path d="M 35 5 L 65 5 L 65 35 L 95 35 L 95 65 L 65 65 L 65 95 L 35 95 L 35 65 L 5 65 L 5 35 L 35 35 Z" fill="#ffcc00" stroke="#000" stroke-width="4" stroke-linejoin="round"/></svg>
    <svg class="bmo-triangle-svg" viewBox="0 0 100 100"><polygon points="50,10 90,90 10,90" fill="#00ccff" stroke="#000" stroke-width="6" stroke-linejoin="round"/></svg>
    <div class="bmo-gc"></div>
    <!-- Red Button triggers JS -->
    <div class="bmo-rc" onclick="window.toggleBmoRecord()"></div>
    <div class="bmo-pill p1"></div><div class="bmo-pill p2"></div>
</div>
"""

def bmo_hardware_controller(history_data):
    """Single unified controller triggered by the red button"""
    global IS_RECORDING, AUDIO_BUFFER
    
    if history_data is None:
        history_data = {"turns": [], "summary": "", "messages": []}
    messages = history_data.get("messages", [])

    try:
        if not IS_RECORDING:
            # --- BUTTON PRESS 1: START RECORDING ---
            AUDIO_BUFFER.clear()
            IS_RECORDING = True
            print("[*] Hardware Recording Activated...")
            yield None, "listening:20:System: Recording Audio...", messages, history_data
            return
        else:
            # --- BUTTON PRESS 2: STOP & PROCESS ---
            IS_RECORDING = False
            print("[*] Hardware Recording Stopped. Processing Array...")
            yield None, "thinking:30:Ears: Transcribing...", messages, history_data
            
            if not AUDIO_BUFFER:
                yield None, "idle:0:Error - No Audio", messages, history_data
                return
                
            # 1. Extract Hardware Array
            audio_data = np.concatenate(AUDIO_BUFFER, axis=0).flatten()
            AUDIO_BUFFER.clear()
            
            # 2. Whisper ASR
            result = whisper_model.transcribe(audio_data, language="fr", fp16=False)
            user_text = result.get("text", "").strip() or "Bonjour BMO!"
            print(f"  -> User: {user_text}")
            history_data["turns"].append({"role": "user", "content": user_text})
            messages.append({"role": "user", "content": user_text})
            
            yield None, "thinking:65:Brain: Generating Output...", messages, history_data
            
            # 3. Qwen LLM
            llm_msgs = [{"role": "system", "content": BMO_SYSTEM_PROMPT}] + history_data["turns"][-4:]
            if has_llm:
                bmo_text = llm.create_chat_completion(messages=llm_msgs, max_tokens=60)["choices"][0]["message"]["content"]
            else:
                bmo_text = f"J'ai entendu : '{user_text}'."
            bmo_text = bmo_text.replace("BMO", "Beemo")
            print(f"  -> BMO: {bmo_text}")
            
            history_data["turns"].append({"role": "assistant", "content": bmo_text})
            messages.append({"role": "assistant", "content": bmo_text})
            
            yield None, "thinking:85:Voice: Synthesizing...", messages, history_data

            # 4. Kokoro TTS & Native Hardware Playback
            if kokoro:
                try:
                    audio, sr_out = kokoro.create(bmo_text, voice="ff_siwis", speed=1.15, lang="fr-fr")
                except Exception as e:
                    print(f"[!] Primary voice failed, falling back: {e}")
                    audio, sr_out = kokoro.create(bmo_text, voice="af_bella", speed=1.15)

                audio_flat = audio.squeeze()

                # Cartoon DSP Shift
                pitch_factor = 1.2599
                new_len = int(len(audio_flat) / pitch_factor)
                samples_shifted = scipy.signal.resample(audio_flat, new_len).astype(np.float32)

                # Yield speaking state FIRST so the UI mouth animates instantly
                yield None, "speaking:100:BMO Speaking!", messages, history_data

                # Play directly through OS hardware speakers (bypasses browser autoplay blocks)
                import time
                sd.play(samples_shifted, sr_out)

                # Calculate audio length and keep mouth moving while audio is physically playing
                duration = len(samples_shifted) / sr_out
                time.sleep(duration)

                # Audio physically finished playing, return to idle
                yield None, "idle:0:Ready", messages, history_data
            else:
                from gtts import gTTS
                tts = gTTS(text=bmo_text, lang='fr', slow=False)
                tts.save("temp_bmo_gtts.mp3")
                import soundfile as sf
                samples_raw, sr_out = sf.read("temp_bmo_gtts.mp3")
                if samples_raw.ndim > 1:
                    samples_raw = samples_raw.mean(axis=1)
                pitch_factor = 1.2599
                new_len = int(len(samples_raw) / pitch_factor)
                samples_shifted = scipy.signal.resample(samples_raw, new_len).astype(np.float32)

                yield None, "speaking:100:BMO Speaking!", messages, history_data
                import time
                sd.play(samples_shifted, sr_out)
                duration = len(samples_shifted) / sr_out
                time.sleep(duration)
                yield None, "idle:0:Ready", messages, history_data

    except Exception as e:
        print(f"\n[CRITICAL ERROR]\n{traceback.format_exc()}")
        yield None, "idle:0:Error - Check Console", messages, history_data


with gr.Blocks(head=head_js) as demo:
    gr.Markdown("<h1 style='text-align: center; color: #3ca993;'>🤖 BMO Live Edge Tutor</h1>")
    chat_memory = gr.State({"turns": [], "summary": "", "messages": []})

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(bmo_html)
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(type="messages", elem_classes="cute-chat", height=520)
            bmo_state = gr.Textbox(visible=False)

            with gr.Group(elem_classes="cloak-audio"):
                # HIDDEN HARDWARE TRIGGER
                trigger_btn = gr.Button("Trigger", elem_id="bmo-trigger")
                bmo_voice = gr.Audio(autoplay=True)

    # Clicking the red HTML button forces a click on trigger_btn, which fires the Python logic natively
    trigger_btn.click(
        fn=bmo_hardware_controller,
        inputs=[chat_memory],
        outputs=[bmo_voice, bmo_state, chatbot, chat_memory]
    )

    bmo_voice.stop(fn=lambda: "idle:0:Ready", outputs=[bmo_state])

    bmo_state.change(
        fn=None, inputs=[bmo_state],
        js="(stateStr) => { if(!stateStr) return; const p = stateStr.split(':'); window.setBMOState(p[0]); window.setBMOProgress(p[1], p[2]); }"
    )

if __name__ == "__main__":
    print("[*] Launching Hardware-Native BMO Dashboard on http://127.0.0.1:7925 ...")
    demo.launch(server_name="127.0.0.1", server_port=7925)