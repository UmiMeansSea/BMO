"""
BMO Live Edge Tutor - High Performance Instant Audio Pipeline & Diagnostic Logging
=====================================================================================
Fixes & Upgrades:
1. Added missing scipy.signal import to fix TTS crash.
2. Removed duplicate event triggers to prevent race conditions.
3. Added strict try/except blocks to yield errors directly to the UI console.
"""

import sys
import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal # FIXED: Added missing import
from io import BytesIO
import traceback

try:
    import gradio as gr
    import whisper
    from llama_cpp import Llama
    from gtts import gTTS
    import soundfile as sf
except ImportError as e:
    print(f"[!] Missing dependency: {e}")
    sys.exit(1)

MODELS_DIR = Path(r"D:\BMO-Research\models")
LLM_PATH = MODELS_DIR / "bmo-model-4bit.gguf"

print("[*] Initializing Fast Push-to-Talk BMO Dashboard System...")
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
/* BMO Chassis & Body */
#bmo-container { background-color: #3ca993; width: 400px; height: 520px; border: 4px solid #000; border-radius: 20px; position: relative; margin: 0 auto; }
.bmo-screen { background-color: #b7efcc; width: 350px; height: 220px; border: 4px solid #000; border-radius: 15px; position: absolute; top: 20px; left: 21px; box-sizing: border-box; transition: background 0.2s ease; overflow: hidden; }

/* --- STATE 1: IDLE --- */
.bmo-eye { background: #000; width: 16px; height: 16px; border-radius: 50%; position: absolute; top: 35%; animation: blink 4s infinite; }
.bmo-eye.left { left: 25%; } .bmo-eye.right { right: 25%; }
@keyframes blink { 0%, 96%, 98%, 100% { transform: scaleY(1); } 97% { transform: scaleY(0.1); } }
.bmo-mouth { position: absolute; top: 50%; left: 50%; transform: translateX(-50%); width: 65px; height: 32px; border: 4px solid #000; border-top: transparent; border-left: transparent; border-right: transparent; border-radius: 0 0 50px 50px; transition: all 0.15s ease; }

/* --- STATE 2: LISTENING --- */
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

/* --- STATE 3: THINKING --- */
.bmo-thinking-box { display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); flex-direction: column; align-items: center; justify-content: center; width: 85%; }
.bmo-thinking-grid { display: grid; width: 70px; height: 70px; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 10px; }
.bmo-screen.thinking { background-color: #1a4237; }
.bmo-screen.thinking .bmo-eye, .bmo-screen.thinking .bmo-mouth, .bmo-screen.thinking .bmo-waveform { display: none; }
.bmo-screen.thinking .bmo-thinking-box { display: flex; }
.grid-dot { background: #ffcc00; border-radius: 4px; animation: dot-pulse 0.8s infinite alternate; }
.grid-dot:nth-child(1) { animation-delay: 0.0s; } .grid-dot:nth-child(2) { animation-delay: 0.2s; }
.grid-dot:nth-child(3) { animation-delay: 0.4s; } .grid-dot:nth-child(4) { animation-delay: 0.6s; }
.grid-dot:nth-child(5) { animation-delay: 0.8s; } .grid-dot:nth-child(6) { animation-delay: 0.4s; }
.grid-dot:nth-child(7) { animation-delay: 0.2s; } .grid-dot:nth-child(8) { animation-delay: 0.6s; }
.grid-dot:nth-child(9) { animation-delay: 0.0s; }
@keyframes dot-pulse { 0% { opacity: 0.2; transform: scale(0.8); } 100% { opacity: 1; transform: scale(1.1); background: #ff5500; } }
.bmo-progress-container { width: 100%; background: #0c211b; border: 2px solid #000; border-radius: 10px; height: 16px; position: relative; overflow: hidden; }
.bmo-progress-fill { background: linear-gradient(90deg, #33ff99, #66ffff); width: 0%; height: 100%; transition: width 0.2s ease-in-out; }
.bmo-progress-text { font-family: monospace; font-size: 13px; font-weight: bold; color: #ffcc00; margin-top: 5px; text-shadow: 1px 1px 0 #000; }

/* --- STATE 4: SPEAKING --- */
.bmo-mouth.speaking { animation: talk-anim 0.22s infinite alternate ease-in-out; background: #112a20; border: 3px solid #000; }
@keyframes talk-anim { 0% { height: 12px; width: 35px; border-radius: 20px; top: 58%; } 50% { height: 28px; width: 48px; border-radius: 50% 50% 45% 45%; top: 52%; } 100% { height: 38px; width: 52px; border-radius: 40% 40% 50% 50%; top: 50%; } }

/* Hardware */
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

    window.bmoLog = function(msg) {
        console.log('[BMO Client Log]:', msg);
    };

    window.playBmoBeep = function(freq) {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, ctx.currentTime);
            gain.gain.setValueAtTime(0.1, ctx.currentTime);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.15);
        } catch(e) {
            console.error('[BMO Audio Beep Error]:', e);
        }
    };

    window.toggleBmoRecord = function() {
        const micContainer = document.getElementById('bmo-mic');
        if (!micContainer) return;

        if (!window.isBmoRecording) {
            // Fuzzy search for any button containing "Record" or "record" in its aria-label
            const recordBtn = micContainer.querySelector('button[aria-label*="Record"]') || 
                              micContainer.querySelector('button[aria-label*="record"]') || 
                              micContainer.querySelector('button');
            if (recordBtn) {
                window.bmoLog('Clicking RECORD button...');
                recordBtn.click();
                window.isBmoRecording = true;
                window.playBmoBeep(880);
                window.setBMOState('listening');
            }
        } else {
            // Fuzzy search for any button containing "Stop" or "stop"
            const stopBtn = micContainer.querySelector('button[aria-label*="Stop"]') || 
                            micContainer.querySelector('button[aria-label*="stop"]') || 
                            micContainer.querySelector('button');
            if (stopBtn) {
                window.bmoLog('Clicking STOP button...');
                stopBtn.click();
                window.isBmoRecording = false;
                window.playBmoBeep(440);
                window.setBMOProgress(20, 'System: Uploading Audio...');
                window.setBMOState('thinking');
            }
        }
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

        screen.className = 'bmo-screen';
        mouth.classList.remove('speaking');

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
        <div class="bmo-eye left"></div>
        <div class="bmo-eye right"></div>
        <div id="bmo-mouth" class="bmo-mouth"></div>
        <div class="bmo-waveform">
            <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
            <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
        </div>
        <div class="bmo-thinking-box">
            <div class="bmo-thinking-grid">
                <div class="grid-dot"></div><div class="grid-dot"></div><div class="grid-dot"></div>
                <div class="grid-dot"></div><div class="grid-dot"></div><div class="grid-dot"></div>
                <div class="grid-dot"></div><div class="grid-dot"></div><div class="grid-dot"></div>
            </div>
            <div class="bmo-progress-container"><div id="bmo-progress-fill" class="bmo-progress-fill"></div></div>
            <div id="bmo-progress-text" class="bmo-progress-text">0% - Initializing</div>
        </div>
    </div>
    <div class="bmo-slot"></div><div class="bmo-sbc"></div>
    <svg class="bmo-dpad-svg" viewBox="0 0 100 100"><path d="M 35 5 L 65 5 L 65 35 L 95 35 L 95 65 L 65 65 L 65 95 L 35 95 L 35 65 L 5 65 L 5 35 L 35 35 Z" fill="#ffcc00" stroke="#000" stroke-width="4" stroke-linejoin="round"/></svg>
    <svg class="bmo-triangle-svg" viewBox="0 0 100 100"><polygon points="50,10 90,90 10,90" fill="#00ccff" stroke="#000" stroke-width="6" stroke-linejoin="round"/></svg>
    <div class="bmo-gc"></div>
    <div class="bmo-rc" onclick="window.toggleBmoRecord()"></div>
    <div class="bmo-pill p1"></div><div class="bmo-pill p2"></div>
</div>
"""

def summarize_turns(old_turns: list, current_summary: str) -> str:
    lines = []
    for turn in old_turns:
        role = "User" if turn.get("role") == "user" else "BMO"
        lines.append(f"{role}: {turn.get('content', '')}")
    turn_text = " | ".join(lines)
    new_summary = f"{current_summary}; {turn_text}" if current_summary else f"Summary of past topics: {turn_text}"
    return new_summary[-250:] if len(new_summary) > 250 else new_summary

def bmo_response_generator(audio_data, history_data):
    """Wrapped generator to catch errors and yield them dynamically."""
    try:
        # Step 1: Init pipeline
        print("\n[*] [PIPELINE START] Processing incoming audio stream...")
        yield None, "thinking:30:Ears: Reading Audio...", history_data.get("messages", []), history_data
        
        if history_data is None:
            history_data = {"turns": [], "summary": "", "messages": []}

        turns = history_data.get("turns", [])
        summary = history_data.get("summary", "")
        messages = history_data.get("messages", [])

        if audio_data is None:
            print("[!] audio_data is None. Bailing out.")
            yield None, "idle:0:Awaiting Input", messages, history_data
            return

        # Extract Tuple Data
        sr, arr = audio_data
        print(f"[*] Audio loaded: Sample Rate = {sr}, Shape = {arr.shape}")
        
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if np.max(np.abs(arr)) > 1.0:
            arr = arr / 32768.0

        if sr != 16000:
            step = sr / 16000.0
            indices = np.arange(0, len(arr), step).astype(int)
            indices = indices[indices < len(arr)]
            arr_16k = arr[indices]
        else:
            arr_16k = arr

        # Step 2: Whisper Transcribe
        print("[*] [STEP 1/3] Transcribing with Whisper...")
        yield None, "thinking:45:Ears: Transcribing...", messages, history_data
        result = whisper_model.transcribe(arr_16k, language="fr", fp16=False)
        user_text = result.get("text", "").strip()

        if not user_text:
            user_text = "Bonjour BMO!"
        print(f"  -> Transcribed: \"{user_text}\"")

        turns.append({"role": "user", "content": user_text})
        messages.append({"role": "user", "content": user_text})

        # Step 3: LLM Generation
        print("[*] [STEP 2/3] Generating LLM response...")
        yield None, "thinking:65:Brain: Generating Output...", messages, history_data

        SLIDING_WINDOW_SIZE = 4
        if len(turns) > SLIDING_WINDOW_SIZE:
            old_turns = turns[:-SLIDING_WINDOW_SIZE]
            turns = turns[-SLIDING_WINDOW_SIZE:]
            summary = summarize_turns(old_turns, summary)

        system_prompt = BMO_SYSTEM_PROMPT
        if summary:
            system_prompt += f"\nBACKGROUND MEMORY: {summary}"

        llm_messages = [{"role": "system", "content": system_prompt}] + turns

        if has_llm:
            response = llm.create_chat_completion(messages=llm_messages, max_tokens=60)
            bmo_text = response["choices"][0]["message"]["content"]
        else:
            bmo_text = f"Salut ! J'ai bien entendu : '{user_text}'."
        
        bmo_text = bmo_text.replace("BMO", "Beemo")
        print(f"  -> Generated: \"{bmo_text}\"")

        turns.append({"role": "assistant", "content": bmo_text})
        messages.append({"role": "assistant", "content": bmo_text})
        history_data = {"turns": turns, "summary": summary, "messages": messages}

        # Step 4: TTS & Pitch Shift
        print("[*] [STEP 3/3] Synthesizing audio...")
        yield None, "thinking:85:Voice: Pitch Shifting...", messages, history_data

        out_wav = "bmo_live_response.wav"
        tts = gTTS(text=bmo_text, lang='fr', slow=False)
        tts.save("temp_bmo_gtts.mp3")

        samples_raw, sr_out = sf.read("temp_bmo_gtts.mp3")
        if samples_raw.ndim > 1:
            samples_raw = samples_raw.mean(axis=1)

        # Apply SciPy Pitch Shift (previously caused the silent crash)
        pitch_factor = 1.2599
        new_len = int(len(samples_raw) / pitch_factor)
        samples_shifted = scipy.signal.resample(samples_raw, new_len).astype(np.float32)
        samples_int16 = (samples_shifted * 32767).astype(np.int16)
        wav.write(out_wav, sr_out, samples_int16)

        print("[*] [PIPELINE COMPLETE] Yielding audio to frontend.")
        yield out_wav, "speaking:100:BMO Speaking!", messages, history_data

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"\n[CRITICAL PIPELINE ERROR]\n{error_trace}")
        # Yield the error directly to the BMO screen so you don't have to guess why it stuck
        yield None, f"idle:0:Error - Check Console", history_data.get("messages", []), history_data


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
                audio_input = gr.Audio(sources=["microphone"], type="numpy", elem_id="bmo-mic")
                bmo_voice = gr.Audio(autoplay=True)

    # Use the native stop_recording event so it only fires when audio is completely finalized
    audio_input.stop_recording(
        fn=bmo_response_generator,
        inputs=[audio_input, chat_memory],
        outputs=[bmo_voice, bmo_state, chatbot, chat_memory]
    )

    # Reset both state AND clear audio_input on voice stop so next prompt is ready
    bmo_voice.stop(
        fn=lambda: (None, "idle:0:Ready"),
        inputs=[],
        outputs=[audio_input, bmo_state]
    )

    bmo_state.change(
        fn=None,
        inputs=[bmo_state],
        js="""(stateStr) => {
            if (!stateStr) return;
            window.bmoLog('State transitioned to: ' + stateStr);
            const parts = stateStr.split(':');
            window.setBMOState(parts[0] || 'idle');
            window.setBMOProgress(parts[1] || '0', parts[2] || '');
        }"""
    )

if __name__ == "__main__":
    print("[*] Launching Diagnostic BMO Dashboard on http://127.0.0.1:7925 ...")
    demo.launch(server_name="127.0.0.1", server_port=7925)