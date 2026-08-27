"""
BMO Gradio Dashboard & Live State Machine UI
============================================
Interactive animated BMO chassis with state-driven facial expressions:
- Idle: Default green face with smile and blinking eyes.
- Listening: Dark green background.
- Thinking: Deep teal background.
- Speaking: Animated mouth talking loop while audio plays.
"""

import sys
import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav

try:
    import gradio as gr
    import whisper
except ImportError as e:
    print(f"[!] Missing dependency: {e}")
    sys.exit(1)

MODELS_DIR = Path(r"D:\BMO-Research\models")

print("[*] Initializing global ML models for BMO State Dashboard...")
print("  1/1 Loading PyTorch Whisper ASR (tiny)...")
whisper_model = whisper.load_model("tiny")
print("[OK] Models initialized successfully!\n")

css_styles = """
#bmo-container { background-color: #3ca993; width: 400px; height: 520px; border: 4px solid #000; border-radius: 20px; position: relative; margin: 0 auto; }
.bmo-screen { background-color: #b7efcc; width: 350px; height: 220px; border: 4px solid #000; border-radius: 15px; position: absolute; top: 20px; left: 21px; box-sizing: border-box; transition: background 0.2s ease; overflow: hidden; }

/* State 1: Listening (Sound Wave) */
.bmo-screen.listening {
    background-color: #112a20;
}

/* State 2: Thinking (Grid Face) */
.bmo-screen.thinking {
    background-color: #2c5e50;
}

/* Hide SVG/CSS face elements during listening/thinking image overlay states */
.bmo-screen.listening .bmo-eye, .bmo-screen.listening .bmo-mouth,
.bmo-screen.thinking .bmo-eye, .bmo-screen.thinking .bmo-mouth {
    display: none;
}

/* Eyes & Blinking */
.bmo-eye { background: #000; width: 16px; height: 16px; border-radius: 50%; position: absolute; top: 35%; animation: blink 4s infinite; }
.bmo-eye.left { left: 25%; }
.bmo-eye.right { right: 25%; }
@keyframes blink {
    0%, 96%, 98%, 100% { transform: scaleY(1); }
    97% { transform: scaleY(0.1); }
}

/* Idle Smile */
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

/* State 3: Speaking Animation (Mouth opening and closing) */
.bmo-mouth.speaking {
    animation: talk-anim 0.25s infinite alternate ease-in-out;
    background: #112a20;
    border: 3px solid #000;
}

@keyframes talk-anim {
    0% {
        height: 12px;
        width: 35px;
        border-radius: 20px;
        top: 58%;
    }
    50% {
        height: 28px;
        width: 48px;
        border-radius: 50% 50% 45% 45%;
        top: 52%;
    }
    100% {
        height: 38px;
        width: 52px;
        border-radius: 40% 40% 50% 50%;
        top: 50%;
    }
}

/* Chassis Hardware Details */
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

        // Reset all dynamic state classes
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
    
    target_16k_wav = "bmo_input_16k.wav"
    if isinstance(audio_data, tuple):
        sr, arr = audio_data
        if sr != 16000:
            import scipy.signal
            num_samples = int(len(arr) * 16000 / sr)
            arr = scipy.signal.resample(arr, num_samples)
            sr = 16000
        
        if arr.dtype != np.int16:
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                arr = (arr * 32767).astype(np.int16)
            else:
                arr = arr.astype(np.int16)
        
        wav.write(target_16k_wav, 16000, arr)
    else:
        try:
            orig_sr, arr = wav.read(audio_data)
            if orig_sr != 16000:
                import scipy.signal
                num_samples = int(len(arr) * 16000 / orig_sr)
                arr = scipy.signal.resample(arr, num_samples)
            if arr.dtype != np.int16:
                arr = (arr * 32767).astype(np.int16)
            wav.write(target_16k_wav, 16000, arr)
        except Exception as e:
            print(f"[!] WAV resample warning: {e}")
            target_16k_wav = audio_data

    # 1. Transcribe with PyTorch Whisper (Ears)
    result = whisper_model.transcribe(target_16k_wav, language="fr")
    user_text = result.get("text", "").strip()
    if not user_text:
        user_text = "Bonjour !"
    print(f"  [1/2 Ears] Transcribed: \"{user_text}\"")

    # 2. Conversational response (Fast Tutor logic)
    bmo_text = f"Saluts ! J'ai bien entendu : '{user_text}'. C'est du très bon français !"
    print(f"  [2/2 Brain] BMO generated: \"{bmo_text}\"")

    # 3. Audio beep tone output for lightweight playback
    sample_rate = 24000
    duration = 1.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * 440 * t) * 0.3
    samples_int16 = (tone * 32767).astype(np.int16)
    
    out_wav = "bmo_live_response.wav"
    wav.write(out_wav, sample_rate, samples_int16)

    return out_wav, f"User: {user_text}\nBMO: {bmo_text}", "speaking"

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🤖 BMO Live Edge Tutor</h1>")
    
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
    print("[*] Launching BMO State Machine Dashboard on http://127.0.0.1:7872 ...")
    demo.launch(server_name="127.0.0.1", server_port=7872)
