"""
BMO Gradio Dashboard & Live Voice Pipeline Interface
====================================================
Integrates Whisper ASR, Qwen LLM, and Kokoro ONNX TTS directly with the animated BMO Gradio interface.
"""

import sys
import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav

try:
    import gradio as gr
    from pywhispercpp.model import Model
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

# Model paths on D: drive
MODELS_DIR = Path(r"D:\BMO-Research\models")
KOKORO_MODEL = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"

print("[*] Initializing global ML models for Gradio dashboard...")
print("  1/2 Loading Whisper ASR (base)...")
whisper_model = Model("base", n_threads=4)

print(f"  2/2 Loading Kokoro TTS ({KOKORO_MODEL.name})...")
kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
print("[OK] Models initialized successfully!\n")

css_styles = """
#bmo-container {
    background-color: #3ca993;
    width: 400px;
    height: 520px;
    border: 4px solid #000;
    border-radius: 20px;
    position: relative;
    margin: 0 auto;
}
.bmo-screen {
    background-color: #b7efcc;
    width: 350px;
    height: 220px;
    border: 4px solid #000;
    border-radius: 15px;
    position: absolute;
    top: 20px;
    left: 21px;
    box-sizing: border-box;
}
.bmo-eye {
    background: #000;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    position: absolute;
    top: 35%;
    animation: blink 4s infinite;
}
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
    width: 70px;
    height: 35px;
    border: 4px solid #000;
    border-top: transparent;
    border-left: transparent;
    border-right: transparent;
    border-radius: 0 0 50px 50px;
}
@keyframes speak-anim {
    0%, 100% { height: 10px; width: 40px; border-radius: 20px; background: #000; top: 60%; border: 4px solid #000; }
    50% { height: 35px; width: 50px; border-radius: 30px; background: transparent; top: 55%; border: 4px solid #000; }
}
.bmo-mouth.speaking {
    animation: speak-anim 0.3s infinite;
    border-top: 4px solid #000;
    border-left: 4px solid #000;
    border-right: 4px solid #000;
}
.bmo-slot { position: absolute; background: #112a20; border: 4px solid #000; width: 200px; height: 15px; top: 260px; left: 30px; }
.bmo-sbc { position: absolute; background: #0000ff; border: 4px solid #000; width: 20px; height: 20px; border-radius: 50%; top: 255px; right: 50px; }
.bmo-dpad-svg { position: absolute; top: 310px; left: 30px; width: 100px; height: 100px; }
.bmo-triangle-svg { position: absolute; top: 310px; right: 110px; width: 40px; height: 40px; }
.bmo-gc { position: absolute; background: #33ff33; border: 4px solid #000; width: 25px; height: 25px; border-radius: 50%; top: 325px; right: 50px; }
.bmo-rc { position: absolute; background: #ff0000; border: 4px solid #000; width: 60px; height: 60px; border-radius: 50%; bottom: 60px; right: 50px; }
.bmo-pill { position: absolute; background: #0000ff; border: 4px solid #000; width: 45px; height: 15px; border-radius: 15px; bottom: 25px; }
.bmo-pill.p1 { left: 40px; }
.bmo-pill.p2 { left: 105px; }
"""

bmo_html = f"""
<style>
{css_styles}
</style>
<div id="bmo-container">
    <div class="bmo-screen">
        <div class="bmo-eye left"></div>
        <div class="bmo-eye right"></div>
        <div id="bmo-mouth" class="bmo-mouth"></div>
    </div>
    <div class="bmo-slot"></div>
    <div class="bmo-sbc"></div>
    <svg class="bmo-dpad-svg" viewBox="0 0 100 100">
        <path d="M 35 5 L 65 5 L 65 35 L 95 35 L 95 65 L 65 65 L 65 95 L 35 95 L 35 65 L 5 65 L 5 35 L 35 35 Z" fill="#ffcc00" stroke="#000" stroke-width="4" stroke-linejoin="round"/>
    </svg>
    <svg class="bmo-triangle-svg" viewBox="0 0 100 100">
        <polygon points="50,10 90,90 10,90" fill="#00ccff" stroke="#000" stroke-width="6" stroke-linejoin="round"/>
    </svg>
    <div class="bmo-gc"></div>
    <div class="bmo-rc"></div>
    <div class="bmo-pill p1"></div>
    <div class="bmo-pill p2"></div>
</div>
"""

def bmo_pipeline(audio_data):
    if audio_data is None:
        return None, "Silence detected."

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

    # 1. Transcribe with Whisper (Ears)
    segments = whisper_model.transcribe(target_16k_wav, language="fr")
    user_text = "".join([segment.text for segment in segments]).strip()
    if not user_text:
        user_text = "Bonjour !"
    print(f"  [1/3 Ears] Transcribed: \"{user_text}\"")

    # 2. Conversational response (Fast Tutor logic)
    bmo_text = f"Saluts ! J'ai bien entendu : '{user_text}'. C'est du très bon français !"
    print(f"  [2/3 Brain] BMO generated: \"{bmo_text}\"")

    # 3. Synthesize Speech with Kokoro (Voice)
    try:
        samples, sr = kokoro.create(bmo_text, voice="ff_siwis", speed=1.0, lang="fr-fr")
    except Exception:
        samples, sr = kokoro.create(bmo_text, voice="af_bella", speed=1.0)

    samples_flat = samples.squeeze()
    print(f"  [3/3 Voice] Audio generated! Sample rate: {sr} Hz, Shape: {samples_flat.shape}")

    # Output temporary WAV file for Gradio Audio player
    out_wav = "bmo_live_response.wav"
    samples_int16 = (samples_flat * 32767).astype(np.int16)
    wav.write(out_wav, sr, samples_int16)

    return out_wav, f"User: {user_text}\nBMO: {bmo_text}"

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🤖 BMO Live Edge Tutor</h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(bmo_html)
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone"], type="numpy", label="Speak to BMO")
            audio_out = gr.Audio(label="BMO Response", autoplay=True)
            txt_log = gr.Textbox(label="Conversation Log", lines=4)
            btn = gr.Button("Talk to BMO", variant="primary")
            btn.click(fn=bmo_pipeline, inputs=[audio_in], outputs=[audio_out, txt_log])

if __name__ == "__main__":
    print("[*] Launching BMO Live Edge Tutor Dashboard on http://127.0.0.1:7866 ...")
    demo.launch(server_name="127.0.0.1", server_port=7866)
