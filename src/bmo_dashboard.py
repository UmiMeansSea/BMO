"""
BMO Gradio Dashboard & Audio Stream UI
======================================
Interactive animated BMO chassis UI with Gradio Blocks and real-time audio interaction.
"""

import sys
from pathlib import Path
import numpy as np

try:
    import gradio as gr
except ImportError as e:
    print(f"[!] Missing dependency: {e}")
    print("  Run: pip install gradio")
    sys.exit(1)

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

def bmo_chat(audio_path):
    return audio_path

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🤖 BMO Live Edge Tutor</h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(bmo_html)
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Speak to BMO")
            audio_out = gr.Audio(label="BMO Response", autoplay=True)
            btn = gr.Button("Talk to BMO", variant="primary")
            btn.click(fn=bmo_chat, inputs=[audio_in], outputs=[audio_out])

if __name__ == "__main__":
    print("[*] Launching BMO Gradio Dashboard...")
    demo.launch(server_name="127.0.0.1", server_port=7861)
