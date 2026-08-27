import os
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    os.system("pip install fpdf2")
    from fpdf import FPDF

class BmoResearchPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'BMO: Benchmarking Model-quantized Offline Agents', 0, 1, 'L')
        self.line(10, 18, 200, 18)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Draft Manuscript - Page {self.page_no()}', 0, 0, 'C')

docs_dir = Path("docs")
docs_dir.mkdir(exist_ok=True)
pdf_path = docs_dir / "BMO_Research_Draft_Manuscript.pdf"

pdf = BmoResearchPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title Section
pdf.set_font('helvetica', 'B', 15)
pdf.set_text_color(20, 40, 30)
pdf.multi_cell(0, 7, 'BMO: Benchmarking Model-quantized Offline Agents for Sustainable Language Tutoring', 0, 'C')
pdf.ln(2)

pdf.set_font('helvetica', 'I', 10)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 5, 'A Sovereign Edge AI Framework for Resource-Constrained Pedagogical Environments', 0, 'C')
pdf.ln(6)

# Abstract
pdf.set_font('helvetica', 'B', 11)
pdf.cell(0, 6, 'Abstract', 0, 1, 'L')
pdf.set_font('helvetica', '', 9.5)
abstract_text = (
    "Current AI language learning tools rely heavily on cloud APIs, imposing a hidden carbon footprint. "
    "This paper introduces BMO, a fully autonomous, voice-to-voice French language learning companion "
    "running entirely offline on entry-level edge hardware. By coupling Whisper ASR, 4-bit GGUF SLMs, "
    "and Kokoro-82M TTS, we evaluate the trade-offs between quantization and pedagogical accuracy. "
    "Our empirical telemetry demonstrates a 66.0% pass rate while consuming only 27.49 Wh of energy."
)
pdf.multi_cell(0, 5, abstract_text)
pdf.ln(5)

# Introduction & Core Findings
pdf.set_font('helvetica', 'B', 11)
pdf.cell(0, 6, '1. Introduction & Core Benchmark Findings', 0, 1, 'L')
pdf.set_font('helvetica', '', 9.5)
intro_text = (
    "BMO evaluates small language models (SLMs) running locally on CPU-only hardware. "
    "Through a 200-question standardized exam of A2/B1 French conversational prompts, "
    "we measured pedagogical error correction, latency, and energy telemetry. "
    "Key findings include:\n"
    "- Pedagogical Pass Rate: 66.0% (132/200 turns passed)\n"
    "- Total Energy Consumption: 27.49 Wh (0.137 Wh/sentence)\n"
    "- Total Carbon Footprint: 19.61 g CO2e (0.098 g CO2e/sentence)\n"
    "- Average Response Latency: 13.79 seconds per turn on 4-thread CPU"
)
pdf.multi_cell(0, 5, intro_text)
pdf.ln(5)

# Architecture & State Engines
pdf.set_font('helvetica', 'B', 11)
pdf.cell(0, 6, '2. Edge Architecture & State Engines', 0, 1, 'L')
pdf.set_font('helvetica', '', 9.5)
arch_text = (
    "BMO couples three specialized state engines:\n"
    "1. RoleplayEngine: Handles 3-4 turn real-life scenario simulations (Parisian Cafe, Bakery, Train Station, Hotel) with automated exit debriefs.\n"
    "2. ScaffoldingEngine: Implements a 3-tiered hint hierarchy and hesitation recovery playing at slowed 0.70x speech pacing.\n"
    "3. SessionMemoryEngine: Maintains longitudinal student memory in session_review.json, generating startup warm-up quizzes and extracting vocabulary on session exit."
)
pdf.multi_cell(0, 5, arch_text)

# Save PDF
pdf.output(str(pdf_path))
print(f"PDF successfully generated at: {pdf_path.resolve()}")
