#!/usr/bin/env python3
"""Generate the Sprint 2 progress deck from the checked-in verification report."""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "sim" / "results" / "verification_report.json"
OUTPUT = Path(__file__).with_name("Sprint2_Progress.pptx")

NAVY = RGBColor(10, 20, 34)
PANEL = RGBColor(20, 35, 56)
PANEL_LIGHT = RGBColor(28, 48, 72)
WHITE = RGBColor(242, 247, 252)
MUTED = RGBColor(145, 164, 188)
BLUE = RGBColor(105, 161, 255)
GREEN = RGBColor(48, 211, 158)
AMBER = RGBColor(255, 190, 96)
RED = RGBColor(255, 132, 127)
GRID = RGBColor(45, 65, 89)
FONT = "Aptos"
MONO = "Aptos Mono"


def add_box(slide, x, y, w, h, color=PANEL, radius=True, line_color=None):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(
        shape_type, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background() if line_color is None else None
    if line_color is not None:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    if radius:
        shape.adjustments[0] = 0.12
    return shape


def add_text(
    slide,
    text,
    x,
    y,
    w,
    h,
    *,
    size=18,
    color=WHITE,
    bold=False,
    font=FONT,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.MIDDLE,
    margin=0,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_base(prs, section, title, subtitle, slide_number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    add_text(
        slide,
        "VITAL HARDWARE  /  SPRINT 2",
        0.55,
        0.26,
        5.5,
        0.28,
        size=10,
        color=BLUE,
        bold=True,
    )
    add_text(slide, section.upper(), 10.1, 0.26, 2.65, 0.28, size=9, color=MUTED, align=PP_ALIGN.RIGHT)
    add_text(slide, title, 0.55, 0.68, 12.2, 0.55, size=28, bold=True)
    add_text(slide, subtitle, 0.57, 1.28, 12.0, 0.34, size=13, color=MUTED)
    add_box(slide, 0.55, 7.12, 12.2, 0.012, color=GRID, radius=False)
    add_text(
        slide,
        "SOFTWARE EVIDENCE ≠ FPGA VALIDATION",
        0.58,
        7.2,
        5.8,
        0.22,
        size=8,
        color=MUTED,
        bold=True,
    )
    add_text(slide, f"{slide_number}  /  6", 11.6, 7.2, 1.1, 0.22, size=8, color=MUTED, align=PP_ALIGN.RIGHT)
    return slide


def add_bullet(slide, text, x, y, w, *, size=15, color=WHITE, bullet_color=GREEN):
    add_text(slide, "•", x, y, 0.22, 0.30, size=size, color=bullet_color, bold=True)
    add_text(slide, text, x + 0.28, y, w - 0.28, 0.48, size=size, color=color, valign=MSO_ANCHOR.TOP)


def add_stat(slide, x, y, w, h, label, value, detail, accent=GREEN):
    add_box(slide, x, y, w, h, PANEL, line_color=GRID)
    add_text(slide, label.upper(), x + 0.2, y + 0.15, w - 0.4, 0.26, size=9, color=MUTED, bold=True)
    add_text(slide, value, x + 0.2, y + 0.52, w - 0.4, 0.58, size=29, bold=True, color=WHITE)
    add_text(slide, detail, x + 0.2, y + h - 0.53, w - 0.4, 0.34, size=10, color=accent)


def make_deck(report):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "Sprint 2 Progress — ViTAl-Hardware"
    prs.core_properties.subject = "Five-minute update on streaming-attention implementation and validation"
    prs.core_properties.author = "Dominic Murphy"

    numeric = report["numeric"]
    lut = report["lut_and_exponential"]
    protocol = report["protocol"]

    # 1. Title and honest status.
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    add_text(slide, "VITAL HARDWARE  /  EC601", 0.7, 0.48, 5.8, 0.35, size=12, color=BLUE, bold=True)
    add_text(slide, "SPRINT 2", 0.72, 1.28, 4.8, 0.72, size=48, bold=True)
    add_text(slide, "From a tested model\ntoward a board demo", 0.72, 2.05, 7.5, 1.25, size=31, bold=True)
    add_text(slide, "Streaming Base-2 attention on the PYNQ-Z2", 0.75, 3.52, 7.8, 0.42, size=18, color=MUTED)
    add_box(slide, 9.2, 1.35, 3.35, 2.55, RGBColor(18, 53, 47), line_color=GREEN)
    add_text(slide, "SOFTWARE MODEL", 9.48, 1.78, 2.8, 0.3, size=11, color=GREEN, bold=True)
    add_text(slide, "CHECKS PASS", 9.48, 2.22, 2.8, 0.52, size=24, bold=True)
    add_text(slide, "FPGA / BOARD", 9.48, 2.96, 2.8, 0.3, size=11, color=AMBER, bold=True)
    add_text(slide, "NOT YET TESTED", 9.48, 3.28, 2.9, 0.36, size=17, bold=True)
    add_text(slide, "Progress snapshot · 8 November 2026", 0.75, 6.62, 7.0, 0.3, size=12, color=MUTED)
    add_text(slide, "5 MIN", 11.5, 6.62, 1.0, 0.3, size=12, color=BLUE, bold=True, align=PP_ALIGN.RIGHT)

    # 2. Scope and data path.
    slide = add_base(
        prs,
        "Demo scope",
        "One attention head, streamed end to end",
        "The Sprint 2 demo target is a fixed Q/K/V kernel—not full image classification.",
        2,
    )
    labels = [
        ("INPUT", "q: 32 values\n196 key/value rows", BLUE),
        ("SCORE", "dot(q, k)\nscaled to Q12", BLUE),
        ("ONLINE SOFTMAX", "running max + sum\n64-entry LUT · Q15", GREEN),
        ("OUTPUT", "16 weighted values\nsigned Q4", BLUE),
    ]
    xs = [0.62, 3.78, 6.94, 10.10]
    for (heading, body, accent), x in zip(labels, xs):
        add_box(slide, x, 2.15, 2.55, 1.42, PANEL_LIGHT, line_color=GRID)
        add_text(slide, heading, x + 0.18, 2.34, 2.18, 0.27, size=10, color=accent, bold=True)
        add_text(slide, body, x + 0.18, 2.70, 2.18, 0.62, size=15, bold=True)
    for x in (3.27, 6.43, 9.59):
        add_text(slide, "→", x, 2.62, 0.42, 0.42, size=24, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_box(slide, 0.62, 4.12, 12.03, 1.52, PANEL, line_color=GRID)
    add_text(slide, "STREAM SHAPE", 0.9, 4.36, 1.7, 0.26, size=10, color=MUTED, bold=True)
    add_text(slide, "32 query words  +  196 × (32 key + 16 value) words", 0.9, 4.72, 7.15, 0.4, size=20, bold=True)
    add_text(slide, "9,440 32-bit stream words per transaction", 8.28, 4.45, 3.95, 0.56, size=14, color=GREEN, bold=True, align=PP_ALIGN.RIGHT)
    add_text(slide, "The design keeps the query, current key/value, recurrence state, and output accumulators—no N × N score buffer.", 0.9, 5.88, 11.4, 0.48, size=14, color=MUTED)

    # 3. What is implemented.
    slide = add_base(
        prs,
        "Implementation",
        "The demo path is now represented in code",
        "A build recipe and host driver exist; they still need to be compiled, simulated, and exercised on the secured board.",
        3,
    )
    cards = [
        ("01", "RTL kernel", "AXI-stream input/output\nQ12 score · Q15 weights\nrunning max/sum recurrence", GREEN),
        ("02", "Vivado recipe", "Zynq PS + AXI DMA\nclock/reset + bitstream Tcl\nXC7Z020 target", BLUE),
        ("03", "PYNQ runner", "seeded int8 Q/K/V vectors\nDMA transfer + NumPy compare\npass/fail threshold", BLUE),
        ("04", "Verification harness", "fixed-point mirror + edge cases\n128 seeded random cases\nJSON / CSV / slide report", BLUE),
    ]
    for index, (number, title, description, accent) in enumerate(cards):
        x = 0.62 + index * 3.08
        add_box(slide, x, 2.2, 2.78, 2.63, PANEL, line_color=GRID)
        add_text(slide, number, x + 0.22, 2.42, 0.55, 0.35, size=12, color=accent, bold=True)
        add_text(slide, title, x + 0.22, 2.92, 2.32, 0.44, size=18, bold=True)
        add_text(slide, description, x + 0.22, 3.52, 2.30, 0.98, size=13, color=MUTED, valign=MSO_ANCHOR.TOP)
    add_box(slide, 0.62, 5.28, 12.03, 0.72, RGBColor(49, 39, 27), line_color=AMBER)
    add_text(slide, "BOARD SECURED — first physical bring-up and FPGA validation are still ahead.", 0.9, 5.43, 11.4, 0.38, size=16, color=AMBER, bold=True)

    # 4. Numeric results.
    slide = add_base(
        prs,
        "Numerical evidence",
        "The software arithmetic mirror stays inside its test threshold",
        "132 synthetic Q4 cases: 4 edge cases + 128 deterministic random seeds.",
        4,
    )
    add_stat(slide, 0.62, 2.0, 3.0, 1.52, "Worst vs Python", f"{numeric['max_abs_output_error_vs_python']:.5f}", "limit: 0.20 value units", GREEN)
    add_stat(slide, 3.82, 2.0, 3.0, 1.52, "Worst vs PyTorch", f"{numeric['max_abs_output_error_vs_torch']:.5f}", "same test vectors", BLUE)
    add_stat(slide, 7.02, 2.0, 2.65, 1.52, "LUT entries", "64 / 64", "match Python table", GREEN)
    add_stat(slide, 9.87, 2.0, 2.78, 1.52, "Exp deltas", "65,536", "match + monotonic", GREEN)
    bar_x, bar_y, bar_w = 0.95, 4.30, 10.8
    add_text(slide, "MAX ABS ERROR  /  ALLOWED BOARD-CHECK TOLERANCE", 0.95, 3.88, 11.0, 0.28, size=10, color=MUTED, bold=True)
    add_box(slide, bar_x, bar_y, bar_w, 0.34, PANEL_LIGHT, radius=False)
    error_ratio = numeric["max_abs_output_error_vs_python"] / numeric["board_tolerance_dequantized_units"]
    add_box(slide, bar_x, bar_y, bar_w * error_ratio, 0.34, GREEN, radius=False)
    add_text(slide, f"{numeric['max_abs_output_error_vs_python']:.5f}", bar_x, 4.72, 2.0, 0.3, size=13, color=GREEN, bold=True)
    add_text(slide, f"{numeric['board_tolerance_dequantized_units']:.2f} limit", bar_x + bar_w - 1.6, 4.72, 1.6, 0.3, size=13, color=MUTED, bold=True, align=PP_ALIGN.RIGHT)
    add_text(slide, "These are synthetic vectors and a tolerance chosen for this demo; they are not image-accuracy or FPGA measurements.", 0.95, 5.52, 11.2, 0.5, size=13, color=AMBER)

    # 5. Streaming/system evidence and its boundaries.
    slide = add_base(
        prs,
        "Streaming evidence",
        "The software checks support the streaming design claim",
        "Source inspection and a Python schedule model are useful preflight—not substitutes for RTL simulation.",
        5,
    )
    add_box(slide, 0.62, 2.05, 5.7, 3.58, PANEL, line_color=GRID)
    add_text(slide, "WHAT PASSED LOCALLY", 0.94, 2.34, 4.9, 0.3, size=11, color=GREEN, bold=True)
    add_bullet(slide, "20 ready/valid schedule trials with randomized gaps and back-pressure", 0.94, 2.92, 4.95)
    add_bullet(slide, "188,800 input words and 320 output words accounted for", 0.94, 3.68, 4.95)
    add_bullet(slide, "No N × N score array declared in the RTL source", 0.94, 4.44, 4.95)
    add_box(slide, 6.62, 2.05, 6.03, 3.58, PANEL, line_color=GRID)
    add_text(slide, "WHAT THIS DOES NOT PROVE", 6.94, 2.34, 5.25, 0.3, size=11, color=AMBER, bold=True)
    add_bullet(slide, "No HDL compiler or simulator ran here", 6.94, 2.92, 5.2, bullet_color=AMBER)
    add_bullet(slide, "No Vivado build, timing, resource, or power report", 6.94, 3.68, 5.2, bullet_color=AMBER)
    add_bullet(slide, "No bitstream or DMA transfer has run on the board yet", 6.94, 4.44, 5.2, bullet_color=AMBER)
    add_text(slide, "Report explicitly marks HDL simulation as NOT RUN.", 0.95, 5.98, 11.2, 0.34, size=15, color=WHITE, bold=True)

    # 6. Sprint close / next proof points.
    slide = add_base(
        prs,
        "Next proof points",
        "Sprint 2 target: evidence from the actual board",
        "The objective is to replace “code prepared” with a reproducible build, simulation log, and board result.",
        6,
    )
    steps = [
        ("1", "Bring up board", "PYNQ image · serial · SSH"),
        ("2", "Run HDL testbench", "xvlog → xelab → xsim"),
        ("3", "Build overlay", "Vivado bitstream + HWH"),
        ("4", "Run DMA demo", "board output vs Python"),
    ]
    for i, (n, title, detail) in enumerate(steps):
        x = 0.62 + i * 3.08
        add_box(slide, x, 2.1, 2.78, 1.45, PANEL_LIGHT, line_color=GRID)
        add_text(slide, n, x + 0.18, 2.3, 0.36, 0.44, size=20, color=GREEN, bold=True)
        add_text(slide, title, x + 0.62, 2.28, 1.95, 0.38, size=15, bold=True)
        add_text(slide, detail, x + 0.2, 2.91, 2.38, 0.32, size=11, color=MUTED)
    add_box(slide, 0.62, 4.05, 12.03, 1.2, RGBColor(18, 53, 47), line_color=GREEN)
    add_text(slide, "SPRINT 2 DEMO BAR", 0.92, 4.27, 2.0, 0.25, size=10, color=GREEN, bold=True)
    add_text(slide, "Rebuild it. Simulate it. Run the same seeded vectors on the PYNQ-Z2.", 0.92, 4.62, 10.9, 0.38, size=18, bold=True)
    add_text(slide, "Only after that: measure latency/resources, then integrate projections and real image/model inputs.", 0.92, 5.65, 11.3, 0.48, size=14, color=MUTED)

    prs.save(OUTPUT)
    return OUTPUT


def main():
    report = json.loads(REPORT_PATH.read_text())
    output = make_deck(report)
    print(f"Generated {output}")


if __name__ == "__main__":
    main()
