#!/usr/bin/env python3
"""Generate a self-contained, animated Sprint 2 browser presentation."""

from __future__ import annotations

import html
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "sim" / "results" / "verification_report.json"
SCRIPT = Path(__file__).with_name("Sprint2_Speaker_Script.md")
OUTPUT = Path(__file__).with_name("Sprint2_Interactive.html")


def extract_notes(markdown: str) -> list[str]:
    sections = re.findall(r"## Slide \d+.*?\n\n“(.*?)”", markdown, re.S)
    if len(sections) != 6:
        raise ValueError(f"Expected 6 speaker-script sections; found {len(sections)}")
    interactive_sections = re.findall(
        r"## Interactive slide: (System diagram|Streaming progress trace)\n\n“(.*?)”",
        markdown,
        re.S,
    )
    interactive = {title: note for title, note in interactive_sections}
    if set(interactive) != {"System diagram", "Streaming progress trace"}:
        raise ValueError("Expected talk tracks for both interactive slides")
    notes = [html.escape(" ".join(section.split())) for section in sections]
    notes[3:3] = [html.escape(" ".join(interactive["System diagram"].split()))]
    notes[5:5] = [
        html.escape(" ".join(interactive["Streaming progress trace"].split()))
    ]
    return notes


def build_stream_trace() -> list[dict[str, int | float]]:
    import numpy as np

    sys.path.insert(0, str(ROOT / "software"))
    from base2_softmax import WEIGHT_FRACTIONAL_BITS, WEIGHT_ONE, exp_neg_q15, streaming_attention

    rng = np.random.default_rng(20261108)

    def quantize(values: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(values * 16), -128, 127).astype(np.int8)

    query = quantize(rng.normal(0.0, 0.5, size=32))
    keys = quantize(rng.normal(0.0, 0.5, size=(196, 32)))
    values = quantize(rng.normal(0.0, 0.5, size=(196, 16)))
    query_real = query.astype(np.float64) / 16
    keys_real = keys.astype(np.float64) / 16
    values_real = values.astype(np.float64) / 16
    scores = np.fromiter(
        (float(np.dot(key, query_real)) / math.sqrt(query.size) for key in keys_real),
        dtype=np.float64,
        count=keys.shape[0],
    )
    scores_q12 = np.rint(scores * (1 << 12)).astype(np.int64)

    maximum: int | None = None
    denominator_q15 = 0
    accumulator = np.zeros(values.shape[1], dtype=np.float64)
    trace: list[dict[str, int | float]] = []
    for index, (score_value, value) in enumerate(zip(scores_q12, values_real)):
        score = int(score_value)
        if maximum is None:
            maximum = score
            denominator_q15 = WEIGHT_ONE
            accumulator = value.copy()
        else:
            new_maximum = max(maximum, score)
            alpha = exp_neg_q15(new_maximum - maximum, 12)
            beta = exp_neg_q15(new_maximum - score, 12)
            denominator_q15 = (
                (denominator_q15 * alpha) >> WEIGHT_FRACTIONAL_BITS
            ) + beta
            accumulator = (
                accumulator * (alpha / WEIGHT_ONE)
                + value * (beta / WEIGHT_ONE)
            )
            maximum = new_maximum

        trace.append(
            {
                "token": index + 1,
                "scoreQ12": score,
                "maxQ12": maximum,
                "denominatorQ15": denominator_q15,
                "output0": float(accumulator[0] / (denominator_q15 / WEIGHT_ONE)),
            }
        )

    reference = streaming_attention(query_real, keys_real, values_real)
    if not math.isclose(trace[-1]["output0"], float(reference[0]), abs_tol=1e-12):
        raise AssertionError("streaming trace does not match the reference output")
    return trace


def build_html(report: dict, notes: list[str]) -> str:
    numeric = report["numeric"]
    exp = report["lut_and_exponential"]
    protocol = report["protocol"]
    data = {
        "notes": notes,
        "trace": build_stream_trace(),
        "metrics": {
            "caseCount": numeric["numeric_cases"],
            "edgeCases": numeric["deterministic_edge_cases"],
            "seeds": numeric["random_seeds"],
            "pythonError": numeric["max_abs_output_error_vs_python"],
            "torchError": numeric["max_abs_output_error_vs_torch"],
            "tolerance": numeric["board_tolerance_dequantized_units"],
            "lutEntries": exp["rtl_lut_entries"],
            "expDeltas": exp["exp_input_deltas_checked"],
            "maxExpError": exp["max_abs_exp_error"],
            "protocolTrials": protocol["protocol_trials"],
            "inputWords": protocol["input_words_transferred"],
            "outputWords": protocol["output_words_transferred"],
        },
    }
    data_json = json.dumps(data, ensure_ascii=True).replace("</", "<\\/")
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#06182d">
<title>Sprint 2 | ViTAl-Hardware</title>
<style>
:root {
  color-scheme: dark;
  --bg: #06182d; --panel: #0d2b49; --panel2: #123a5e;
  --ink: #f3f8ff; --muted: #a4bad0; --blue: #68b7ff;
  --green: #62dda4; --amber: #e8ad66; --red: #f18a7e;
  --line: #315477; --shadow: 0 24px 80px #020b18aa;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
* { box-sizing: border-box }
html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: var(--bg); color: var(--ink) }
body { background-image: radial-gradient(ellipse at 86% 4%, #247ecb30, transparent 34%), radial-gradient(ellipse at 7% 96%, #25c89114, transparent 32%) }
button { font: inherit; color: inherit }
.app { height: 100%; display: grid; grid-template-rows: 1fr auto; padding: 22px clamp(16px, 3vw, 48px) 14px; gap: 12px }
.stage { position: relative; min-height: 0; border: 1px solid #37618a; border-radius: 26px; background: linear-gradient(145deg,#0d2b49f5,#071b32f7); box-shadow: var(--shadow); overflow: hidden }
.stage:before { content:""; position:absolute; inset:0; pointer-events:none; opacity:.18; background-image:radial-gradient(circle at 24px 24px,#8ec9ff70 0 2px,transparent 2.6px),radial-gradient(circle at 84px 84px,#8ec9ff55 0 2px,transparent 2.6px),linear-gradient(90deg,transparent 23px,#6ba6dc1c 24px 25px,transparent 26px 83px,#6ba6dc16 84px 85px,transparent 86px),linear-gradient(0deg,transparent 23px,#6ba6dc1c 24px 25px,transparent 26px 83px,#6ba6dc16 84px 85px,transparent 86px); background-size:120px 120px }
.slides { position:relative; width:100%; height:100% }
.slide { position:absolute; inset:0; padding:clamp(22px,4vw,58px); display:flex; flex-direction:column; gap:clamp(14px,2vh,24px); opacity:0; transform:translateX(48px) scale(.99); pointer-events:none; transition:opacity .42s ease, transform .55s cubic-bezier(.2,.8,.2,1); overflow:auto }
.slide.active { opacity:1; transform:none; pointer-events:auto; z-index:1 }
.eyebrow { color:var(--blue); font-size:clamp(10px,1vw,14px); font-weight:800; letter-spacing:.18em; text-transform:uppercase }
h1,h2,h3,p { margin:0 }
h1 { font-size:clamp(42px,7vw,94px); line-height:.97; letter-spacing:-.055em; max-width:900px }
h2 { font-size:clamp(27px,4vw,50px); line-height:1.04; letter-spacing:-.035em }
h3 { font-size:clamp(15px,1.5vw,20px) }
.subtitle { color:var(--muted); font-size:clamp(15px,1.7vw,22px); max-width:900px; line-height:1.45 }
.label { color:var(--muted); font-size:11px; font-weight:800; letter-spacing:.12em; text-transform:uppercase }
.small { color:var(--muted); font-size:clamp(11px,1vw,14px); line-height:1.5 }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace }
.pill { display:inline-flex; align-items:center; gap:8px; padding:9px 13px; border:1px solid var(--line); border-radius:99px; color:#d5e1ef; background:#0d2b49ad; font-size:12px; font-weight:700 }
.dot { width:8px; height:8px; display:inline-block; border-radius:50%; background:var(--green); box-shadow:0 0 12px currentColor }
.status { display:grid; grid-template-columns:1fr 1fr; gap:12px; width:min(580px,100%); margin-top:14px }
.status > div { background:#0d2b49cc; border:1px solid var(--line); border-radius:16px; padding:17px 20px }
.status strong { display:block; font-size:clamp(16px,2vw,24px); margin-top:7px }
.green { color:var(--green) }.amber { color:var(--amber) }.blue { color:var(--blue) }
.hero { flex:1; display:flex; align-items:center; justify-content:space-between; gap:30px }
.heroCopy { display:flex; flex-direction:column; gap:22px; max-width:850px }
.orbit { position:relative; width:min(28vw,330px); aspect-ratio:1; flex:0 0 auto; border:1px solid #5b8fd344; border-radius:50%; display:grid; place-items:center; background:radial-gradient(circle,#1e426028,transparent 67%) }
.orbit:before,.orbit:after { content:""; position:absolute; border:1px solid #5b8fd333; border-radius:50%; inset:15% }
.orbit:after { inset:31% }
.core { z-index:2; width:34%; aspect-ratio:1; border-radius:50%; display:grid; place-items:center; background:#123e55; border:1px solid #62dda466; color:var(--green); font-weight:900; box-shadow:0 0 42px #62dda233 }
.orbitNode { position:absolute; width:11px; height:11px; border-radius:50%; background:var(--blue); box-shadow:0 0 20px var(--blue); animation:orbit 9s linear infinite }
.orbitNode:nth-child(2) { animation-delay:-3s; background:var(--green); color:var(--green) }
.orbitNode:nth-child(3) { animation-delay:-6s }
@keyframes orbit { from{transform:rotate(0) translateX(125px) rotate(0)} to{transform:rotate(360deg) translateX(125px) rotate(-360deg)} }
.layout { display:grid; grid-template-columns:repeat(12,1fr); gap:clamp(10px,1.3vw,18px); align-items:stretch }
.card { position:relative; background:linear-gradient(145deg,#12395b,#0d2946); border:1px solid var(--line); border-radius:18px; padding:clamp(14px,1.7vw,24px); min-width:0; overflow:hidden }
.card:after { content:""; position:absolute; inset:auto -20% -72% -20%; height:100%; background:radial-gradient(ellipse,#4c8ee10d,transparent 65%); pointer-events:none }
.span3 { grid-column:span 3 }.span4{grid-column:span 4}.span5{grid-column:span 5}.span6{grid-column:span 6}.span7{grid-column:span 7}.span8{grid-column:span 8}.span12{grid-column:span 12}
.flow { display:grid; grid-template-columns:repeat(4,1fr); gap:clamp(8px,1.2vw,16px); align-items:center; margin-top:auto; margin-bottom:auto }
.flowNode { min-height:155px; cursor:pointer; transition:transform .2s,border-color .2s,box-shadow .2s }
.flowNode:hover,.flowNode:focus-visible { transform:translateY(-5px); border-color:#6faaff99; box-shadow:0 14px 40px #02081277; outline:none }
.flowNode .step { color:var(--blue); font-size:11px; font-weight:900; letter-spacing:.12em }
.flowNode h3 { margin:16px 0 9px }
.arrow { color:#7089a9; text-align:center; font-size:30px; animation:nudge 1.8s ease-in-out infinite }
@keyframes nudge { 50%{transform:translateX(5px);color:var(--green)} }
.streamViz { display:grid; grid-template-columns:1fr auto 1fr auto 1fr; align-items:center; gap:12px; margin:4px 0 12px }
.tokenWell { min-height:62px; display:flex; flex-wrap:wrap; align-content:center; gap:5px; padding:12px; border:1px solid var(--line); background:#0b1725; border-radius:14px }
.token { width:7px; height:7px; border-radius:50%; background:#4d76a6; opacity:.75; transition:background .3s,transform .3s,box-shadow .3s }
.token.hot { background:var(--green); transform:scale(1.65); box-shadow:0 0 12px #62dda4aa; opacity:1 }
.stateBox { text-align:center; min-width:130px; padding:15px; border-radius:14px; background:#123e39; border:1px solid #62dda455; transition:transform .25s,box-shadow .25s }
.stateBox.pulse { transform:scale(1.06); box-shadow:0 0 28px #62dda444 }
.equation { display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:center }
.eq { padding:11px 15px; border:1px solid #62dda455; border-radius:12px; background:#102e36; font-size:clamp(14px,1.6vw,21px); transition:transform .2s }
.eq.flash { transform:translateY(-4px); color:var(--green) }
.meter { height:12px; border-radius:99px; background:#263b55; overflow:hidden; margin:12px 0 8px }
.meter > i { display:block; height:100%; width:0; border-radius:inherit; background:linear-gradient(90deg,#41d5a5,#75a9ff); transition:width 1.3s cubic-bezier(.2,.8,.2,1) }
.metricValue { font-size:clamp(30px,4vw,54px); font-weight:850; letter-spacing:-.04em; font-variant-numeric:tabular-nums }
.plot { width:100%; height:210px; display:block }
.plot .gridline { stroke:#6e9bc455; stroke-width:1 }
.plot .line { fill:none; stroke:var(--blue); stroke-width:3; stroke-dasharray:1000; stroke-dashoffset:1000 }
.slide.active .plot .line { animation:draw 1.8s ease forwards }
@keyframes draw { to{stroke-dashoffset:0} }
.workflow { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:auto 0 }
.workStep { cursor:pointer; transition:transform .2s,border-color .2s }
.workStep:hover,.workStep.selected { border-color:#62dda488; transform:translateY(-4px) }
.workStep .num { width:35px;height:35px;border-radius:50%;display:grid;place-items:center;background:#123e55;color:var(--green);font-weight:900;margin-bottom:16px }
.detail { min-height:82px; border-left:3px solid var(--green); padding:12px 18px; color:#d5e1ef; background:#0d2b49; border-radius:0 12px 12px 0 }
.meterRow { display:grid; grid-template-columns:120px 1fr 80px; gap:12px; align-items:center; margin:13px 0; font-size:13px }
.meterRow .meter { margin:0 }
.statusRows { display:grid;gap:12px;margin-top:16px }
.statusRow { display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center;font-size:13px }
.statusTag { padding:5px 9px;border-radius:99px;background:#0a1f35;border:1px solid var(--line);font-size:10px;font-weight:900;letter-spacing:.08em;white-space:nowrap }
.statusTag.done { color:var(--green);border-color:#62dda466 }
.statusTag.pending { color:var(--amber);border-color:#e8ad6666 }
.check { display:flex; gap:12px; align-items:flex-start; margin:13px 0; font-size:14px; line-height:1.45 }
.check b { color:var(--green) }
.check.pending b { color:var(--amber) }
.controls { display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:16px; min-height:54px }
.controlGroup { display:flex; align-items:center; gap:8px }
.btn { border:1px solid var(--line); background:#0d2b49; border-radius:11px; padding:10px 14px; cursor:pointer; transition:background .2s,border-color .2s,transform .15s }
.btn:hover,.btn:focus-visible { background:#203650; border-color:#6faaff99; outline:none }
.btn:active { transform:scale(.97) }
.progress { height:5px; background:#23364b; border-radius:99px; overflow:hidden }
.progress > i { display:block;height:100%;width:0;background:linear-gradient(90deg,var(--blue),var(--green));transition:width .35s ease }
.toolbar { display:flex; justify-content:space-between; align-items:center; gap:10px }
.notes { position:fixed; z-index:8; right:22px; bottom:84px; width:min(520px,calc(100vw - 32px)); max-height:55vh; overflow:auto; display:none; padding:20px; border:1px solid #4b78a3; border-radius:18px; background:#0d2b49f5; box-shadow:var(--shadow); line-height:1.55 }
.notes.show { display:block; animation:appear .2s ease-out }
@keyframes appear { from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none} }
.modal { position:fixed; inset:0; z-index:9; display:none; place-items:center; padding:20px; background:#020812bb; backdrop-filter:blur(8px) }
.modal.show { display:grid }
.modalInner { width:min(600px,100%); padding:28px; border:1px solid #4b78a3; border-radius:22px; background:#0d2b49; box-shadow:var(--shadow) }
.modalClose { float:right }
.toast { position:fixed; z-index:10; bottom:84px; left:50%; transform:translate(-50%,10px); opacity:0; pointer-events:none; background:#0d2b49; border:1px solid var(--line); padding:10px 15px; border-radius:99px; color:var(--muted); transition:.2s }
.toast.show { transform:translate(-50%,0);opacity:1 }
.caseGrid { display:grid;grid-template-columns:repeat(16,1fr);gap:5px }
.caseCell { aspect-ratio:1;border-radius:3px;background:#62dda4; opacity:.82; animation:caseIn .35s both }
@keyframes caseIn { from{transform:scale(.2);opacity:0}to{transform:scale(1);opacity:.82} }
.diagramFrame,.traceFrame { border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#0d2b49,#081d34);overflow:hidden }
.diagramFrame { height:min(42vh,420px);min-height:220px;display:flex;justify-content:center;flex:none }
.systemDiagram { display:block;width:100%;height:100%;max-height:100% }
.slide[data-title="System diagram"] h2 { font-size:clamp(24px,3.4vw,42px) }
.systemDiagram .zone { fill:#0d2b49;stroke:#4b78a3;stroke-width:2 }
.systemDiagram .node { fill:#16416a;stroke:#6ca1d1;stroke-width:2 }
.systemDiagram .nodeAccent { fill:#123e55;stroke:#62dda4;stroke-width:2 }
.systemDiagram .wire { fill:none;stroke:#68b7ff;stroke-width:3;marker-end:url(#arrowBlue) }
.systemDiagram .wireGreen { fill:none;stroke:#62dda4;stroke-width:3;marker-end:url(#arrowGreen) }
.systemDiagram .wireControl { fill:none;stroke:#e8ad66;stroke-width:2;stroke-dasharray:8 7;marker-end:url(#arrowAmber) }
.systemDiagram text { font-family:inherit;fill:#f2f7fc }
.systemDiagram .caption { fill:#b3cae0;font-size:13px }
.systemDiagram .zoneTitle { fill:#8cc9ff;font-size:16px;font-weight:800;letter-spacing:2px }
.systemDiagram .wireLabel { fill:#f3f8ff;font-size:12px;font-weight:800 }
.systemDiagram .labelPlate { fill:#16416a;stroke:#6ca1d1;stroke-width:1 }
.systemDiagram .packet { fill:#c4e5ff;filter:drop-shadow(0 0 7px #68b7ff) }
.systemDiagram .packetOut { fill:#62dda4;filter:drop-shadow(0 0 7px #62dda4) }
.systemDiagram .pulseNode { transform-box:fill-box;transform-origin:center;animation:nodePulse 2.4s ease-in-out infinite }
.systemDiagram .pulseNode:nth-of-type(2) { animation-delay:.8s }
.systemDiagram .pulseNode:nth-of-type(3) { animation-delay:1.6s }
@keyframes nodePulse { 50%{filter:drop-shadow(0 0 14px #39d6a266);stroke:#74f0c5} }
.traceFrame { padding:8px 14px 4px;flex:0 0 auto }
.tracePlot { width:100%;height:clamp(110px,20vh,170px);display:block }
.traceGrid { stroke:#6e9bc455;stroke-width:1 }
.traceAxis { stroke:#a4bad0;stroke-width:1.5 }
.traceAxisText { fill:#b3cae0;font-size:13px }
.traceAxisTitle { fill:#f3f8ff;font-size:14px;font-weight:800 }
.traceFull { fill:none;stroke:#68b7ff65;stroke-width:3 }
.traceShown { fill:none;stroke:#62dda4;stroke-width:4;stroke-linecap:round;stroke-linejoin:round }
.traceCursor { stroke:#e8ad66;stroke-width:2;stroke-dasharray:5 6 }
.traceDot { fill:#e8ad66;stroke:#fff1cf;stroke-width:2;filter:drop-shadow(0 0 9px #e8ad66) }
.traceStats { display:grid;grid-template-columns:repeat(4,1fr);gap:10px }
.traceStat { border:1px solid var(--line);border-radius:13px;padding:8px 12px;background:#0d2b49 }
.traceStat strong { display:block;margin-top:4px;font-size:clamp(15px,1.7vw,21px);font-variant-numeric:tabular-nums }
.traceControls { display:flex;align-items:center;gap:8px;flex-wrap:wrap }
.traceControls .btn { padding:8px 11px;font-size:14px }
.traceControls .fine { font-size:10px }
.traceControls .fine { margin-left:auto }
.traceEquation { display:flex;justify-content:center;gap:10px;flex-wrap:wrap }
.traceEquation .eq { font-size:clamp(12px,1.3vw,17px) }
.slide[data-title="Streaming progress trace"] { gap:6px;padding:10px clamp(12px,2vw,24px) }
.slide[data-title="Streaming progress trace"] h2 { font-size:clamp(22px,3.2vw,40px) }
.fine { font-size:11px; line-height:1.45; color:var(--muted) }
.pause * { animation-play-state:paused !important }
@media(max-width:850px) {
 .app{padding:10px;gap:8px}.stage{border-radius:18px}
 .slide{padding:20px;gap:12px}.orbit{display:none}
 .flow{grid-template-columns:1fr 1fr}.arrow{display:none}
 .span3,.span4,.span5,.span6,.span7,.span8{grid-column:span 6}
 .workflow{grid-template-columns:1fr 1fr}
 .controls{gap:7px}.btn{padding:8px 10px}.toolbar .label{display:none}
}
@media(max-width:540px) {
 .status{grid-template-columns:1fr 1fr}
 .flow{grid-template-columns:1fr 1fr}.flowNode{min-height:120px}
 .layout{gap:8px}.span3,.span4,.span5,.span6,.span7,.span8{grid-column:span 12}
 .slide{overflow:auto}.controls{grid-template-columns:auto 1fr}
 .toolbar{grid-column:1/-1;justify-content:center}
 .caseGrid{grid-template-columns:repeat(12,1fr)}
 .traceStats{grid-template-columns:repeat(2,1fr)}
}
@media(prefers-reduced-motion:reduce) {
 *,*:before,*:after{animation-duration:.01ms !important;animation-iteration-count:1 !important;scroll-behavior:auto !important;transition-duration:.01ms !important}
}
</style>
</head>
<body>
<main class="app" id="app">
  <section class="stage" aria-label="Sprint 2 interactive presentation">
    <div class="slides">
      <article class="slide active" data-title="Sprint 2">
        <div class="eyebrow">ViTAl-Hardware · EC601 · Progress snapshot · 8 November 2026</div>
        <div class="hero">
          <div class="heroCopy">
            <h1>From tested model<br>toward a board demo</h1>
            <p class="subtitle">Streaming Base-2 attention on the PYNQ-Z2</p>
            <div class="status">
              <div><span class="label"><i class="dot"></i> Board bring-up</span><strong class="green">Complete</strong></div>
              <div><span class="label">Repository code on board</span><strong class="amber">Not yet tested</strong></div>
            </div>
            <span class="pill">Interactive Sprint 2 presentation <span class="blue">·</span> use ← → to navigate</span>
          </div>
          <div class="orbit" aria-hidden="true"><i class="orbitNode"></i><i class="orbitNode"></i><i class="orbitNode"></i><div class="core">STREAM</div></div>
        </div>
      </article>

      <article class="slide" data-title="Attention flow">
        <div class="eyebrow">01 / Demo scope</div>
        <h2>One attention head, streamed end to end</h2>
        <p class="subtitle">Select a stage for the plain-language explanation. The dots animate key rows into the running state.</p>
        <div class="flow">
          <button class="card flowNode" data-detail="One 32-number query is compared with 196 key vectors. Each key has 32 signed-byte values; its matching value vector has 16." aria-label="Explain input stage">
            <div class="step">01 · INPUT</div><h3>Query + key/value rows</h3><p class="small">32 query values<br>196 token rows</p>
          </button><div class="arrow" aria-hidden="true">→</div>
          <button class="card flowNode" data-detail="Each query/key dot product is scaled by 1 over the square root of 32 and rounded to the model's signed Q12 score format." aria-label="Explain score stage">
            <div class="step">02 · SCORE</div><h3>Dot product</h3><p class="small">q · k<br>scale + Q12 quantize</p>
          </button><div class="arrow" aria-hidden="true">→</div>
          <button class="card flowNode" data-detail="The running maximum stabilizes the weights. A 64-entry fractional LUT and shifts approximate the exponential; Q15 weights update the running denominator and weighted-value accumulator." aria-label="Explain online Softmax stage">
            <div class="step green">03 · ONLINE SOFTMAX</div><h3>Update state</h3><p class="small">running max + sum<br>64-entry LUT · Q15</p>
          </button><div class="arrow" aria-hidden="true">→</div>
          <button class="card flowNode" data-detail="After all 196 rows have been consumed, divide the weighted-value accumulator by the weight denominator. The demo emits 16 signed Q4 outputs." aria-label="Explain output stage">
            <div class="step">04 · OUTPUT</div><h3>Normalize</h3><p class="small">a / l<br>16 signed Q4 values</p>
          </button>
        </div>
        <div class="streamViz">
          <div><div class="label">196 streamed tokens</div><div class="tokenWell" id="tokens" aria-label="Animated key-token stream"></div></div>
          <span class="arrow">→</span>
          <div class="stateBox" id="stateBox"><div class="label">Running state</div><strong class="mono">m, l, a</strong><div class="small">max · denominator · values</div></div>
          <span class="arrow">→</span>
          <div class="stateBox" style="border-color:#4b78a3;background:#123253"><div class="label">Output vector</div><strong class="mono blue">16 values</strong><div class="small">not a score matrix</div></div>
        </div>
        <div class="detail" id="flowDetail" aria-live="polite">Click one of the four pipeline blocks to inspect what that stage does.</div>
      </article>

      <article class="slide" data-title="Implementation progress">
        <div class="eyebrow">02 / Implementation</div>
        <h2>The demo path is represented in code</h2>
        <p class="subtitle">Click each milestone to see the deliverable and its current evidence state.</p>
        <div class="workflow">
          <button class="card workStep selected" data-work="rtl"><div class="num">01</div><h3>RTL kernel</h3><p class="small">AXI stream · Q12/Q15<br>running max + sum</p></button>
          <button class="card workStep" data-work="build"><div class="num">02</div><h3>Vivado recipe</h3><p class="small">Zynq PS + AXI DMA<br>overlay build Tcl</p></button>
          <button class="card workStep" data-work="driver"><div class="num">03</div><h3>PYNQ runner</h3><p class="small">seeded vectors<br>DMA + reference compare</p></button>
          <button class="card workStep" data-work="tests"><div class="num">04</div><h3>Host verification</h3><p class="small">integer mirror<br>edge + random cases</p></button>
        </div>
        <div class="detail" id="workDetail" aria-live="polite">RTL source exists. It has not yet been compiled or run in an HDL simulator.</div>
        <div class="layout"><div class="card span6"><div class="label">Milestone tracker</div><div class="statusRows"><div class="statusRow"><span>Software reference</span><b class="statusTag done">TESTS PASS</b></div><div class="statusRow"><span>PYNQ-Z2 bring-up</span><b class="statusTag done">COMPLETE</b></div><div class="statusRow"><span>Repository RTL / overlay</span><b class="statusTag pending">NOT TESTED</b></div><div class="statusRow"><span>Attention demo on board</span><b class="statusTag pending">PENDING</b></div></div></div><div class="card span6"><div class="label">What changed this sprint</div><h3 style="margin-top:12px">Hardware bring-up is complete.</h3><p class="small" style="margin-top:10px">The next proof is repository code: compile and simulate RTL, build the overlay, and run the project's DMA test on the board.</p></div></div>
      </article>

      <article class="slide" data-title="System diagram">
        <div class="eyebrow">03 / Board system · bring-up complete</div>
        <h2>One streamed path from DDR to attention output</h2>
        <p class="subtitle">The board is brought up; these moving packets show the repository's proposed data path, which has not yet been run.</p>
        <div class="diagramFrame">
          <svg class="systemDiagram" id="systemDiagram" viewBox="0 0 1200 480" role="img" aria-label="Animated PYNQ-Z2 system diagram showing Python on ARM, DDR buffers, AXI DMA, streaming attention in programmable logic, and output comparison">
            <defs>
              <marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 10 5 0 10z" fill="#68b7ff"/></marker>
              <marker id="arrowGreen" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 10 5 0 10z" fill="#62dda4"/></marker>
              <marker id="arrowAmber" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 10 5 0 10z" fill="#e8ad66"/></marker>
            </defs>
            <rect class="zone" x="22" y="46" width="290" height="390" rx="22"/>
            <rect class="zone" x="354" y="46" width="820" height="390" rx="22"/>
            <text class="zoneTitle" x="48" y="78">PROCESSING SYSTEM</text>
            <text class="caption" x="48" y="101">ARM Cortex-A9 · PYNQ Linux</text>
            <rect class="node" x="50" y="128" width="230" height="92" rx="16"/>
            <text x="72" y="163" font-size="20" font-weight="800">Python app</text>
            <text class="caption" x="72" y="189">pack inputs · configure DMA</text>
            <rect class="node" x="50" y="286" width="230" height="100" rx="16"/>
            <text x="72" y="322" font-size="20" font-weight="800">DDR buffers</text>
            <text class="caption" x="72" y="348">Q / K / V input · 16 outputs</text>
            <text class="caption" x="72" y="369">Python reads result + compares</text>
            <path class="wireControl" d="M280 162 C340 162 335 128 394 128" />
            <rect class="labelPlate" x="294" y="108" width="146" height="25" rx="8"/>
            <text class="wireLabel" x="367" y="125" text-anchor="middle">AXI-Lite control</text>
            <path class="wire" d="M280 330 C330 330 342 215 394 215" />
            <rect class="labelPlate" x="294" y="289" width="146" height="25" rx="8"/>
            <text class="wireLabel" x="367" y="306" text-anchor="middle">DDR read · HP0</text>
            <path class="wireGreen" d="M394 260 C350 290 327 326 280 360" />
            <rect class="labelPlate" x="290" y="365" width="202" height="25" rx="8"/>
            <text class="wireLabel" x="391" y="382" text-anchor="middle">S2MM writes outputs to DDR</text>

            <text class="zoneTitle" x="382" y="78">PROGRAMMABLE LOGIC</text>
            <text class="caption" x="382" y="101">AXI DMA moves data; custom attention core updates m, l, a per key</text>
            <rect class="node" x="394" y="146" width="190" height="138" rx="17"/>
            <text x="416" y="181" font-size="19" font-weight="800">AXI DMA</text>
            <text class="caption" x="416" y="208">MM2S · memory to stream</text>
            <text class="caption" x="416" y="233">S2MM · stream to memory</text>
            <text class="caption" x="416" y="258">32-bit AXI words</text>
            <rect class="nodeAccent pulseNode" x="700" y="146" width="238" height="138" rx="17"/>
            <text x="724" y="181" font-size="19" font-weight="800">Attention core</text>
            <text class="caption" x="724" y="208">QK score · Q12</text>
            <text class="caption" x="724" y="233">64-entry LUT · online softmax</text>
            <text class="caption" x="724" y="258">16 running value accumulators</text>
            <path id="sysInPath" class="wire" d="M584 205 L700 205"/>
            <text class="caption" x="642" y="188" text-anchor="middle">MM2S input stream</text>
            <path id="sysOutPath" class="wireGreen" d="M700 250 C666 292 620 292 584 250"/>
            <rect class="labelPlate" x="598" y="300" width="200" height="25" rx="8"/>
            <text class="wireLabel" x="698" y="317" text-anchor="middle">S2MM · 16 output values</text>
            <text class="caption" x="700" y="365">Python reads DDR output and compares with NumPy reference</text>
            <circle class="packet" r="7"><animateMotion dur="2.8s" repeatCount="indefinite"><mpath href="#sysInPath"/></animateMotion></circle>
            <circle class="packet" r="5"><animateMotion dur="2.8s" begin="-1.4s" repeatCount="indefinite"><mpath href="#sysInPath"/></animateMotion></circle>
            <circle class="packetOut" r="7"><animateMotion dur="2.2s" repeatCount="indefinite"><mpath href="#sysOutPath"/></animateMotion></circle>
            <circle class="packetOut" r="5"><animateMotion dur="2.2s" begin="-1.1s" repeatCount="indefinite"><mpath href="#sysOutPath"/></animateMotion></circle>
          </svg>
        </div>
        <p class="fine">Board bring-up is complete. Repository RTL, overlay build, DMA transfer, and FPGA output comparison remain untested.</p>
      </article>

      <article class="slide" data-title="Numerical evidence">
        <div class="eyebrow">04 / Numerical evidence</div>
        <h2>132 synthetic cases stayed within the demo threshold</h2>
        <p class="subtitle">Four deterministic edge cases plus 128 seeded random vectors. Click “replay” to animate the result comparison.</p>
        <div class="layout">
          <div class="card span6"><div class="label">RTL arithmetic mirror vs Python reference</div><div class="metricValue" id="pythonMetric">0.00000</div><div class="small">worst maximum absolute difference · value units</div><div class="meter"><i id="pythonBar"></i></div><div class="small">threshold <strong class="green" id="toleranceText">0.20</strong> · smaller is better</div></div>
          <div class="card span6"><div class="label">RTL arithmetic mirror vs PyTorch</div><div class="metricValue blue" id="torchMetric">0.00000</div><div class="small">same Q4 inputs · numerical comparison, not bit-identical behavior</div><div class="meter"><i id="torchBar"></i></div><div class="small">worst case across the test set</div></div>
          <div class="card span5"><div class="label">Coverage · click replay</div><div class="metricValue" id="caseCount">132</div><div class="caseGrid" id="caseGrid" aria-label="One animated square per numerical test case"></div><p class="small" style="margin-top:8px">4 edge + 128 seeded random cases</p></div>
          <div class="card span7"><div class="label">Output error relative to threshold</div><svg class="plot" viewBox="0 0 700 210" role="img" aria-label="Comparison of maximum output error and tolerance"><line class="gridline" x1="70" y1="30" x2="670" y2="30"/><line class="gridline" x1="70" y1="85" x2="670" y2="85"/><line class="gridline" x1="70" y1="140" x2="670" y2="140"/><line class="gridline" x1="70" y1="175" x2="670" y2="175"/><text x="12" y="34" fill="#91a4bc" font-size="12">0.20</text><text x="12" y="89" fill="#91a4bc" font-size="12">0.15</text><text x="12" y="144" fill="#91a4bc" font-size="12">0.10</text><text x="12" y="179" fill="#91a4bc" font-size="12">0.00</text><line x1="70" y1="30" x2="670" y2="30" stroke="#ffbe60" stroke-dasharray="6 6"/><path class="line" d="M75 175 L180 173 L285 170 L390 168 L495 165 L600 162 L660 160"/><circle cx="660" cy="160" r="7" fill="#39d6a2"/><text x="75" y="202" fill="#91a4bc" font-size="12">seeded cases →</text><text x="520" y="52" fill="#ffbe60" font-size="12">0.20 tolerance</text></svg><button class="btn" id="replay" type="button">↻ Replay visual</button></div>
        </div>
        <p class="fine">Synthetic inputs only. The tolerance is a demo acceptance setting, not a measured image-accuracy requirement.</p>
      </article>

      <article class="slide" data-title="Streaming progress trace">
        <div class="eyebrow">05 / Animated progress trace</div>
        <h2>Watch the answer settle as keys stream in</h2>
        <div class="traceFrame">
          <svg class="tracePlot" id="tracePlot" viewBox="0 0 960 180" preserveAspectRatio="none" role="img" aria-label="Animated line graph of output value zero against the number of streamed keys, with labeled axes">
            <g id="traceYTicks"></g>
            <line class="traceAxis" x1="100" y1="132" x2="930" y2="132"/>
            <line class="traceAxis" x1="100" y1="24" x2="100" y2="132"/>
            <text class="traceAxisText" x="100" y="149" text-anchor="middle">1</text>
            <text class="traceAxisText" x="930" y="149" text-anchor="middle">196</text>
            <text class="traceAxisTitle" x="515" y="173" text-anchor="middle">Keys consumed (token index)</text>
            <text class="traceAxisTitle" x="23" y="78" text-anchor="middle" transform="rotate(-90 23 78)">Output value[0]</text>
            <path class="traceFull" id="traceFull"/>
            <path class="traceShown" id="traceShown"/>
            <line class="traceCursor" id="traceCursor" x1="100" y1="24" x2="100" y2="132"/>
            <circle class="traceDot" id="traceDot" cx="100" cy="132" r="7"/>
          </svg>
        </div>
        <div class="traceControls">
          <button class="btn" type="button" id="tracePlay">▶ Play stream</button>
          <button class="btn" type="button" id="traceStep">Step one key</button>
          <button class="btn" type="button" id="traceReset">↺ Restart</button>
          <span class="fine">Fixed seed 20261108 · Python reference trace, not an FPGA waveform</span>
        </div>
        <div class="traceStats">
          <div class="traceStat"><span class="label">Keys consumed</span><strong id="traceToken" class="blue">1 / 196</strong></div>
          <div class="traceStat"><span class="label">Running max · Q12</span><strong id="traceMax" class="amber">—</strong></div>
          <div class="traceStat"><span class="label">Running sum · Q15</span><strong id="traceDenom" class="blue">—</strong></div>
          <div class="traceStat"><span class="label">Output value[0]</span><strong id="traceOutput" class="green">—</strong></div>
        </div>
      </article>

      <article class="slide" data-title="Streaming evidence">
        <div class="eyebrow">06 / Streaming evidence</div>
        <h2>The software checks support the streaming design claim</h2>
        <p class="subtitle">The design stores working state as it consumes keys. The checks below model expected behavior; they do not execute the SystemVerilog.</p>
        <div class="layout">
          <div class="card span7"><div class="label">Expected attention working set</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:18px">
              <div class="card" style="padding:13px"><span class="label">Query</span><h3 style="margin-top:8px">32</h3><p class="fine">int8 values</p></div>
              <div class="card" style="padding:13px"><span class="label">Key</span><h3 style="margin-top:8px">32</h3><p class="fine">current row</p></div>
              <div class="card" style="padding:13px"><span class="label">Value</span><h3 style="margin-top:8px">16</h3><p class="fine">current row</p></div>
              <div class="card" style="padding:13px"><span class="label">Accumulator</span><h3 style="margin-top:8px">16</h3><p class="fine">output lanes</p></div>
            </div>
            <svg viewBox="0 0 700 125" class="plot" style="height:130px" aria-label="Small linear working state compared with a large square score matrix"><rect x="14" y="26" width="270" height="62" rx="12" fill="#123e39" stroke="#62dda4"/><text x="34" y="52" fill="#f3f8ff" font-size="16">Q + current K/V + running state</text><text x="34" y="73" fill="#62dda4" font-size="12">bounded working state per query</text><path d="M305 57h55" stroke="#a4bad0" stroke-width="2"/><path d="m351 50 9 7-9 7" fill="none" stroke="#a4bad0" stroke-width="2"/><rect x="386" y="15" width="90" height="90" rx="5" fill="#f18a7e22" stroke="#f18a7e"/><path d="M389 18l84 84M473 18l-84 84" stroke="#f18a7e66"/><text x="495" y="49" fill="#f18a7e" font-size="15">196 × 196</text><text x="495" y="71" fill="#f6b3aa" font-size="12">score matrix</text><text x="495" y="91" fill="#f6b3aa" font-size="11">not materialized</text></svg>
          </div>
          <div class="card span5"><div class="label">Ready / valid schedule model</div><div class="metricValue" id="trialCount">20</div><p class="small">randomized source gaps + output back-pressure</p><div class="meter"><i data-meter="100"></i></div><div class="check"><b>✓</b><span id="wordCount"></span></div><div class="check"><b>✓</b><span>16 result values per transaction; same protocol schedule completes under stalls.</span></div><div class="check pending"><b>!</b><span>This schedule was simulated in Python, not in an RTL simulator.</span></div></div>
        </div>
      </article>

      <article class="slide" data-title="Next proof points">
        <div class="eyebrow">07 / Next proof points</div>
        <h2>Bring-up is done; now validate the repository code</h2>
        <p class="subtitle">Click each stage to expand the proof artifact we need before claiming the FPGA demo works.</p>
        <div class="workflow">
          <button class="card workStep selected" data-next="simulate"><div class="num">01</div><h3>Simulate RTL</h3><p class="small">compile · testbench · logs</p></button>
          <button class="card workStep" data-next="build"><div class="num">02</div><h3>Build overlay</h3><p class="small">bitstream · timing · resources</p></button>
          <button class="card workStep" data-next="board"><div class="num">03</div><h3>Run repo DMA code</h3><p class="small">overlay · seeded vectors</p></button>
          <button class="card workStep" data-next="compare"><div class="num">04</div><h3>Compare output</h3><p class="small">FPGA result vs Python</p></button>
        </div>
        <div class="detail" id="nextDetail" aria-live="polite">Board bring-up and initial checks are complete. Repository RTL has not yet been compiled or run.</div>
        <div class="layout"><div class="card span8"><div class="label">Next proof bar</div><h3 style="font-size:clamp(20px,2.4vw,31px);margin-top:15px">Compile it. Simulate it. Build the overlay. Run the same vectors on the PYNQ-Z2.</h3><p class="small" style="margin-top:12px">Then add projections and real image/model inputs; measure latency and power before making performance claims.</p></div><div class="card span4" style="border-color:#e8ad6677"><div class="label amber">Current evidence boundary</div><h3 class="amber" style="margin-top:15px">Board ready. Repository code not yet tested.</h3></div></div>
      </article>
    </div>
  </section>
  <footer class="controls">
    <div class="controlGroup"><button class="btn" id="prev" aria-label="Previous slide">←</button><button class="btn" id="next" aria-label="Next slide">→</button><span class="label" id="counter">1 / 6</span></div>
    <div class="progress" aria-label="Presentation progress"><i id="progressBar"></i></div>
    <div class="toolbar"><button class="btn" id="notesBtn">Speaker notes</button><button class="btn" id="pauseBtn">Pause motion</button><button class="btn" id="fullBtn">Fullscreen</button></div>
  </footer>
</main>
<aside class="notes" id="notes" aria-live="polite"></aside>
<div class="modal" id="modal" role="dialog" aria-modal="true" aria-labelledby="modalTitle"><div class="modalInner"><button class="btn modalClose" id="modalClose">Close ×</button><div class="eyebrow">Acronym / detail</div><h2 id="modalTitle" style="font-size:29px;margin:12px 0"></h2><p id="modalText" class="subtitle" style="font-size:17px"></p></div></div>
<div class="toast" id="toast" role="status">Use ← → or Space to navigate. Press N for speaker notes.</div>
<script>
const DATA = __DATA__;
const slides = [...document.querySelectorAll(".slide")];
let current = 0, paused = false, toastTimer;
const $ = id => document.getElementById(id);
function showSlide(index) {
  current = (index + slides.length) % slides.length;
  slides.forEach((s,i) => { s.classList.toggle("active", i===current); s.setAttribute("aria-hidden",i===current?"false":"true"); });
  $("counter").textContent = `${current+1} / ${slides.length}`;
  $("progressBar").style.width = `${100*(current+1)/slides.length}%`;
  $("notes").innerHTML = `<span class="label">Speaker notes · slide ${current+1}</span><p style="margin-top:12px">${DATA.notes[current]}</p>`;
  const title = slides[current].dataset.title;
  if(title==="Attention flow") startTokenAnimation();
  if(title==="Implementation progress" || title==="Streaming evidence") animateMeters();
  if(title==="System diagram") syncSystemMotion();
  if(title==="Numerical evidence") replayMetrics();
  if(title==="Streaming progress trace") startTraceAnimation(true);
  else stopTraceAnimation();
}
function next(){showSlide(current+1)} function prev(){showSlide(current-1)}
$("next").addEventListener("click",next); $("prev").addEventListener("click",prev);
document.addEventListener("keydown",e=>{
  if($("modal").classList.contains("show")){if(e.key==="Escape")$("modal").classList.remove("show");return}
  if(["ArrowRight"," ","PageDown"].includes(e.key)){e.preventDefault();next()}
  if(["ArrowLeft","PageUp"].includes(e.key)){e.preventDefault();prev()}
  if(e.key.toLowerCase()==="n")$("notes").classList.toggle("show");
  if(e.key.toLowerCase()==="m")toggleMotion();
  if(e.key==="Escape")$("notes").classList.remove("show");
});
$("notesBtn").addEventListener("click",()=>$("notes").classList.toggle("show"));
$("fullBtn").addEventListener("click",async()=>{try{if(!document.fullscreenElement)await document.documentElement.requestFullscreen();else await document.exitFullscreen()}catch{showToast("Fullscreen is unavailable in this browser.")}});
function syncSystemMotion(){const svg=$("systemDiagram");if(!svg)return;if(paused)svg.pauseAnimations();else svg.unpauseAnimations()}
function toggleMotion(){paused=!paused;$("app").classList.toggle("pause",paused);$("pauseBtn").textContent=paused?"Resume motion":"Pause motion";syncSystemMotion()}
$("pauseBtn").addEventListener("click",toggleMotion);
function showToast(text){$("toast").textContent=text;$("toast").classList.add("show");clearTimeout(toastTimer);toastTimer=setTimeout(()=>$("toast").classList.remove("show"),2600)}
function animateMeters(){document.querySelectorAll(".slide.active [data-meter]").forEach(el=>{el.style.width="0";requestAnimationFrame(()=>el.style.width=el.dataset.meter+"%")})}
const flowCopy=[...document.querySelectorAll(".flowNode")];
flowCopy.forEach(button=>button.addEventListener("click",()=>{$("flowDetail").textContent=button.dataset.detail;flowCopy.forEach(b=>b.style.borderColor="");button.style.borderColor="#39d6a2"}));
const workCopy={rtl:"RTL source exists. It has not yet been compiled or run in an HDL simulator.",build:"A Vivado Tcl build recipe exists. Vivado is unavailable in the current validation environment, so no bitstream has been produced.",driver:"The PYNQ DMA runner is written, but has not yet executed against the physical board or a generated overlay.",tests:"The host verifier ran 132 numeric cases and produced JSON, CSV, Markdown, and SVG reports. This validates a software mirror, not the RTL itself."};
document.querySelectorAll("[data-work]").forEach(button=>button.addEventListener("click",()=>{document.querySelectorAll("[data-work]").forEach(b=>b.classList.remove("selected"));button.classList.add("selected");$("workDetail").textContent=workCopy[button.dataset.work]}));
const nextCopy={simulate:"First repository-code proof: compile/elaborate the RTL and pass the testbench with captured simulator output.",build:"Generate the bitstream and HWH, then capture implementation timing and resource reports.",board:"Load the overlay and run the repository's PYNQ DMA runner with deterministic Q/K/V vectors.",compare:"Capture actual board output and compare it with the Python reference; repeat with the same seed."};
document.querySelectorAll("[data-next]").forEach(button=>button.addEventListener("click",()=>{document.querySelectorAll("[data-next]").forEach(b=>b.classList.remove("selected"));button.classList.add("selected");$("nextDetail").textContent=nextCopy[button.dataset.next]}));
const explain={
  "Q12":"A fixed-point score representation with 12 fractional bits. The score is stored as an integer, interpreted as that integer divided by 4096.",
  "Q15":"A fixed-point weight representation with 15 fractional bits. In this model, 32768 represents weight 1.0.",
  "Q4":"The demo input/output integer is interpreted with four fractional bits: divide the stored integer by 16 to get its approximate real value.",
  "AXI stream":"A streaming hardware interface: data is transferred when valid and ready are both high on a clock edge.",
  "DMA":"Direct Memory Access. The processor asks a DMA engine to move data between memory and the FPGA stream without copying each word itself."
};
document.querySelectorAll(".slide").forEach(s=>s.addEventListener("dblclick",e=>{const words=Object.keys(explain);const word=words.find(w=>e.target.textContent.includes(w));if(!word)return;$("modalTitle").textContent=word;$("modalText").textContent=explain[word];$("modal").classList.add("show")}));
$("modalClose").addEventListener("click",()=>$("modal").classList.remove("show"));
$("modal").addEventListener("click",e=>{if(e.target===$("modal"))$("modal").classList.remove("show")});
const tokens=$("tokens");
for(let i=0;i<196;i++){const dot=document.createElement("i");dot.className="token";tokens.appendChild(dot)}
let tokenTimer;
function startTokenAnimation(){clearInterval(tokenTimer);const dots=[...tokens.children];let idx=0;tokenTimer=setInterval(()=>{if(paused)return;dots.forEach(d=>d.classList.remove("hot"));dots[idx%dots.length].classList.add("hot");$("stateBox").classList.add("pulse");setTimeout(()=>$("stateBox").classList.remove("pulse"),170);idx++},90)}
function replayMetrics(){
 const m=DATA.metrics;
 $("toleranceText").textContent=m.tolerance.toFixed(2);
 $("caseCount").textContent="0";
 $("pythonMetric").textContent="0.00000"; $("torchMetric").textContent="0.00000";
 $("pythonBar").style.width="0"; $("torchBar").style.width="0";
 const grid=$("caseGrid");grid.innerHTML="";
 let c=0;const caseTimer=setInterval(()=>{if(paused)return;if(c>=m.caseCount){clearInterval(caseTimer);return}
   const cell=document.createElement("i");cell.className="caseCell";cell.style.animationDelay=`${(c%16)*12}ms`;grid.appendChild(cell);c++;$("caseCount").textContent=c;
 },8);
 setTimeout(()=>{if(slides[current].dataset.title!=="Numerical evidence")return;$("pythonMetric").textContent=m.pythonError.toFixed(5);$("torchMetric").textContent=m.torchError.toFixed(5);$("pythonBar").style.width=`${Math.min(100,100*m.pythonError/m.tolerance)}%`;$("torchBar").style.width=`${Math.min(100,100*m.torchError/m.tolerance)}%`},850);
}
$("replay").addEventListener("click",replayMetrics);
let traceIndex=0, traceTimer, traceRun=0;
const traceValues=DATA.trace.map(point=>point.output0);
const traceLow=Math.min(...traceValues), traceHigh=Math.max(...traceValues);
const tracePad=(traceHigh-traceLow)*.12||.1;
const traceY=value=>132-(value-(traceLow-tracePad))/(traceHigh-traceLow+2*tracePad)*108;
const traceX=index=>100+index/(DATA.trace.length-1)*830;
const tracePath=DATA.trace.map((point,index)=>`${index?"L":"M"}${traceX(index).toFixed(2)} ${traceY(point.output0).toFixed(2)}`).join(" ");
$("traceFull").setAttribute("d",tracePath);
const tickRange=traceHigh-traceLow+2*tracePad;
$("traceYTicks").innerHTML=Array.from({length:5},(_,index)=>{
 const y=24+index*27, value=traceHigh+tracePad-index*tickRange/4;
 return `<line class="traceGrid" x1="100" y1="${y}" x2="930" y2="${y}"/><text class="traceAxisText" x="88" y="${y+4}" text-anchor="end">${value.toFixed(3)}</text>`;
}).join("");
function renderTrace(index){
 traceIndex=Math.max(0,Math.min(DATA.trace.length-1,index));
 const point=DATA.trace[traceIndex], x=traceX(traceIndex), y=traceY(point.output0);
 $("traceShown").setAttribute("d",DATA.trace.slice(0,traceIndex+1).map((item,i)=>`${i?"L":"M"}${traceX(i).toFixed(2)} ${traceY(item.output0).toFixed(2)}`).join(" "));
 $("traceCursor").setAttribute("x1",x);$("traceCursor").setAttribute("x2",x);
 $("traceDot").setAttribute("cx",x);$("traceDot").setAttribute("cy",y);
 $("traceToken").textContent=`${point.token} / ${DATA.trace.length}`;
 $("traceMax").textContent=point.maxQ12.toLocaleString();
 $("traceDenom").textContent=point.denominatorQ15.toLocaleString();
 $("traceOutput").textContent=point.output0.toFixed(4);
}
function startTraceAnimation(reset){
 stopTraceAnimation();
 if(reset)renderTrace(0);
 $("tracePlay").textContent="Ⅱ Pause stream";
 const run=traceRun;
 traceTimer=setInterval(()=>{
   if(run!==traceRun)return;
   if(paused||slides[current].dataset.title!=="Streaming progress trace")return;
   if(traceIndex>=DATA.trace.length-1){clearInterval(traceTimer);traceTimer=null;$("tracePlay").textContent="▶ Play stream";return}
   renderTrace(traceIndex+1);
 },65);
}
function stopTraceAnimation(){clearInterval(traceTimer);traceTimer=null;traceRun++}
$("tracePlay").addEventListener("click",()=>{
 if(traceTimer){stopTraceAnimation();$("tracePlay").textContent="▶ Play stream"}
 else {if(traceIndex>=DATA.trace.length-1)traceIndex=0;startTraceAnimation(false)}
});
$("traceStep").addEventListener("click",()=>{
 stopTraceAnimation();$("tracePlay").textContent="▶ Play stream";
 renderTrace(traceIndex>=DATA.trace.length-1?0:traceIndex+1);
});
$("traceReset").addEventListener("click",()=>startTraceAnimation(true));
renderTrace(0);
$("trialCount").textContent=DATA.metrics.protocolTrials;
$("wordCount").textContent=`${DATA.metrics.inputWords.toLocaleString()} input words and ${DATA.metrics.outputWords.toLocaleString()} output words accounted for across the randomized schedule tests.`;
showSlide(0);
setTimeout(()=>showToast("Tip: click diagram stages, press N for notes, or use ← → to navigate."),650);
</script>
</body>
</html>
""".replace("__DATA__", data_json)


def main() -> None:
    report = json.loads(REPORT.read_text())
    notes = extract_notes(SCRIPT.read_text())
    OUTPUT.write_text(build_html(report, notes), encoding="utf-8")
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
