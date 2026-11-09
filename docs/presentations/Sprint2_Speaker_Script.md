# Sprint 2 Progress — Five-Minute Speaker Script

**Deck:** [Sprint2_Progress.pptx](./Sprint2_Progress.pptx)
**Interactive version:** [Sprint2_Interactive.html](./Sprint2_Interactive.html)
**PowerPoint length:** 5:00 at a clear presentation pace
**Status snapshot:** 8 November 2026

> Accuracy note: board bring-up and initial hardware checks are complete.
> Repository code has not been run on the board: the SystemVerilog has not
> been compiled or run in an HDL simulator, no Vivado bitstream has been
> built, and the repository's DMA demo has not been tested.

For the interactive version, navigate with the arrow keys or bottom buttons.
Click pipeline stages and milestones to expand explanations; use **Speaker
notes** for the narration, **Pause motion** to stop animations, and
**Fullscreen** to present.
The browser version adds two slides beyond the timed PowerPoint; optional
talk tracks for those slides follow the timing table.

## Slide 1 — Sprint 2: from model to board demo (0:30)

“This Sprint 2 update is about moving from an algorithm we can test in Python
toward a small attention kernel for the PYNQ-Z2. Board bring-up and the
initial hardware checks are complete. The repository code has not yet been
tested on the board: there is no RTL simulation, Vivado build, or DMA demo
result. I’ll cover what is implemented, what the software numbers show, and
the next validation steps.”

## Slide 2 — One attention head, streamed end to end (0:50)

“The first target is intentionally smaller than a full Vision Transformer.
The demo accepts one 32-element query and 196 key/value pairs. For each key,
it calculates a dot product and Q12 score, then updates a running maximum and
weighted sum. A 64-entry fractional lookup and Q15 weights approximate the
exponential. The result is 16 weighted value components.

Keys are processed one at a time, so the design does not store a 196-by-196
score table. This is only the attention-normalization kernel; Q/K/V
projections and image classification are not included.”

## Slide 3 — The demo path is represented in code (0:55)

“The repository has four pieces for the demo path: an AXI-stream
SystemVerilog module, a Vivado Tcl recipe for the Zynq processing system and
AXI DMA, a PYNQ runner that sends deterministic inputs and compares output
with Python, and a host verification harness plus RTL testbench. Board
bring-up is now complete. These code artifacts remain untested: next we must
compile and simulate the RTL, build the overlay, and run the repository's DMA
test on the board.”

## Slide 4 — What the local numerical tests say (1:00)

“The local verification ran 132 synthetic input cases: four edge cases and
128 repeatable random seeds. It checks the Python integer-arithmetic mirror
against the NumPy reference, and also compares against PyTorch. The largest
absolute output differences were 0.03124 from Python and 0.03137 from
PyTorch.

The demo threshold is 0.20 dequantized value units, and every software case
was below it. All 64 LUT constants matched Python; 65,536 exponential inputs
matched and were monotonic. This supports consistency of the fixed-point
model. Inputs are synthetic, and that threshold is a demo setting—not an
image-accuracy result.”

## Slide 5 — Streaming evidence and its limits (0:50)

“The report models 20 ready/valid schedules with randomized input gaps and
output back-pressure, accounting for 188,800 input and 320 output words.
Source inspection finds no N-by-N score buffer; the modeled state is the
query, current key/value, recurrence state, and output accumulators.

These checks use Python or source inspection, not SystemVerilog execution.
Board bring-up and initial checks are complete, but there is no RTL
simulation, Vivado timing or resource report, power measurement, or
repository-code DMA result. The software mirror passes; the FPGA attention
kernel remains unverified.”

## Slide 6 — The next proof points (0:55)

“Now that board bring-up and initial checks are complete, the next steps are
to validate the repository code. First, compile and run the HDL testbench in
Vivado simulation. Then build the overlay and run the same seeded test
through the repository's DMA runner. Compare actual board output with Python.

The proof we need is a passing simulation, a reproducible build, and a board
comparison. Only then can we claim the kernel works in programmable logic.
Next we can add projection MACs, real model inputs, and measure latency and
power. This is a step toward the accelerator, not the finished system.”

## Timing check

| Slide | Target |
|---|---:|
| 1. Sprint 2: from model to board demo | 0:30 |
| 2. One attention head, streamed end to end | 0:50 |
| 3. The demo path is represented in code | 0:55 |
| 4. What the local numerical tests say | 1:00 |
| 5. Streaming evidence and its limits | 0:50 |
| 6. The next proof points | 0:55 |
| **Total** | **5:00** |

## Interactive slide: System diagram

“Board bring-up and initial hardware checks are complete. This diagram shows
the repository's planned data path, which has not yet been tested: Python on
the ARM processor prepares signed Q4 query, key, and value words in DDR and
configures AXI DMA. DMA streams the words into the programmable-logic
attention core. The core should return 16 values through the receive side of
DMA; Python then compares them with its NumPy reference.”

## Interactive slide: Streaming progress trace

“This animation replays one fixed-seed synthetic input through the same NumPy
online recurrence used as the reference. Each point is one key/value row: the
running maximum and denominator update, and the first normalized output
component settles as more rows arrive. The chart is a software-model trace,
not an RTL waveform or board result. The important architectural point is
that the core carries m, l, and the value accumulator forward instead of
storing a 196 by 196 score table.”

## Source data

- [`verification_report.md`](../../sim/results/verification_report.md)
- [`verification_report.json`](../../sim/results/verification_report.json)
- [`numeric_cases.csv`](../../sim/results/numeric_cases.csv)
- [`board/README.md`](../../board/README.md)
