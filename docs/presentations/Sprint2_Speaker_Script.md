# Sprint 2 Progress — Five-Minute Speaker Script

**Deck:** [Sprint2_Progress.pptx](./Sprint2_Progress.pptx)
**Interactive version:** [Sprint2_Interactive.html](./Sprint2_Interactive.html)
**Length:** 5:00 at a clear presentation pace
**Status snapshot:** 8 November 2026

> Accuracy note: the numbers below come from host-side Python tests. The
> SystemVerilog has not yet been compiled or run in an HDL simulator, a
> Vivado bitstream has not been built, and the board has not yet run the demo.
> The PYNQ-Z2 is now secured; physical bring-up is the next proof point.

For the interactive version, navigate with the arrow keys or bottom buttons.
Click pipeline stages and milestones to expand explanations; use **Speaker
notes** for the narration, **Pause motion** to stop animations, and
**Fullscreen** to present.

## Slide 1 — Sprint 2: from model to board demo (0:30)

“This Sprint 2 update is about moving from an algorithm we can test in Python
toward a small attention kernel for the PYNQ-Z2. The board is now secured.
Today’s passing results are still host-side: the SystemVerilog has not been
simulated and no bitstream has run on the board. I’ll cover what is
implemented, what the numbers show, and the next validation steps.”

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

“The repository now has four pieces for that path. First is an AXI-stream
SystemVerilog module for the fixed-size operation; second, a Vivado Tcl recipe
for the Zynq processing system and AXI DMA; third, a PYNQ runner that sends
deterministic inputs and compares output with the Python reference; and
fourth, a host verification harness and RTL testbench.

They are implementation artifacts, not proof the hardware works. The board
still needs to boot, the RTL needs to compile and simulate, and the overlay
and DMA runner need an end-to-end board test.”

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
There is no Vivado timing/resource report, power measurement, or board DMA
result. The software mirror passes; FPGA behavior remains unverified.”

## Slide 6 — The next proof points (0:55)

“Now that the board is secured, the next steps are concrete. First, bring up
the PYNQ image and verify serial and network access. Then run the HDL
testbench in Vivado simulation, fix any issues, and build the overlay. Finally,
run the same seeded test through DMA and compare actual board output with
Python.

The proof we need is a build log, passing simulation, and board comparison.
Only then can we claim the kernel works in programmable logic. Next we can
add projection MACs, real model inputs, and measure latency and power. This
is a step toward the accelerator, not the finished system.”

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

## Source data

- [`verification_report.md`](../../sim/results/verification_report.md)
- [`verification_report.json`](../../sim/results/verification_report.json)
- [`numeric_cases.csv`](../../sim/results/numeric_cases.csv)
- [`board/README.md`](../../board/README.md)
