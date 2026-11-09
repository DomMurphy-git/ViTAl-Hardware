# Local Demo Verification Report

**Run time (UTC):** 2026-11-09T01:01:07.955228+00:00
**Overall software-mirror result:** PASS

> This is host-side verification of a Python model mirroring the RTL arithmetic and expected stream schedule. No SystemVerilog compiler, Vivado build, or physical board was run. It is not proof that RTL compiles or that the overlay works on the PYNQ-Z2.

## Results

- **Numeric inputs:** 132 total (4 edge cases, 128 seeded random cases).
- **Pass criterion:** maximum output difference from the Python reference <= 0.20 dequantized value units.
- **Worst-case difference:** 0.031239 (random_seed_0023).
- **All cases passed:** True.
- **RTL LUT:** 64 entries; all match Python.
- **Exponential arithmetic:** 65,536 Q12 deltas match the Python implementation and are monotonic.
- **RTL-width range checks:** True.
- **AXI schedule model:** 20 randomized-stall trials, 188,800 input words and 320 output words transferred.
- **Score matrix:** no N-by-N score buffer is declared; the modeled RTL stores only the query, current key/value, recurrence state, and output accumulators.

## Fixed-point range observations

- Largest absolute Q12 score: 1,471,372.
- Largest absolute accumulator: 89,915,392 (48-bit signed RTL register).
- Largest Q15 denominator: 6,422,528 (32-bit unsigned RTL register).

- Largest absolute difference from PyTorch: 0.031375 (random_seed_0072).

## Evidence boundary / next hardware checks

- The host numeric tests model RTL integer arithmetic in Python; they do not execute the SystemVerilog.
- Compile and run `xvlog -sv rtl/streaming_attention_axis.sv sim/tb_streaming_attention_axis.sv`, `xelab tb_streaming_attention_axis -s tb_attention`, and `xsim tb_attention -runall` in Vivado.
- Then build `vivado/build_pynq_z2.tcl`, load the generated overlay on the board, and run `board/run_demo.py`.
- Timing, resource use, power, and image-level model accuracy are not measured by this report.

## Artifacts

- `verification_report.json`: machine-readable summary.
- `numeric_cases.csv`: per-case data for charts and inspection.
- `verification_report.md`: this presentation-friendly summary.
- `verification_summary.svg`: a 16:9 presentation graphic.
