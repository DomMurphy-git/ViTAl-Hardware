# Local demo verification

Run the host-side checks from the repository root:

```sh
python3 -m unittest discover -s board -p 'test_*.py' -v
python3 -m unittest discover -s sim -p 'test_*.py' -v
python3 sim/verify_demo.py
```

`verify_demo.py` runs four deterministic edge cases plus 128 seeded Q4 test
cases through an integer-arithmetic mirror of the current RTL. It also checks
that all 64 RTL LUT constants match Python, compares the RTL exponential
mapping with the Python implementation across 65,536 Q12 deltas, checks
accumulator/denominator values against their declared register widths, and
stresses the expected ready/valid transaction schedule under random source
gaps and output back-pressure.

The script writes presentation artifacts to `sim/results/`:

- `verification_report.md`: concise summary and an explicit statement of what
  was and was not tested.
- `verification_summary.svg`: 16:9 presentation graphic.
- `verification_report.json`: machine-readable run summary.
- `numeric_cases.csv`: per-case output errors and fixed-point range data.

This is **host software verification, not HDL simulation**. The integer model
is a second implementation of the RTL arithmetic; source-level LUT and
interface checks do not prove that the RTL compiles or that Vivado connects
the overlay as intended. This environment has no Vivado, `xsim`, Icarus, or
Verilator. Use the actual Vivado simulation commands in
[`../board/README.md`](../board/README.md), then run the overlay on the
physical board before claiming FPGA behavior has passed.
