# PYNQ-Z2 streaming-attention demo

This is the first FPGA-backed milestone toward the full ViT accelerator. It
computes **one attention head from fixed Q/K/V feature vectors** in the FPGA
programmable logic (PL), then compares the result with the NumPy reference.
It is not yet a camera/image-to-classification pipeline: the query, keys, and
values are deterministic synthetic inputs, and the Q/K/V projection layers
are not implemented in the PL.

## What is in this demo

- [`../rtl/streaming_attention_axis.sv`](../rtl/streaming_attention_axis.sv):
  fixed-shape AXI4-Stream engine; Q12 scores, a 64-entry fractional LUT,
  Q15 exponential weights, online max/sum recurrence, and signed Q4 output.
- [`../vivado/build_pynq_z2.tcl`](../vivado/build_pynq_z2.tcl): Vivado project
  and Zynq processing-system, AXI DMA, clock/reset, and bitstream build.
- [`../sim/tb_streaming_attention_axis.sv`](../sim/tb_streaming_attention_axis.sv):
  RTL smoke test using equal scores and known values.
- [`run_demo.py`](./run_demo.py): PYNQ DMA transfer, deterministic input
  generation, Python-reference comparison, and pass/fail report.
- [`test_demo.py`](./test_demo.py): host-side checks for test vector shapes,
  stream packing, and reference output.

The stream contains 9,440 32-bit words in this order:

1. 32 query values.
2. For each of 196 tokens: 32 key values followed by 16 value values.

Each word contains one signed int8 value sign-extended to 32 bits. Inputs use
Q4 scaling (integer value divided by 16); the output is a signed Q4 value in
each of 16 lanes. The hardware handles one stream transaction at a time.

## Build on the Linux workstation

1. Install a Vivado release that supports the PYNQ-Z2's XC7Z020 and install
   the Digilent PYNQ-Z2 board files. The build script targets Vivado 2023.2;
   board files must provide `digilentinc.com:pynq-z2:part0:1.0`.
2. From the repository root, run the RTL smoke test in the Vivado command
   shell:

   ```sh
   xvlog -sv rtl/streaming_attention_axis.sv sim/tb_streaming_attention_axis.sv
   xelab tb_streaming_attention_axis -s tb_attention
   xsim tb_attention -runall
   ```

   It should print `PASS` and exit successfully.
3. If simulation passes, build the bitstream from the repository root:

   ```sh
   vivado -mode batch -source vivado/build_pynq_z2.tcl
   ```

4. If the build succeeds, copy these generated files to the board:

   ```text
   build/overlay/vital_pynq_z2.bit
   build/overlay/vital_pynq_z2.hwh
   board/run_demo.py
   software/base2_softmax.py
   ```

   Keep the `.bit` and `.hwh` files together. For example, copy all four
   files to `~/vital-demo/` on the board; the runner finds the model beside
   itself when it is deployed this way.

## Run on the PYNQ-Z2

Use the official PYNQ-Z2 image and its Jupyter terminal or SSH shell. Confirm
that NumPy is available, then run:

```sh
cd ~/vital-demo
python3 run_demo.py --bitstream vital_pynq_z2.bit
```

The program sends the generated vector through AXI DMA, waits for all 16
results, compares them with `base2_softmax.py`, and prints `BOARD DEMO PASSED`
only if every output component is within 0.20 of the reference. Re-run with
the same seed to reproduce the same inputs; use `--seed` to choose another
case.

Run the host-side checks from the repository root before building:

```sh
python3 -m unittest discover -s board -p 'test_*.py'
python3 -m unittest discover -s sim -p 'test_*.py'
python3 sim/verify_demo.py
python3 software/base2_softmax.py --self-test
```

The expanded host verification writes a summary, per-case CSV, JSON metrics,
and presentation SVG to [`../sim/results/`](../sim/results/). It mirrors the
integer arithmetic in Python and does not replace the Vivado testbench or the
physical-board run.

## Important limits

This repository cannot run Vivado, load the PYNQ-Z2, or verify its AXI-DMA
behavior from the current development environment. The Tcl build, timing,
resource use, reset behavior, and on-board numeric tolerance must be verified
on the supported Linux/Vivado setup and physical board before calling this
demo working. A passing software test does not substitute for that test.

This is a hardware proof of the attention recurrence, not a complete ViT
accelerator or a performance result. It does not include image preprocessing,
the model's Q/K/V projection matrices, W4A8 projection MACs, camera input, or
claims of the 30 FPS / 5 W project targets.
