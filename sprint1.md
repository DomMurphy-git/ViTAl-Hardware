# Sprint 1 — Software Baseline and Feasibility

**Status snapshot:** 14 October 2026. Sprint 1 is the first two-week phase of
the 12-week project. This document records what is demonstrated today and what
still needs evidence; an open GitHub issue is not evidence that its work is
complete.

## 1. Mission

For Dominic Murphy, the solo ECE engineer building onboard perception for a
small UAV, ViTAl-Hardware is a Zynq-based streaming Vision Transformer
attention accelerator intended to process image-patch attention within a
sub-33 ms deadline and 5 W envelope without materializing the full score
matrix.

This is the best current mission hypothesis, not a final claim; revise it after
the first end-to-end test.

## 2. Target user

**Primary user:** Dominic Murphy, the solo project engineer responsible for
getting Vision Transformer attention to run on the project’s edge FPGA. Build
the first workflow for this engineer: a reproducible software reference,
quantization checks, and RTL-ready test vectors. The UAV operator is the
downstream beneficiary, not the first user of the development toolchain.

## 3. User stories

These five stories are already tracked as open Sprint 1 issues on the
[GitHub issue board](https://github.com/DomMurphy-git/ViTAl-Hardware/issues).
The acceptance criteria below make the intended evidence explicit; they do
not imply that the open issues have passed.

| # | User story and board issue | Acceptance criteria | Current evidence/status |
|---|---|---|---|
| 1 | **As the project engineer, I want a reproducible host toolchain and a reachable PYNQ-Z2, so I can run the software model now and integrate the FPGA later.** [Issue #1](https://github.com/DomMurphy-git/ViTAl-Hardware/issues/1) | A documented, pinned host environment runs the model and reference tests; Vivado opens with the XC7Z020 target; the board boots PYNQ and its serial console and SSH/Jupyter access are demonstrated. | **Open.** Python, NumPy, and PyTorch have run locally. The board is still reported as pending procurement/bring-up; Vivado and board access have not been demonstrated. A committed dependency manifest is still needed. |
| 2 | **As the project engineer, I want a PyTorch FP32 attention baseline on a fixed patch workload, so quantized results have a trustworthy reference.** [Issue #3](https://github.com/DomMurphy-git/ViTAl-Hardware/issues/3) | Commit the chosen model, dataset or fixture, and runnable baseline script; run the specified sample; record Q/K/V and attention-weight ranges; save a reproducible summary of the results. | **Open.** No FP32 baseline script, selected dataset/model, or activation report is currently committed. Softmax parity tests use seeded synthetic arrays, not images. |
| 3 | **As the project engineer, I want W4A8 quantization compared with FP32, so I can reduce weight traffic without silently losing important activation features.** [Issue #4](https://github.com/DomMurphy-git/ViTAl-Hardware/issues/4) | Commit the W4A8 implementation and run it on the same workload as the FP32 baseline; report weight-size reduction and the agreed task metric; show the measured degradation meets the issue’s less-than-1% target. | **Open.** No W4A8 implementation or same-workload comparison is currently committed. The less-than-1% target is not yet measured. |
| 4 | **As the project engineer, I want a streaming Base-2 online softmax model, so I can validate the recurrence and exponential approximation before writing RTL.** [Issue #5](https://github.com/DomMurphy-git/ViTAl-Hardware/issues/5) | The 64-entry LUT has 64 ordered entries; the exponential mapping is monotonic and within its stated error bound; output probabilities sum to one; seeded tests compare against PyTorch within explicit tolerances; streaming attention matches the reference within tolerance without building the full score matrix. | **Model and self-test pass; issue remains open.** [`base2_softmax.py`](./base2_softmax.py) passes its LUT, recurrence, normalization, PyTorch, and streaming checks. The value accumulator is float64, so this is not yet a fully bit-accurate simulation of every hardware datapath operation. |
| 5 | **As the project engineer, I want binary golden vectors for the RTL testbench, so I can check future hardware results automatically against known outputs.** [Issue #6](https://github.com/DomMurphy-git/ViTAl-Hardware/issues/6) | Commit the chosen input fixture, packed W4 weights, expected outputs, and a loader; the loader round-trips each file and the expected outputs reproduce the software reference within the established tolerance. | **Open.** Binary vectors and their loader are not currently committed. The prerequisite dataset/workload and W4A8 pipeline are not yet fixed. |

The linked issues are the board’s plan. Keep status and execution checklists
there; this document describes the user outcomes and the evidence required to
accept them. The other open Sprint 1 issues remain on the board but are outside
these five top implementation stories.

## 4. Feasibility — evidence and gaps

### Demonstrated on the host

- [`base2_softmax.py`](./base2_softmax.py) is committed and runs on the host
  CPU. The deterministic check is:

  ```sh
  python3 base2_softmax.py --self-test
  ```

- On the seeded test cases run on 14 October 2026, the largest observed
  absolute difference from PyTorch was **0.00229** for softmax probabilities
  and **0.00116** for the streaming attention output. Both comparisons passed
  the test’s configured relative-plus-absolute tolerances. These are results
  from the included synthetic tests, not a real image validation set.
- The streaming attention path generates each query’s score values in tiles
  and updates its running state; it does not allocate an N-by-N score matrix.
  With 196 keys, the outputs for tile sizes 13 and 64 were identical.
- Python syntax compilation and `git diff --check` also passed.

### Not demonstrated yet

- **Dataset/model:** No image dataset or pretrained ViT model is selected,
  downloaded, loaded by a baseline script, or committed. The current softmax
  tests generate synthetic arrays from a fixed random seed. Selecting and
  checking in a reproducible fixture/workload is still required before making
  claims about image accuracy or hazard recall.
- **Quantization:** There is no committed W4A8 implementation or measured
  comparison to the FP32 baseline. In particular, the claimed less-than-1%
  recall change has not been tested in this repository.
- **Hardware and FPGA tools:** PYNQ-Z2 procurement/bring-up is still reported
  as pending. No board boot, serial console, SSH/Jupyter session, Vivado
  synthesis, RTL simulation, timing, power, or on-device result is claimed.
- **Dependencies:** NumPy and PyTorch were installed in the local development
  environment for the tests, but the repository does not yet pin them in a
  committed `requirements.txt` or equivalent environment file.
- **External API keys:** The current host model and comparisons use no external
  APIs or API keys. This does not prove access to any future external service.

So far, the recurrence and the software-side approximation are feasible to
exercise without the board. Dataset-level accuracy, quantization quality, and
hardware feasibility remain unproven and are the main Sprint 1 risks.

## 5. Tooling

- **Python 3.9:** host-side algorithm model, baseline scripts, and vector tools.
- **NumPy:** fixed-point/LUT model and deterministic synthetic test generation.
- **PyTorch:** floating-point reference for numerical comparisons; it is not
  the hardware implementation.
- **SystemVerilog + AMD Vivado ML Standard / xsim:** planned RTL implementation
  and simulation; installation/access is not yet verified.
- **TUL PYNQ-Z2 (XC7Z020):** planned target board; physical access is pending.
- **GitHub Issues:** Sprint 1 work tracking; issues #1, #3, #4, #5, and #6 are
  the five stories listed above.
- **Obsidian:** literature synthesis and project research notes.

## 6. Demo sentence

At the end of two weeks we will show a reproducible, CPU-only pipeline that
loads a committed patch workload, compares FP32 attention with W4A8 plus
streaming Base-2 attention, and exports binary golden vectors with measured
accuracy deltas; FPGA execution will be shown only if board and tool access are
secured.
