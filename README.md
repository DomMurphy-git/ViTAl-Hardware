# ViTAl-Hardware: Vision Transformer Acceleration Hardware
Packed INT4/INT8 Streaming Vision Transformer Attention Accelerator

[![Target Platform](https://img.shields.io/badge/Target-Xilinx%20Zynq--7000%20(XC7Z020)-orange.svg)](https://www.amd.com/en/products/adaptive-socs-and-fpgas/zynq/zynq-7000.html)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2F%20Vivado%20HLS%2FSV-red.svg)](https://pytorch.org/)

An heterogeneous hardware-software accelerator designed to execute Multi-Head Self-Attention (MHSA) for Vision Transformers (ViTs) on resource-constrained, low-power edge platforms (AMD Xilinx Zynq-7000 SoC / PYNQ-Z2).

---

## 🚁 Operational Scenario & Motivation

This project targets **untethered autonomous unmanned aerial vehicles (UAVs / sub-2 kg drones)** navigating cluttered, GPS-denied indoor and industrial spaces. Operating under a strict **$\le 5\text{ W}$ thermal envelope** and a **sub-33 ms ($\ge 30$ FPS) latency deadline**, the flight controller relies on immediate visual hazard maps to avoid high-speed collisions.

The failure cost is strictly asymmetric:
* **False Negative (Missed Hazard):** Dropping a subtle high-speed obstacle (e.g., thin wiring, rebar, or conduit) leads to a **catastrophic hull strike and mission failure**.
* **False Positive (Phantom Hazard):** Numerical noise flagging a non-existent barrier triggers a momentary, recoverable braking maneuver with minimal battery cost.

### Why Standard Edge Accelerators Fall Short
Vision Transformers capture global spatial contexts far better than traditional CNNs, but their attention mechanism ($QK^T \to \text{Softmax} \to V$) materializes a massive $O(N^2)$ score matrix. On small edge FPGAs, constantly writing and reading this intermediate table back to external DDR DRAM saturates the AXI bus, causing thermal throttling and breaking real-time constraints.

---

## ⚡ Core Architectural Innovations

1. **Asymmetric Sub-Byte Quantization ($W4A8$):** Projection weights are packed down to 4 bits (two weights per byte), cutting AXI bus transfer volume and memory energy by **$50\%$**, while activations remain at 8 bits to prevent low-magnitude hazard features from clipping to zero[cite: 1].
2. **Fused Streaming Base-2 Online Softmax:** Adopting principles inspired by *FlashAttention*, the datapath tracks running maximums and denominator sums incrementally across streaming tiles. The full $196 \times 196$ attention matrix is **never written to BRAM or external DRAM**[cite: 1].
3. **Logarithmic Base-2 Exponential Approximation:** Replaces expensive floating-point $e^x$ transcendental hardware with a zero-DSP barrel bit-shift for integer exponents and a compact 64-entry distributed LUT for fractions[cite: 1].

---

## 📊 12-Week Implementation Roadmap

| Sprint / Phase | Timeframe | Focus / Deliverables |
| :--- | :--- | :--- |
| **Sprint 1** | Weeks 1–2 | Board bring-up (PYNQ-Z2), FP32 PyTorch baseline, $W4A8$ quantization modeling, Base-2 online softmax Python simulation, and binary test vector generation. |
| **Sprint 2** | Weeks 3–7 | RTL SystemVerilog design: sub-byte unpacker, Xilinx DSP48E1 SIMD MAC array, and fused streaming normalizer. Standalone simulation in Vivado xsim/Verilator. |
| **Sprint 3** | Weeks 8–9 | SoC Block Design integration: Zynq ARM Processing System (PS), Xilinx AXI DMA controller, and custom Programmable Logic (PL) IP wrapper. Timing closure at $\ge 100\text{ MHz}$. |
| **Sprint 4** | Weeks 10–12 | Linux driver development (`/dev/uio` / CMA buffers), physical hardware benchmarking on PYNQ-Z2, cycle-accurate jitter profiling, and final demo packaging. |

---

## Repository Structure

```text
.
├── board/                    # PYNQ-Z2 runner, tests, and setup instructions
├── docs/                     # Proposal, sprint plan, and presentation assets
│   └── presentations/
├── rtl/                      # FPGA SystemVerilog
├── sim/                      # RTL testbench, host verification, and reports
│   └── results/
├── software/                 # NumPy/PyTorch reference models
└── vivado/                   # Vivado project build scripts
```

Project documents: [Sprint 1 plan](./docs/sprint1.md), [proposal PDF](./docs/Murphy_Edge_ViT_Proposal.pdf), and [presentation assets](./docs/presentations/).

### Streaming Base-2 Softmax Model

[`software/base2_softmax.py`](./software/base2_softmax.py) is a NumPy
reference model for the online running-maximum/running-sum softmax recurrence.
It uses fixed-point scores, a 64-entry fractional base-2 exponential LUT, Q15
weights, and integer shifts. Its `streaming_attention` function processes
key/value blocks without materializing the full query-by-key score matrix.

Run its deterministic checks with:

```sh
python3 software/base2_softmax.py --self-test
```

NumPy is required; if PyTorch is installed, the self-test also compares the
approximate probabilities and attention output with PyTorch Softmax. Since the
LUT and fixed-point conversions are approximations, parity is numerical within
the self-test tolerances rather than bit-identical to floating-point PyTorch.

### PYNQ-Z2 FPGA Demo

The first FPGA-backed attention demo is under [`board/`](./board/README.md).
It includes a fixed-shape AXI-stream attention engine, a Vivado build script,
and a PYNQ DMA runner that compares FPGA output to the NumPy reference. It is
an attention-kernel proof with synthetic Q/K/V vectors, not yet full image
inference or a measured performance result. The Vivado bitstream must be built
on a supported Linux workstation and validated on the physical board.

Host-side verification scripts and generated presentation results are
documented in [`sim/`](./sim/README.md). Their software mirror is not a
substitute for RTL simulation or FPGA testing.

The [`Sprint 2 progress deck`](./docs/presentations/Sprint2_Progress.pptx)
and [timed speaker script](./docs/presentations/Sprint2_Speaker_Script.md)
summarize the current implementation, software-only validation evidence,
and remaining hardware verification. The
[interactive Sprint 2 presentation](./docs/presentations/Sprint2_Interactive.html)
adds animated diagrams, clickable explanations, and slide-by-slide speaker
notes; open it in a browser and use the arrow keys or on-screen controls.
