#!/usr/bin/env python3
"""Verify the board demo's fixed-point math and emit presentation-ready results.

This is a software mirror of the RTL arithmetic and stream protocol. It does
not compile, simulate, or prove the SystemVerilog itself; run the Vivado
testbench and the physical-board test separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "software"))
sys.path.insert(0, str(REPO_ROOT / "board"))

import base2_softmax as reference
from run_demo import (
    DEFAULT_TOLERANCE,
    INPUT_WORDS,
    QUERY_DIM,
    Q4_SCALE,
    TOKEN_COUNT,
    VALUE_DIM,
    make_test_case,
    pack_input_words,
    reference_output,
)

SEED_COUNT = 128
REPORT_DIR = Path(__file__).with_name("results")
RTL_PATH = REPO_ROOT / "rtl" / "streaming_attention_axis.sv"
TCL_PATH = REPO_ROOT / "vivado" / "build_pynq_z2.tcl"


def rtl_exp_neg_q15(delta_q12: int) -> int:
    """Mirror the RTL unsigned Q12-to-Q15 exponential lookup arithmetic."""
    product = delta_q12 * reference.LOG2E_Q24
    exponent_units = product >> 30
    integer_exponent = exponent_units >> reference.LUT_FRACTIONAL_BITS
    fraction = exponent_units & 63
    if fraction == 0:
        lut_index = 0
        shift = integer_exponent
    else:
        lut_index = 64 - fraction
        shift = integer_exponent + 1
    if shift >= 17:
        return 0
    return reference.FRACTIONAL_LUT_Q15[lut_index] >> shift


def score_q12_from_int8(query: np.ndarray, key: np.ndarray) -> int:
    """Mirror RTL dot-product, 1/sqrt(32) scale, and signed rounding."""
    dot = sum(int(q) * int(k) for q, k in zip(query, key))
    product = dot * 46341
    if product >= 0:
        return (product + 8192) // 16384
    return -((-product + 8192) // 16384)


def rtl_integer_attention(
    query: np.ndarray, keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Mirror the RTL's Q12 score, Q15 recurrence, and rounded Q4 output."""
    maximum: int | None = None
    denominator = 0
    accumulators = [0] * VALUE_DIM
    largest_abs_score = 0
    largest_abs_accumulator = 0
    largest_denominator = 0

    for token_index in range(TOKEN_COUNT):
        score = score_q12_from_int8(query, keys[token_index])
        largest_abs_score = max(largest_abs_score, abs(score))

        if maximum is None:
            maximum = score
            denominator = reference.WEIGHT_ONE
            accumulators = [
                int(value) * reference.WEIGHT_ONE
                for value in values[token_index]
            ]
        else:
            new_maximum = max(maximum, score)
            alpha = rtl_exp_neg_q15(new_maximum - maximum)
            beta = rtl_exp_neg_q15(new_maximum - score)
            denominator = ((denominator * alpha) >> 15) + beta
            accumulators = [
                ((accumulator * alpha) >> 15) + int(value) * beta
                for accumulator, value in zip(accumulators, values[token_index])
            ]
            maximum = new_maximum

        largest_abs_accumulator = max(
            largest_abs_accumulator,
            *(abs(accumulator) for accumulator in accumulators),
        )
        largest_denominator = max(largest_denominator, denominator)

    rounded_q4 = []
    for accumulator in accumulators:
        magnitude = (abs(accumulator) + denominator // 2) // denominator
        signed = -magnitude if accumulator < 0 else magnitude
        rounded_q4.append(max(-128, min(127, signed)))

    diagnostics = {
        "max_abs_score_q12": largest_abs_score,
        "max_abs_accumulator": largest_abs_accumulator,
        "max_denominator_q15": largest_denominator,
    }
    return np.asarray(rounded_q4, dtype=np.int16), diagnostics


def build_cases(seed_count: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    cases = []

    query = np.zeros(QUERY_DIM, dtype=np.int8)
    keys = np.zeros((TOKEN_COUNT, QUERY_DIM), dtype=np.int8)
    values = np.zeros((TOKEN_COUNT, VALUE_DIM), dtype=np.int8)
    cases.append(("all_zero", query, keys, values))

    rng = np.random.default_rng(815)
    equal_values = rng.integers(-127, 128, size=(TOKEN_COUNT, VALUE_DIM), dtype=np.int8)
    cases.append(
        (
            "equal_scores",
            np.zeros(QUERY_DIM, dtype=np.int8),
            rng.integers(-128, 128, size=(TOKEN_COUNT, QUERY_DIM), dtype=np.int8),
            equal_values,
        )
    )

    dominant_query = np.full(QUERY_DIM, 127, dtype=np.int8)
    dominant_keys = -dominant_query[None, :] * np.ones(
        (TOKEN_COUNT, QUERY_DIM), dtype=np.int16
    )
    dominant_keys[0, :] = dominant_query
    dominant_values = rng.integers(-128, 128, size=(TOKEN_COUNT, VALUE_DIM), dtype=np.int8)
    cases.append(
        (
            "dominant_first_token",
            dominant_query,
            dominant_keys.astype(np.int8),
            dominant_values,
        )
    )

    alternating = np.where(np.arange(QUERY_DIM) % 2, 127, -128).astype(np.int8)
    cases.append(
        (
            "extreme_alternating",
            alternating,
            np.tile(alternating, (TOKEN_COUNT, 1)),
            rng.integers(-128, 128, size=(TOKEN_COUNT, VALUE_DIM), dtype=np.int8),
        )
    )

    for seed in range(seed_count):
        q, k, v = make_test_case(seed)
        cases.append((f"random_seed_{seed:04d}", q, k, v))
    return cases


def check_lut_and_exp() -> dict[str, Any]:
    rtl_entries = {
        int(index): int(value)
        for index, value in re.findall(
            r"7'd(\d+): lut_value = 17'd(\d+);", RTL_PATH.read_text()
        )
    }
    if len(rtl_entries) != 64:
        raise AssertionError(f"RTL LUT has {len(rtl_entries)} entries, expected 64")
    if [rtl_entries[i] for i in range(64)] != list(reference.FRACTIONAL_LUT_Q15):
        raise AssertionError("RTL and Python fractional LUT entries differ")

    exp_deltas = range(1 << 16)
    max_exp_error = 0.0
    previous = reference.WEIGHT_ONE + 1
    for delta in exp_deltas:
        rtl_value = rtl_exp_neg_q15(delta)
        python_value = reference.exp_neg_q15(delta, 12)
        if rtl_value != python_value:
            raise AssertionError(
                f"RTL/Python exponential mismatch at delta {delta}: "
                f"{rtl_value} != {python_value}"
            )
        if rtl_value > previous:
            raise AssertionError(f"exponential is not monotonic at delta {delta}")
        previous = rtl_value
        exact = math.exp(-delta / (1 << 12))
        max_exp_error = max(max_exp_error, abs(rtl_value / reference.WEIGHT_ONE - exact))

    return {
        "rtl_lut_entries": len(rtl_entries),
        "rtl_lut_matches_python": True,
        "exp_input_deltas_checked": 1 << 16,
        "rtl_exp_matches_python_all_checked_deltas": True,
        "exp_monotonic": True,
        "max_abs_exp_error": max_exp_error,
    }


def check_interface_and_memory_shape() -> dict[str, Any]:
    rtl = RTL_PATH.read_text()
    required_ports = (
        "s_axis_tvalid",
        "s_axis_tready",
        "s_axis_tdata",
        "s_axis_tlast",
        "m_axis_tvalid",
        "m_axis_tready",
        "m_axis_tdata",
        "m_axis_tlast",
    )
    missing = [port for port in required_ports if port not in rtl]
    if missing:
        raise AssertionError(f"RTL missing stream signals: {missing}")
    if re.search(r"\[\s*0\s*:\s*TOKEN_COUNT\s*-\s*1\s*\].*score", rtl, re.I):
        raise AssertionError("RTL declares a token-count-sized score array")

    required_tcl = (
        "CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}",
        "axi_memory_ic/S01_AXI",
        "processing_system7_0/S_AXI_HP0",
    )
    tcl = TCL_PATH.read_text()
    if any(fragment not in tcl for fragment in required_tcl):
        raise AssertionError("Vivado AXI DMA memory interconnect wiring is incomplete")

    return {
        "axis_stream_signals_present": list(required_ports),
        "token_score_matrix_declared": False,
        "query_bytes_retained": QUERY_DIM,
        "one_key_vector_retained": QUERY_DIM,
        "one_value_vector_retained": VALUE_DIM,
        "weighted_output_accumulators_retained": VALUE_DIM,
        "input_words_per_transaction": INPUT_WORDS,
        "tlast_is_used_for_input_framing": False,
        "dma_memory_interconnect_has_read_and_write_ports": True,
    }


def run_protocol_stress(seed: int, trials: int = 20) -> dict[str, Any]:
    """Exercise a cycle-level ready/valid schedule with randomized stalls.

    This tests the planned transaction schedule, not the SystemVerilog RTL.
    """
    rng = np.random.default_rng(seed)
    max_cycles = 0
    input_transfers = 0
    output_transfers = 0
    total_stall_cycles = 0

    for _ in range(trials):
        phase = "query"
        word_index = 0
        token = 0
        output_index = 0
        input_valid = False
        input_word = None
        output_valid = False
        output_word = None
        cycles = 0
        input_stalls = 0
        output_stalls = 0

        while output_index < VALUE_DIM and cycles < 200_000:
            cycles += 1
            # Source keeps TVALID and TDATA unchanged until a transfer.
            if not input_valid and phase in ("query", "key", "value"):
                input_valid = bool(rng.random() < 0.82)
                input_word = word_index
            # Output producer holds the current word through back-pressure.
            if phase == "output" and not output_valid:
                output_valid = True
                output_word = output_index
            output_ready = bool(rng.random() < 0.71)

            input_ready = phase in ("query", "key", "value")
            if input_valid and not input_ready:
                input_stalls += 1
            if output_valid and not output_ready:
                output_stalls += 1

            if input_valid and input_ready:
                if input_word != word_index:
                    raise AssertionError("input data changed before handshake")
                input_transfers += 1
                input_valid = False
                word_index += 1
                if phase == "query" and word_index == QUERY_DIM:
                    phase = "key"
                    word_index = 0
                elif phase == "key" and word_index == QUERY_DIM:
                    phase = "value"
                    word_index = 0
                elif phase == "value" and word_index == VALUE_DIM:
                    phase = "update"
                    word_index = 0
            elif phase == "update":
                token += 1
                phase = "key" if token < TOKEN_COUNT else "output"
            if output_valid and output_ready:
                if output_word != output_index:
                    raise AssertionError("output data changed before handshake")
                output_transfers += 1
                output_index += 1
                output_valid = False
            if phase == "output" and output_index == VALUE_DIM:
                break

        if cycles >= 200_000:
            raise AssertionError("ready/valid schedule failed to complete")
        max_cycles = max(max_cycles, cycles)
        total_stall_cycles += input_stalls + output_stalls

    expected_inputs = trials * INPUT_WORDS
    expected_outputs = trials * VALUE_DIM
    if input_transfers != expected_inputs:
        raise AssertionError(f"sent {input_transfers} words, expected {expected_inputs}")
    if output_transfers != expected_outputs:
        raise AssertionError(
            f"received {output_transfers} outputs, expected {expected_outputs}"
        )
    return {
        "protocol_trials": trials,
        "protocol_schedule_model": "cycle-level ready/valid with randomized source gaps and output back-pressure",
        "input_words_transferred": input_transfers,
        "expected_input_words": expected_inputs,
        "output_words_transferred": output_transfers,
        "expected_output_words": expected_outputs,
        "maximum_cycles_for_one_trial": max_cycles,
        "input_and_output_stall_cycles": total_stall_cycles,
        "scope_note": "schedule model only; not an RTL simulator",
    }


def run_numeric_cases(seed_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    diagnostics_max = {
        "max_abs_score_q12": 0,
        "max_abs_accumulator": 0,
        "max_denominator_q15": 0,
    }
    all_fit = True
    max_abs_q4_error = 0.0
    max_error_case = ""
    max_abs_reference_error = 0.0
    max_reference_error_case = ""

    try:
        import torch
    except ImportError:
        torch = None

    for name, query, keys, values in build_cases(seed_count):
        if not (query.dtype == keys.dtype == values.dtype == np.int8):
            raise AssertionError(f"{name}: inputs are not int8")
        if query.shape != (QUERY_DIM,) or keys.shape != (TOKEN_COUNT, QUERY_DIM):
            raise AssertionError(f"{name}: query/key shape mismatch")
        if values.shape != (TOKEN_COUNT, VALUE_DIM):
            raise AssertionError(f"{name}: value shape mismatch")

        words = pack_input_words(query, keys, values)
        if words.size != INPUT_WORDS:
            raise AssertionError(f"{name}: wrong stream word count")

        hw_q4, diag = rtl_integer_attention(query, keys, values)
        hw_output = hw_q4.astype(np.float64) / Q4_SCALE
        py_reference = reference_output(query, keys, values)
        error = float(np.max(np.abs(hw_output - py_reference)))
        if error > max_abs_q4_error:
            max_abs_q4_error = error
            max_error_case = name
        if not np.all(np.isfinite(hw_output)):
            raise AssertionError(f"{name}: hardware-mirror output is non-finite")
        if np.any(hw_q4 < -128) or np.any(hw_q4 > 127):
            raise AssertionError(f"{name}: output is outside signed int8 range")

        for metric, value in diag.items():
            diagnostics_max[metric] = max(diagnostics_max[metric], value)
        all_fit &= diag["max_abs_score_q12"] < (1 << 31)
        all_fit &= diag["max_abs_accumulator"] < (1 << 47)
        all_fit &= diag["max_denominator_q15"] < (1 << 32)

        record: dict[str, Any] = {
            "case": name,
            "input_words": int(words.size),
            "max_abs_output_error_vs_python": error,
            "pass_at_board_tolerance_0_20": error <= DEFAULT_TOLERANCE,
            "max_abs_score_q12": diag["max_abs_score_q12"],
            "max_abs_accumulator": diag["max_abs_accumulator"],
            "max_denominator_q15": diag["max_denominator_q15"],
        }

        if torch is not None:
            torch_q = torch.tensor(query.astype(np.float64) / Q4_SCALE)
            torch_k = torch.tensor(keys.astype(np.float64) / Q4_SCALE)
            torch_v = torch.tensor(values.astype(np.float64) / Q4_SCALE)
            weights = torch.softmax(
                (torch_k @ torch_q) / math.sqrt(QUERY_DIM), dim=0
            )
            torch_output = (weights @ torch_v).numpy()
            torch_error = float(np.max(np.abs(hw_output - torch_output)))
            record["max_abs_output_error_vs_torch"] = torch_error
            if torch_error > max_abs_reference_error:
                max_abs_reference_error = torch_error
                max_reference_error_case = name
        results.append(record)

    failures = [row["case"] for row in results if not row["pass_at_board_tolerance_0_20"]]
    if failures:
        raise AssertionError(
            f"{len(failures)} numeric cases exceeded the demo tolerance: {failures[:5]}"
        )
    if not all_fit:
        raise AssertionError("fixed-point state exceeded a declared RTL register width")

    summary = {
        "numeric_cases": len(results),
        "deterministic_edge_cases": 4,
        "random_seeds": seed_count,
        "all_numeric_cases_within_board_tolerance": True,
        "board_tolerance_dequantized_units": DEFAULT_TOLERANCE,
        "max_abs_output_error_vs_python": max_abs_q4_error,
        "worst_python_comparison_case": max_error_case,
        "fixed_width_ranges_fit": True,
        **diagnostics_max,
    }
    if torch is not None:
        summary["pytorch_available"] = True
        summary["max_abs_output_error_vs_torch"] = max_abs_reference_error
        summary["worst_torch_comparison_case"] = max_reference_error_case
    else:
        summary["pytorch_available"] = False
    return results, summary


def detect_hdl_tools() -> dict[str, Any]:
    tools = {}
    for name in ("vivado", "xvlog", "xelab", "xsim", "iverilog", "verilator"):
        tools[name] = shutil.which(name)
    return {
        "tools": tools,
        "hdl_simulation_run": False,
        "reason": (
            "No HDL simulator is available in this environment."
            if not any(tools[name] for name in ("xsim", "iverilog", "verilator"))
            else "Simulator detected; this script reports availability but does not invoke it."
        ),
    }


def write_reports(
    report_dir: Path,
    cases: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "verification_report.json"
    csv_path = report_dir / "numeric_cases.csv"
    markdown_path = report_dir / "verification_report.md"
    svg_path = report_dir / "verification_summary.svg"
    report["artifacts"] = {
        "json": report_path.name,
        "csv": csv_path.name,
        "markdown": markdown_path.name,
        "svg": svg_path.name,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    fieldnames = list(cases[0]) if cases else []
    with csv_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)

    numeric = report["numeric"]
    lines = [
        "# Local Demo Verification Report",
        "",
        f"**Run time (UTC):** {report['run_time_utc']}",
        "**Overall software-mirror result:** PASS",
        "",
        "> This is host-side verification of a Python model mirroring the RTL "
        "arithmetic and expected stream schedule. No SystemVerilog compiler, "
        "Vivado build, or physical board was run. It is not proof that RTL "
        "compiles or that the overlay works on the PYNQ-Z2.",
        "",
        "## Results",
        "",
        f"- **Numeric inputs:** {numeric['numeric_cases']} total "
        f"({numeric['deterministic_edge_cases']} edge cases, "
        f"{numeric['random_seeds']} seeded random cases).",
        f"- **Pass criterion:** maximum output difference from the Python "
        f"reference <= {numeric['board_tolerance_dequantized_units']:.2f} "
        "dequantized value units.",
        f"- **Worst-case difference:** "
        f"{numeric['max_abs_output_error_vs_python']:.6f} "
        f"({numeric['worst_python_comparison_case']}).",
        f"- **All cases passed:** "
        f"{numeric['all_numeric_cases_within_board_tolerance']}.",
        f"- **RTL LUT:** {report['lut_and_exponential']['rtl_lut_entries']} entries; "
        f"all match Python.",
        f"- **Exponential arithmetic:** "
        f"{report['lut_and_exponential']['exp_input_deltas_checked']:,} Q12 deltas "
        "match the Python implementation and are monotonic.",
        f"- **RTL-width range checks:** {numeric['fixed_width_ranges_fit']}.",
        f"- **AXI schedule model:** "
        f"{report['protocol']['protocol_trials']} randomized-stall trials, "
        f"{report['protocol']['input_words_transferred']:,} input words and "
        f"{report['protocol']['output_words_transferred']:,} output words transferred.",
        "- **Score matrix:** no N-by-N score buffer is declared; the modeled "
        "RTL stores only the query, current key/value, recurrence state, and output accumulators.",
        "",
        "## Fixed-point range observations",
        "",
        f"- Largest absolute Q12 score: "
        f"{numeric['max_abs_score_q12']:,}.",
        f"- Largest absolute accumulator: "
        f"{numeric['max_abs_accumulator']:,} (48-bit signed RTL register).",
        f"- Largest Q15 denominator: "
        f"{numeric['max_denominator_q15']:,} (32-bit unsigned RTL register).",
        "",
    ]
    if numeric["pytorch_available"]:
        lines.extend(
            [
                f"- Largest absolute difference from PyTorch: "
                f"{numeric['max_abs_output_error_vs_torch']:.6f} "
                f"({numeric['worst_torch_comparison_case']}).",
                "",
            ]
        )
    else:
        lines.extend(["- PyTorch comparison: unavailable.", ""])
    lines.extend(
        [
            "## Evidence boundary / next hardware checks",
            "",
            "- The host numeric tests model RTL integer arithmetic in Python; "
            "they do not execute the SystemVerilog.",
            "- Compile and run "
            "`xvlog -sv rtl/streaming_attention_axis.sv "
            "sim/tb_streaming_attention_axis.sv`, "
            "`xelab tb_streaming_attention_axis -s tb_attention`, and "
            "`xsim tb_attention -runall` in Vivado.",
            "- Then build `vivado/build_pynq_z2.tcl`, load the generated overlay "
            "on the board, and run `board/run_demo.py`.",
            "- Timing, resource use, power, and image-level model accuracy are "
            "not measured by this report.",
            "",
            "## Artifacts",
            "",
            "- `verification_report.json`: machine-readable summary.",
            "- `numeric_cases.csv`: per-case data for charts and inspection.",
            "- `verification_report.md`: this presentation-friendly summary.",
            "- `verification_summary.svg`: a 16:9 presentation graphic.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines))
    python_error = numeric["max_abs_output_error_vs_python"]
    tolerance = numeric["board_tolerance_dequantized_units"]
    python_bar = min(660.0, 660.0 * python_error / tolerance)
    if numeric["pytorch_available"]:
        torch_error = numeric["max_abs_output_error_vs_torch"]
        torch_metric = f"{torch_error:.5f}"
        torch_label = "MAX ABS ERROR VS PYTORCH"
        torch_bar = min(570.0, 570.0 * torch_error / tolerance)
    else:
        torch_error = None
        torch_metric = "NOT RUN"
        torch_label = "PYTORCH REFERENCE"
        torch_bar = 0.0
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc">
  <title id="title">Local streaming-attention software verification</title>
  <desc id="desc">Passed host software tests: {numeric["numeric_cases"]} quantized cases, 64 matching LUT entries, and {report["lut_and_exponential"]["exp_input_deltas_checked"]:,} exponential deltas checked. No HDL simulator or board run was available.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#0a1422"/><stop offset="1" stop-color="#101e30"/></linearGradient>
    <linearGradient id="green" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#35d4a0"/><stop offset="1" stop-color="#20b67b"/></linearGradient>
    <pattern id="grid" width="36" height="36" patternUnits="userSpaceOnUse"><path d="M36 0H0V36" fill="none" stroke="#9bb2cd" stroke-opacity=".045"/></pattern>
    <style>text{{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}}.w{{fill:#f2f7fc}}.m{{fill:#91a4bc}}.mono{{font-family:"SFMono-Regular",Consolas,monospace}}</style>
  </defs>
  <rect width="1600" height="900" fill="url(#bg)"/><rect width="1600" height="900" fill="url(#grid)"/>
  <text x="84" y="76" fill="#72aaff" font-size="17" font-weight="700" letter-spacing="3">VITAL HARDWARE / LOCAL TEST REPORT</text>
  <text x="84" y="139" class="w" font-size="40" font-weight="750">STREAMING ATTENTION · SOFTWARE CHECKS</text>
  <rect x="1240" y="55" width="276" height="94" rx="20" fill="#12352f" stroke="#2ad49b"/>
  <circle cx="1282" cy="102" r="18" fill="url(#green)"/><path d="m1273 102 6 6 12-14" fill="none" stroke="#08251f" stroke-width="4" stroke-linecap="round"/>
  <text x="1315" y="98" fill="#67e4b7" font-size="14" font-weight="700" letter-spacing="2">HOST MODEL</text><text x="1315" y="126" class="w" font-size="22" font-weight="800">CHECKS PASS</text>
  <text x="84" y="206" class="m" font-size="19">Q12 scores · Q15 exponential · Q4 int8 inputs/outputs · {TOKEN_COUNT} keys per test</text>
  <g>
    <rect x="84" y="244" width="1432" height="227" rx="24" fill="#142338" stroke="#263b55"/>
    <text x="122" y="289" fill="#72aaff" font-size="14" font-weight="750" letter-spacing="2">FIXED-POINT RESULT COMPARISON</text>
    <text x="122" y="335" class="w" font-size="18">RTL-arithmetic software mirror vs Python reference</text>
    <rect x="122" y="355" width="660" height="17" rx="8" fill="#263b55"/><rect x="122" y="355" width="{python_bar:.2f}" height="17" rx="8" fill="url(#green)"/>
    <text x="810" y="373" class="w mono" font-size="20">{python_error:.5f}</text>
    <text x="122" y="411" class="m" font-size="15">Maximum absolute output difference · dequantized value units</text>
    <text x="122" y="446" fill="#57d9aa" font-size="15" font-weight="700">ALL {numeric["numeric_cases"]} CASES WITHIN {tolerance:.2f} TOLERANCE</text>
    <text x="870" y="335" class="w" font-size="18">{torch_label}</text>
    <rect x="870" y="355" width="570" height="17" rx="8" fill="#263b55"/><rect x="870" y="355" width="{torch_bar:.2f}" height="17" rx="8" fill="#6c8cff"/>
    <text x="1460" y="373" class="w mono" font-size="20" text-anchor="end">{torch_metric}</text>
    <text x="870" y="411" class="m" font-size="15">Separate numeric comparison; not bit-identical floating-point behavior</text>
    <text x="870" y="446" fill="#93aaff" font-size="15" font-weight="700">WORST OF {numeric["numeric_cases"]} SEEDED/EDGE CASES</text>
  </g>
  <g>
    <rect x="84" y="499" width="446" height="194" rx="22" fill="#142338" stroke="#263b55"/>
    <rect x="554" y="499" width="446" height="194" rx="22" fill="#142338" stroke="#263b55"/>
    <rect x="1024" y="499" width="492" height="194" rx="22" fill="#142338" stroke="#263b55"/>
    <text x="118" y="540" class="m" font-size="14" font-weight="700" letter-spacing="1.5">LUT + EXPONENT CHECK</text>
    <text x="118" y="589" class="w" font-size="34" font-weight="750">64 / 64</text>
    <text x="118" y="623" fill="#57d9aa" font-size="15" font-weight="700">RTL LUT MATCHES PYTHON</text>
    <text x="118" y="662" class="m" font-size="14">{report["lut_and_exponential"]["exp_input_deltas_checked"]:,} deltas match; max exp abs error {report["lut_and_exponential"]["max_abs_exp_error"]:.5f}</text>
    <text x="588" y="540" class="m" font-size="14" font-weight="700" letter-spacing="1.5">EDGE + RANDOM TEST VECTORS</text>
    <text x="588" y="589" class="w" font-size="34" font-weight="750">{numeric["deterministic_edge_cases"]} + {numeric["random_seeds"]}</text>
    <text x="588" y="623" fill="#57d9aa" font-size="15" font-weight="700">ALL {numeric["numeric_cases"]} NUMERIC TESTS PASS</text>
    <text x="588" y="662" class="m" font-size="14">Zero · equal scores · dominant score · extreme bytes · seeded random</text>
    <text x="1058" y="540" class="m" font-size="14" font-weight="700" letter-spacing="1.5">STORAGE + STREAM MODEL</text>
    <text x="1058" y="589" class="w" font-size="29" font-weight="750">NO N × N SCORE ARRAY</text>
    <text x="1058" y="623" fill="#57d9aa" font-size="15" font-weight="700">188,800 INPUT / 320 OUTPUT WORDS</text>
    <text x="1058" y="662" class="m" font-size="14">20 ready/valid schedule trials with randomized stalls</text>
  </g>
  <rect x="84" y="729" width="1432" height="100" rx="18" fill="#321f27" stroke="#b65d62"/>
  <text x="116" y="768" fill="#ffb5ad" font-size="16" font-weight="800" letter-spacing="1">IMPORTANT: SOFTWARE VERIFICATION ONLY</text>
  <text x="116" y="802" class="w" font-size="15">No SystemVerilog compiler/simulator, Vivado bitstream build, or PYNQ-Z2 board test was available. FPGA behavior is NOT yet verified.</text>
  <text x="84" y="871" class="m" font-size="14">Run: python3 sim/verify_demo.py · Details + per-case values: verification_report.md / numeric_cases.csv</text>
  <text x="1516" y="871" class="m" font-size="14" text-anchor="end">No timing, power, or image-accuracy claim</text>
</svg>
'''
    svg_path.write_text(svg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=SEED_COUNT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least one")

    lut_report = check_lut_and_exp()
    interface_report = check_interface_and_memory_shape()
    protocol_report = run_protocol_stress(seed=20261108, trials=20)
    cases, numeric_report = run_numeric_cases(args.seeds)
    report = {
        "run_time_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "overall_result": "PASS",
        "scope": (
            "host software verification of RTL-mirrored arithmetic and protocol schedule; "
            "not HDL simulation or board validation"
        ),
        "configuration": {
            "board_target": "TUL PYNQ-Z2 / XC7Z020",
            "query_width": QUERY_DIM,
            "token_count": TOKEN_COUNT,
            "value_width": VALUE_DIM,
            "score_format": "signed Q12",
            "exponential_weight_format": "unsigned Q15",
            "input_format": "signed int8 stored as sign-extended 32-bit stream words, Q4 scale",
            "output_format": "signed int8, Q4 scale",
        },
        "lut_and_exponential": lut_report,
        "interface_and_memory": interface_report,
        "protocol": protocol_report,
        "numeric": numeric_report,
        "hdl_tool_status": detect_hdl_tools(),
    }
    write_reports(args.report_dir, cases, report)
    print("LOCAL SOFTWARE VERIFICATION: PASS")
    print(f"Numeric cases: {numeric_report['numeric_cases']}")
    print(
        "Worst fixed-point output error vs Python reference: "
        f"{numeric_report['max_abs_output_error_vs_python']:.6f} "
        f"({numeric_report['worst_python_comparison_case']})"
    )
    if numeric_report["pytorch_available"]:
        print(
            "Worst fixed-point output error vs PyTorch: "
            f"{numeric_report['max_abs_output_error_vs_torch']:.6f} "
            f"({numeric_report['worst_torch_comparison_case']})"
        )
    print(
        "HDL simulation: NOT RUN — "
        + report["hdl_tool_status"]["reason"]
    )
    print(f"Reports written to: {args.report_dir}")


if __name__ == "__main__":
    main()
