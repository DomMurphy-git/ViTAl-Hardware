#!/usr/bin/env python3
"""Run one deterministic streaming-attention transaction on a PYNQ-Z2 overlay."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_DIR = REPO_ROOT / "software"
if not (MODEL_DIR / "base2_softmax.py").is_file():
    MODEL_DIR = SCRIPT_DIR
sys.path.insert(0, str(MODEL_DIR))

from base2_softmax import streaming_attention

QUERY_DIM = 32
VALUE_DIM = 16
TOKEN_COUNT = 196
INPUT_WORDS = QUERY_DIM + TOKEN_COUNT * (QUERY_DIM + VALUE_DIM)
Q4_SCALE = 16.0
DEFAULT_TOLERANCE = 0.20


def make_test_case(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible Q4-scaled signed-byte query, key, and value tensors."""
    rng = np.random.default_rng(seed)

    def quantize(values: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(values * Q4_SCALE), -128, 127).astype(np.int8)

    query = quantize(rng.normal(0.0, 0.5, size=QUERY_DIM))
    keys = quantize(rng.normal(0.0, 0.5, size=(TOKEN_COUNT, QUERY_DIM)))
    values = quantize(rng.normal(0.0, 0.5, size=(TOKEN_COUNT, VALUE_DIM)))
    return query, keys, values


def pack_input_words(
    query: np.ndarray, keys: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Pack signed-byte inputs as sign-extended 32-bit AXI stream words."""
    expected_shapes = (
        (query.shape == (QUERY_DIM,)),
        (keys.shape == (TOKEN_COUNT, QUERY_DIM)),
        (values.shape == (TOKEN_COUNT, VALUE_DIM)),
    )
    if not all(expected_shapes):
        raise ValueError("input arrays do not match the demo's fixed dimensions")

    per_token_stream = np.column_stack((keys, values)).reshape(-1)
    byte_stream = np.concatenate((query.reshape(-1), per_token_stream))
    words = byte_stream.astype(np.int32).view(np.uint32)
    if words.size != INPUT_WORDS:
        raise AssertionError(f"expected {INPUT_WORDS} input words, got {words.size}")
    return np.ascontiguousarray(words)


def reference_output(
    query: np.ndarray, keys: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Compute the same quantized-input attention operation in the Python model."""
    return streaming_attention(
        query.astype(np.float64) / Q4_SCALE,
        keys.astype(np.float64) / Q4_SCALE,
        values.astype(np.float64) / Q4_SCALE,
        key_block_size=13,
    )


def run_on_board(bitstream: Path, seed: int, tolerance: float) -> None:
    try:
        from pynq import Overlay, allocate
    except ImportError as exc:
        raise RuntimeError(
            "PYNQ is not installed. Run this script on the PYNQ-Z2 Linux image."
        ) from exc

    if not bitstream.is_file():
        raise FileNotFoundError(f"overlay bitstream not found: {bitstream}")
    if not bitstream.with_suffix(".hwh").is_file():
        raise FileNotFoundError(
            f"matching HWH metadata not found: {bitstream.with_suffix('.hwh')}"
        )

    query, keys, values = make_test_case(seed)
    input_words = pack_input_words(query, keys, values)
    expected = reference_output(query, keys, values)

    overlay = Overlay(str(bitstream), download=True)
    if not hasattr(overlay, "axi_dma_0"):
        raise RuntimeError("overlay metadata does not contain axi_dma_0")
    dma = overlay.axi_dma_0

    input_buffer = allocate(shape=(INPUT_WORDS,), dtype=np.uint32)
    output_buffer = allocate(shape=(VALUE_DIM,), dtype=np.uint32)
    try:
        input_buffer[:] = input_words
        output_buffer[:] = 0

        # Arm the receiver first, then start the input stream.
        dma.recvchannel.transfer(output_buffer)
        dma.sendchannel.transfer(input_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        actual_q4 = output_buffer.view(np.int32).astype(np.float64)
        actual = actual_q4 / Q4_SCALE
        max_abs_error = float(np.max(np.abs(actual - expected)))
        print(f"Board: PYNQ-Z2 / XC7Z020")
        print(f"Input: {TOKEN_COUNT} keys, query width {QUERY_DIM}, value width {VALUE_DIM}")
        print(f"Maximum absolute output error vs. Python model: {max_abs_error:.6f}")
        print(f"Allowed tolerance: {tolerance:.6f}")
        print("FPGA output:", np.array2string(actual, precision=4))
        print("Reference:  ", np.array2string(expected, precision=4))

        if not math.isfinite(max_abs_error) or max_abs_error > tolerance:
            raise AssertionError(
                f"board output error {max_abs_error:.6f} exceeds tolerance {tolerance:.6f}"
            )
        print("BOARD DEMO PASSED")
    finally:
        input_buffer.freebuffer()
        output_buffer.freebuffer()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bitstream",
        type=Path,
        default=Path(__file__).with_name("vital_pynq_z2.bit"),
        help="bitstream path (matching .hwh metadata must sit beside it)",
    )
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--tolerance must be a finite nonnegative number")
    run_on_board(args.bitstream, args.seed, args.tolerance)


if __name__ == "__main__":
    main()
