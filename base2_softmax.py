"""Fixed-point model of a streaming base-2 online softmax.

Scores are quantized to a configurable fixed-point format. Exponential
weights use a 64-entry Q15 LUT and integer right shifts. The attention path
consumes key/value blocks and never constructs the full query-by-key score
matrix.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable

import numpy as np


LUT_FRACTIONAL_BITS = 6
WEIGHT_FRACTIONAL_BITS = 15
WEIGHT_ONE = 1 << WEIGHT_FRACTIONAL_BITS
LOG2E_Q24 = 24_204_406
DEFAULT_SCORE_FRACTIONAL_BITS = 12

# One entry for each six-bit fractional exponent, rounded to nearest Q15.
FRACTIONAL_LUT_Q15 = tuple(
    int(math.floor((2.0 ** (index / (1 << LUT_FRACTIONAL_BITS))) * WEIGHT_ONE + 0.5))
    for index in range(1 << LUT_FRACTIONAL_BITS)
)


def _validate_score_fractional_bits(bits: int) -> None:
    if not isinstance(bits, int) or isinstance(bits, bool) or not 0 <= bits <= 24:
        raise ValueError("score_fractional_bits must be an integer from 0 to 24")


def _quantize_scores(scores: np.ndarray, fractional_bits: int) -> np.ndarray:
    _validate_score_fractional_bits(fractional_bits)
    values = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values")
    scaled = np.rint(values * (1 << fractional_bits))
    if np.any(scaled < np.iinfo(np.int64).min) or np.any(
        scaled > np.iinfo(np.int64).max
    ):
        raise ValueError("score is outside the supported signed 64-bit range")
    return scaled.astype(np.int64)


def exp_neg_q15(delta_score_q: int, score_fractional_bits: int) -> int:
    """Approximate exp(-delta) as an unsigned Q15 integer using the LUT.

    ``delta_score_q`` is a nonnegative score difference represented with
    ``score_fractional_bits`` fractional bits. The six fractional exponent
    bits are truncated; the integer exponent is applied with a right shift.
    """
    _validate_score_fractional_bits(score_fractional_bits)
    if not isinstance(delta_score_q, (int, np.integer)) or delta_score_q < 0:
        raise ValueError("delta_score_q must be a nonnegative integer")

    exponent_units = (
        int(delta_score_q) * LOG2E_Q24 * (1 << LUT_FRACTIONAL_BITS)
    ) >> (score_fractional_bits + 24)
    integer_part, fractional_part = divmod(exponent_units, 1 << LUT_FRACTIONAL_BITS)

    if fractional_part == 0:
        lut_index = 0
        shift = integer_part
    else:
        lut_index = (1 << LUT_FRACTIONAL_BITS) - fractional_part
        shift = integer_part + 1

    if shift >= WEIGHT_FRACTIONAL_BITS + 1:
        return 0
    return FRACTIONAL_LUT_Q15[lut_index] >> shift


def _online_weights_q15(
    quantized_scores: np.ndarray, score_fractional_bits: int
) -> tuple[list[int], int]:
    """Run the online max/denominator recurrence, retaining O(N) row weights."""
    maximum: int | None = None
    weights: list[int] = []

    for score in quantized_scores:
        current = int(score)
        if maximum is None:
            maximum = current
            weights.append(WEIGHT_ONE)
            continue

        new_maximum = max(maximum, current)
        alpha = exp_neg_q15(new_maximum - maximum, score_fractional_bits)
        beta = exp_neg_q15(new_maximum - current, score_fractional_bits)
        weights = [(weight * alpha) >> WEIGHT_FRACTIONAL_BITS for weight in weights]
        weights.append(beta)
        maximum = new_maximum

    if not weights:
        raise ValueError("softmax requires at least one score")
    return weights, sum(weights)


def online_softmax(
    scores: Iterable[float] | np.ndarray,
    score_fractional_bits: int = DEFAULT_SCORE_FRACTIONAL_BITS,
) -> np.ndarray:
    """Return approximate softmax probabilities for one streamed score row.

    The returned probability vector is O(N), as required for the requested
    output. The online recurrence itself keeps only the running maximum,
    denominator weights, and the output-sized vector; it does not form an
    N-by-N score matrix.
    """
    values = np.asarray(list(scores) if not isinstance(scores, np.ndarray) else scores)
    if values.ndim != 1:
        raise ValueError("scores must be a one-dimensional row")
    quantized = _quantize_scores(values, score_fractional_bits)
    weights, denominator = _online_weights_q15(quantized, score_fractional_bits)
    return np.asarray(weights, dtype=np.float64) / denominator


def streaming_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    key_block_size: int = 64,
    score_scale: float | None = None,
    score_fractional_bits: int = DEFAULT_SCORE_FRACTIONAL_BITS,
) -> np.ndarray:
    """Compute one-query attention by streaming key/value blocks.

    Additional score storage is bounded by ``key_block_size``; there is no
    query-by-key score matrix. The online recurrence updates the running
    maximum ``m``, denominator ``l``, and weighted value accumulator ``a``:

        m' = max(m, x)
        alpha = exp(x_old_max - m')
        beta = exp(x - m')
        l' = alpha*l + beta
        a' = alpha*a + beta*v

    Exponentials are represented with the Q15 LUT and shifts. Value
    accumulation uses float64 to keep the model's main approximation focused
    on score quantization and exponentiation.
    """
    _validate_score_fractional_bits(score_fractional_bits)
    if (
        not isinstance(key_block_size, int)
        or isinstance(key_block_size, bool)
        or key_block_size < 1
    ):
        raise ValueError("key_block_size must be a positive integer")

    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(keys, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if q.ndim != 1 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("query must be 1-D and keys/values must be 2-D")
    if k.shape[0] == 0 or k.shape[0] != v.shape[0] or k.shape[1] != q.size:
        raise ValueError("keys/values must have equal nonzero rows and match query width")
    if q.size == 0 or v.shape[1] == 0:
        raise ValueError("query and value widths must be nonzero")
    if (
        not np.all(np.isfinite(q))
        or not np.all(np.isfinite(k))
        or not np.all(np.isfinite(v))
    ):
        raise ValueError("query, keys, and values must contain only finite values")

    scale = (1.0 / math.sqrt(q.size)) if score_scale is None else float(score_scale)
    if not math.isfinite(scale):
        raise ValueError("score_scale must be finite")

    maximum: int | None = None
    denominator_q15 = 0
    accumulator = np.zeros(v.shape[1], dtype=np.float64)

    for start in range(0, k.shape[0], key_block_size):
        stop = min(start + key_block_size, k.shape[0])
        score_block = np.fromiter(
            (float(np.dot(k[index], q)) * scale for index in range(start, stop)),
            dtype=np.float64,
            count=stop - start,
        )
        score_block_q = _quantize_scores(score_block, score_fractional_bits)

        for offset, score in enumerate(score_block_q):
            current = int(score)
            value = v[start + offset]
            if maximum is None:
                maximum = current
                denominator_q15 = WEIGHT_ONE
                accumulator = value.copy()
                continue

            new_maximum = max(maximum, current)
            alpha = exp_neg_q15(new_maximum - maximum, score_fractional_bits)
            beta = exp_neg_q15(new_maximum - current, score_fractional_bits)
            denominator_q15 = (
                (denominator_q15 * alpha) >> WEIGHT_FRACTIONAL_BITS
            ) + beta
            accumulator = (
                accumulator * (alpha / WEIGHT_ONE)
                + value * (beta / WEIGHT_ONE)
            )
            maximum = new_maximum

    return accumulator / (denominator_q15 / WEIGHT_ONE)


def _run_self_test() -> None:
    if len(FRACTIONAL_LUT_Q15) != 64 or FRACTIONAL_LUT_Q15[0] != WEIGHT_ONE:
        raise AssertionError("fractional LUT shape or unity entry is incorrect")
    if any(
        lower > upper
        for lower, upper in zip(FRACTIONAL_LUT_Q15, FRACTIONAL_LUT_Q15[1:])
    ):
        raise AssertionError("fractional LUT entries must be monotonic")
    if exp_neg_q15(0, DEFAULT_SCORE_FRACTIONAL_BITS) != WEIGHT_ONE:
        raise AssertionError("zero score difference must map to one")
    if any(
        exp_neg_q15(delta, DEFAULT_SCORE_FRACTIONAL_BITS)
        < exp_neg_q15(delta + 1, DEFAULT_SCORE_FRACTIONAL_BITS)
        for delta in range(4096)
    ):
        raise AssertionError("base-2 exponential mapping must be monotonic")
    max_exp_error = max(
        abs(
            exp_neg_q15(delta, DEFAULT_SCORE_FRACTIONAL_BITS) / WEIGHT_ONE
            - math.exp(-delta / (1 << DEFAULT_SCORE_FRACTIONAL_BITS))
        )
        for delta in range(1 << 16)
    )
    if max_exp_error > 0.011:
        raise AssertionError(f"base-2 exponential absolute error is too large: {max_exp_error}")

    rng = np.random.default_rng(20261014)
    try:
        import torch
    except ImportError:
        torch = None

    test_cases = [np.array([0.0]), np.array([1.0, 1.0, 1.0])]
    test_cases.extend(rng.normal(0.0, 2.0, size=length) for length in (2, 17, 196))
    for scores in test_cases:
        actual = online_softmax(scores)
        if not np.isclose(actual.sum(), 1.0, atol=1e-12):
            raise AssertionError("online softmax probabilities do not sum to one")
        if torch is not None:
            expected = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=0).numpy()
            np.testing.assert_allclose(actual, expected, rtol=0.02, atol=2e-4)

    query = rng.normal(size=32).astype(np.float32)
    keys = rng.normal(size=(196, 32)).astype(np.float32)
    values = rng.normal(size=(196, 16)).astype(np.float32)
    streamed = streaming_attention(query, keys, values, key_block_size=13)
    alternate_tiles = streaming_attention(query, keys, values, key_block_size=64)
    np.testing.assert_array_equal(streamed, alternate_tiles)

    if torch is not None:
        torch_query = torch.tensor(query)
        torch_keys = torch.tensor(keys)
        torch_values = torch.tensor(values)
        expected = torch.softmax(
            (torch_keys @ torch_query) / math.sqrt(query.size), dim=0
        ) @ torch_values
        np.testing.assert_allclose(
            streamed, expected.numpy(), rtol=0.025, atol=2e-3
        )
        print("PyTorch Softmax parity: passed (within stated approximation tolerances)")
    else:
        print("PyTorch comparison skipped (install PyTorch to enable it)")

    print("LUT, online recurrence, normalization, and streaming attention: passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="verify the LUT, online recurrence, and optional PyTorch parity",
    )
    args = parser.parse_args()
    if args.self_test:
        _run_self_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
