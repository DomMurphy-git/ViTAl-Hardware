import unittest

import numpy as np

from verify_demo import (
    check_lut_and_exp,
    rtl_integer_attention,
    run_protocol_stress,
    score_q12_from_int8,
)
from board.run_demo import make_test_case, reference_output, TOKEN_COUNT


class RTLArithmeticMirrorTests(unittest.TestCase):
    def test_rtl_lut_and_exponent_match_python_reference(self):
        result = check_lut_and_exp()
        self.assertEqual(result["rtl_lut_entries"], 64)
        self.assertTrue(result["rtl_lut_matches_python"])
        self.assertTrue(result["rtl_exp_matches_python_all_checked_deltas"])
        self.assertLessEqual(result["max_abs_exp_error"], 0.011)

    def test_score_scale_and_rounding_for_positive_negative_and_zero(self):
        query = np.zeros(32, dtype=np.int8)
        key = np.zeros(32, dtype=np.int8)
        self.assertEqual(score_q12_from_int8(query, key), 0)

        query[0] = 1
        key[0] = 1
        self.assertEqual(score_q12_from_int8(query, key), 3)
        key[0] = -1
        self.assertEqual(score_q12_from_int8(query, key), -3)

    def test_rtl_integer_mirror_matches_reference_within_demo_tolerance(self):
        query, keys, values = make_test_case(31)
        output_q4, diagnostics = rtl_integer_attention(query, keys, values)
        actual = output_q4.astype(np.float64) / 16.0
        expected = reference_output(query, keys, values)
        self.assertEqual(output_q4.shape, (16,))
        self.assertLessEqual(np.max(np.abs(actual - expected)), 0.20)
        self.assertLess(diagnostics["max_abs_accumulator"], 1 << 47)
        self.assertLess(diagnostics["max_denominator_q15"], 1 << 32)

    def test_zero_query_yields_mean_value_for_equal_scores(self):
        query = np.zeros(32, dtype=np.int8)
        rng = np.random.default_rng(42)
        keys = rng.integers(-128, 128, size=(TOKEN_COUNT, 32), dtype=np.int8)
        values = rng.integers(-128, 128, size=(TOKEN_COUNT, 16), dtype=np.int8)
        output_q4, _ = rtl_integer_attention(query, keys, values)
        expected_q4 = np.rint(values.astype(np.float64).mean(axis=0)).astype(np.int16)
        self.assertTrue(np.all(np.abs(output_q4.astype(int) - expected_q4) <= 1))

    def test_ready_valid_schedule_survives_randomized_stalls(self):
        result = run_protocol_stress(seed=55, trials=3)
        self.assertEqual(
            result["input_words_transferred"],
            result["expected_input_words"],
        )
        self.assertEqual(
            result["output_words_transferred"],
            result["expected_output_words"],
        )


if __name__ == "__main__":
    unittest.main()
