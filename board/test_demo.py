import unittest

import numpy as np

from run_demo import (
    INPUT_WORDS,
    QUERY_DIM,
    TOKEN_COUNT,
    VALUE_DIM,
    make_test_case,
    pack_input_words,
    reference_output,
)


class DemoDataTests(unittest.TestCase):
    def test_seeded_case_is_reproducible_and_correctly_shaped(self):
        first = make_test_case(7)
        second = make_test_case(7)
        expected_shapes = (
            (QUERY_DIM,),
            (TOKEN_COUNT, QUERY_DIM),
            (TOKEN_COUNT, VALUE_DIM),
        )
        for left, right, shape in zip(first, second, expected_shapes):
            self.assertEqual(left.shape, shape)
            self.assertTrue(np.array_equal(left, right))
            self.assertEqual(left.dtype, np.int8)

    def test_dma_words_are_sign_extended_and_key_value_interleaved(self):
        query = np.zeros(QUERY_DIM, dtype=np.int8)
        keys = np.zeros((TOKEN_COUNT, QUERY_DIM), dtype=np.int8)
        values = np.zeros((TOKEN_COUNT, VALUE_DIM), dtype=np.int8)
        query[0] = -1
        keys[0, 0] = -2
        values[0, 0] = 3
        keys[1, 0] = 4
        values[1, 0] = -5

        words = pack_input_words(query, keys, values)

        self.assertEqual(words.size, INPUT_WORDS)
        self.assertEqual(int(words[0]), 0xFFFFFFFF)
        first_token = QUERY_DIM
        second_token = QUERY_DIM + QUERY_DIM + VALUE_DIM
        self.assertEqual(int(words[first_token]), 0xFFFFFFFE)
        self.assertEqual(int(words[first_token + QUERY_DIM]), 3)
        self.assertEqual(int(words[second_token]), 4)
        self.assertEqual(int(words[second_token + QUERY_DIM]), 0xFFFFFFFB)

    def test_python_reference_produces_finite_output_vector(self):
        query, keys, values = make_test_case(19)
        output = reference_output(query, keys, values)
        self.assertEqual(output.shape, (VALUE_DIM,))
        self.assertTrue(np.all(np.isfinite(output)))
        lower = values.min(axis=0) / 16.0
        upper = values.max(axis=0) / 16.0
        self.assertTrue(np.all(output >= lower - 1e-9))
        self.assertTrue(np.all(output <= upper + 1e-9))


if __name__ == "__main__":
    unittest.main()
