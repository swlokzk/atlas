import unittest

import numpy as np

from deploy.preprocessing import (
    InputValidationError,
    SIGNAL_LENGTH,
    STFT_FREQUENCY_BINS,
    preprocess_batch,
    preprocess_iq,
)


class PreprocessingTests(unittest.TestCase):
    def test_valid_signal_preserves_iq_shape(self) -> None:
        iq = np.vstack((np.linspace(-1.0, 1.0, SIGNAL_LENGTH), np.zeros(SIGNAL_LENGTH)))
        result = preprocess_iq(iq)
        self.assertEqual(result.iq.shape, (2, SIGNAL_LENGTH))
        self.assertEqual(result.stft.shape, (1, STFT_FREQUENCY_BINS, SIGNAL_LENGTH))
        self.assertEqual(result.std.shape, (2, SIGNAL_LENGTH))
        np.testing.assert_array_equal(result.iq, iq.astype(np.float32))

    def test_variable_length_interpolates_with_fixed_endpoints(self) -> None:
        result = preprocess_iq([[0.0, 2.0], [1.0, 3.0]])
        self.assertEqual(result.iq.shape, (2, SIGNAL_LENGTH))
        np.testing.assert_allclose(result.iq[:, 0], [0.0, 1.0])
        np.testing.assert_allclose(result.iq[:, -1], [2.0, 3.0])

    def test_length_one_repeats_without_interpolation_error(self) -> None:
        result = preprocess_iq([[1.5], [-2.0]])
        np.testing.assert_array_equal(result.iq[0], np.full(SIGNAL_LENGTH, 1.5))
        np.testing.assert_array_equal(result.iq[1], np.full(SIGNAL_LENGTH, -2.0))

    def test_invalid_iq_is_rejected(self) -> None:
        invalid_inputs = (
            [[], []],
            [[1.0, 2.0], [3.0]],
            [[np.nan], [0.0]],
            [[np.inf], [0.0]],
            [["bad"], [0.0]],
        )
        for invalid_iq in invalid_inputs:
            with self.subTest(invalid_iq=invalid_iq):
                with self.assertRaises(InputValidationError):
                    preprocess_iq(invalid_iq)

    def test_std_formula_and_zero_energy_signal(self) -> None:
        result = preprocess_iq([[2.0], [3.0]])
        self.assertEqual(result.std[0, 0], -5.0)
        self.assertEqual(result.std[1, 0], 12.0)

        zero_result = preprocess_iq(np.zeros((2, SIGNAL_LENGTH), dtype=np.float32))
        self.assertTrue(np.array_equal(zero_result.std, np.zeros((2, SIGNAL_LENGTH))))
        self.assertTrue(np.array_equal(zero_result.stft, np.zeros((1, 32, 128))))

    def test_preprocessing_is_deterministic(self) -> None:
        iq = np.vstack((np.arange(64), -np.arange(64))).astype(np.float32)
        first = preprocess_iq(iq)
        second = preprocess_iq(iq)
        np.testing.assert_array_equal(first.iq, second.iq)
        np.testing.assert_array_equal(first.stft, second.stft)
        np.testing.assert_array_equal(first.std, second.std)

    def test_batch_shape_contract(self) -> None:
        iq_batch = np.zeros((3, 2, SIGNAL_LENGTH), dtype=np.float32)
        iq, stft, std = preprocess_batch(iq_batch)
        self.assertEqual(iq.shape, (3, 2, SIGNAL_LENGTH))
        self.assertEqual(stft.shape, (3, 1, STFT_FREQUENCY_BINS, SIGNAL_LENGTH))
        self.assertEqual(std.shape, (3, 2, SIGNAL_LENGTH))


if __name__ == "__main__":
    unittest.main()