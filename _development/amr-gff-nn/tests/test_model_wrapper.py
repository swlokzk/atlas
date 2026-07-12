import unittest

import torch

from deploy.model_wrapper import GFFInferenceWrapper


class ModelWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.wrapper, cls.config, _ = GFFInferenceWrapper.from_checkpoint()

    def test_checkpoint_backed_output_contract(self) -> None:
        batch_size = 2
        with torch.inference_mode():
            logits, weights = self.wrapper(
                torch.zeros(batch_size, 2, 128),
                torch.zeros(batch_size, 1, 32, 128),
                torch.zeros(batch_size, 2, 128),
            )
        self.wrapper.validate_outputs(logits, weights)
        self.assertEqual(tuple(logits.shape), (batch_size, 11))
        self.assertEqual(tuple(weights.shape), (batch_size, 3))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(batch_size)))


if __name__ == "__main__":
    unittest.main()