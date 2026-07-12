import unittest

import numpy as np

from deploy.model_wrapper import GFFInferenceWrapper
from deploy.predict import predict_iq


class PyTorchInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.wrapper, cls.config, _ = GFFInferenceWrapper.from_checkpoint()

    def test_prediction_response_contract(self) -> None:
        result = predict_iq(np.zeros((2, 128), dtype=np.float32), self.wrapper, self.config)
        self.assertIn(result["prediction"], self.config["labels"])
        self.assertEqual(result["class_id"], self.config["labels"].index(result["prediction"]))
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=6)
        self.assertAlmostEqual(sum(result["gating_weights"].values()), 1.0, places=6)
        self.assertTrue(all(value >= 0.0 for value in result["gating_weights"].values()))
        self.assertTrue(all(result[key] >= 0.0 for key in (
            "preprocessing_latency_ms", "inference_latency_ms", "total_latency_ms"
        )))


if __name__ == "__main__":
    unittest.main()