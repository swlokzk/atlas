import tempfile
import unittest
from pathlib import Path

import onnx
import onnxruntime as ort

from deploy.export_onnx import export_fp32_onnx


class OnnxExportTests(unittest.TestCase):
    def test_export_checker_and_runtime_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = export_fp32_onnx(Path(directory))
            onnx.checker.check_model(onnx.load(artifact_path))
            session = ort.InferenceSession(str(artifact_path), providers=["CPUExecutionProvider"])
            self.assertEqual([entry.name for entry in session.get_inputs()], ["iq", "stft", "std"])
            self.assertEqual(
                [entry.name for entry in session.get_outputs()],
                ["logits", "gating_weights"],
            )


if __name__ == "__main__":
    unittest.main()