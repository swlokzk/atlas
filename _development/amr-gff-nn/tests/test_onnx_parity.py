import tempfile
import unittest
from pathlib import Path

from deploy.export_onnx import export_fp32_onnx
from deploy.validate_onnx import validate_onnx_parity


class OnnxParityTests(unittest.TestCase):
    def test_deterministic_pytorch_onnx_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            artifact_path = export_fp32_onnx(output_dir)
            report = validate_onnx_parity(artifact_path, output_dir)
        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["top1_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()