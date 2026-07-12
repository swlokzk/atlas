import tempfile
import unittest
from pathlib import Path

import onnxruntime as ort

from deploy.export_onnx import export_fp32_onnx
from deploy.validate_onnx import validate_onnx_parity
from deploy.quantize_onnx import CalibrationDataRequiredError, quantize_dynamic_int8, quantize_static_int8


class QuantizationTests(unittest.TestCase):
    def test_dynamic_quantized_model_loads_after_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            artifact_path = export_fp32_onnx(output_dir)
            validate_onnx_parity(artifact_path, output_dir)
            quantized = quantize_dynamic_int8(output_dir)
            session = ort.InferenceSession(str(quantized), providers=["CPUExecutionProvider"])
        self.assertEqual([output.name for output in session.get_outputs()], ["logits", "gating_weights"])

    def test_static_quantization_requires_calibration_data(self) -> None:
        with self.assertRaises(CalibrationDataRequiredError):
            quantize_static_int8(Path("missing-calibration.npz"))


if __name__ == "__main__":
    unittest.main()