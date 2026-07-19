import unittest

from deploy.runtime import RUNTIME_PROVIDERS, resolve_runtime


class RuntimeSelectionTests(unittest.TestCase):
    def test_cpu_uses_only_cpu_provider(self) -> None:
        selection = resolve_runtime("cpu", ["CPUExecutionProvider", "CUDAExecutionProvider"])
        self.assertEqual(selection.selected, "cpu")
        self.assertEqual(selection.providers, ("CPUExecutionProvider",))

    def test_auto_uses_cpu_on_cpu_only_runtime(self) -> None:
        selection = resolve_runtime("auto", ["CPUExecutionProvider"])
        self.assertEqual(selection.selected, "cpu")
        self.assertEqual(selection.providers, RUNTIME_PROVIDERS["cpu"])

    def test_cuda_requires_cuda_provider(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
            resolve_runtime("cuda", ["CPUExecutionProvider"])

    def test_tensorrt_provider_chain_has_cpu_fallback(self) -> None:
        selection = resolve_runtime("tensorrt", ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(selection.providers, RUNTIME_PROVIDERS["tensorrt"])


if __name__ == "__main__":
    unittest.main()