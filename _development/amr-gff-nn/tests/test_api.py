import unittest

from deploy.service.app import ServiceRuntime
from deploy.service.schemas import ClassifyRequest
from pydantic import ValidationError


class ApiTests(unittest.TestCase):
    def test_request_validation_rejects_malformed_iq(self) -> None:
        for value in ([[]], [[1.0], [2.0, 3.0]], [[float("nan")], [0.0]]):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    ClassifyRequest(iq=value)

    def test_prebuilt_runtime_loads(self) -> None:
        runtime = ServiceRuntime()
        runtime.load()
        self.assertIsNotNone(runtime.session)
        self.assertIsNotNone(runtime.artifact_path)


if __name__ == "__main__":
    unittest.main()