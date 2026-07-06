import torch
from typing import Sequence, Tuple


class ModelExporter:
    """Simple ONNX exporter for models in the `models` package.

    Usage:
        exporter = ModelExporter(model, device='cpu')
        exporter.export_onnx(file_path, input_shapes, opset_version=13)
    """

    def __init__(self, model, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device

    def _make_dummy_inputs(self, input_shapes: Sequence[Tuple[int, ...]]):
        return [torch.randn(shape, device=self.device) for shape in input_shapes]

    def export_onnx(self, file_path: str, input_shapes: Sequence[Tuple[int, ...]], opset_version: int = 13):
        self.model.eval()
        dummy_inputs = self._make_dummy_inputs(input_shapes)
        example_inputs = tuple(dummy_inputs) if len(dummy_inputs) > 1 else dummy_inputs[0]
        input_names = [f'input{i}' for i in range(len(dummy_inputs))] if len(dummy_inputs) > 1 else ['input']
        dynamic_axes = {name: {0: 'batch'} for name in input_names}
        output_names = ['logits', 'weights'] if hasattr(self.model, 'head') else ['output']

        torch.onnx.export(
            self.model,
            example_inputs,
            file_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
