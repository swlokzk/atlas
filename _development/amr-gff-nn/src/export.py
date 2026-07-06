import torch
import argparse
import os
import sys
from typing import Tuple

# 假設您的模型都定義在 models 文件夾下
import models

# Try to import a reusable ModelExporter from models package; if missing,
# provide a lightweight fallback here so `src/export.py` stays runnable.
try:
    from models.exporter import ModelExporter  # 引用之前寫好的類
except Exception:
    import torch
    from typing import Sequence, Tuple

    class ModelExporter:
        """Fallback ONNX exporter if models.exporter is not available."""

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

def get_args():
    parser = argparse.ArgumentParser(description="AMC-AMR 模型通用 ONNX 導出工具")
    
    # 1. 模型選擇與配置
    parser.add_argument('--model', type=str, default='GatedFusionFormer', 
                        help='模型類名 (e.g., CNN2, ModRecNet, GatedFusionFormer)')
    parser.add_argument('--weights', type=str, default=None, 
                        help='預訓練權重路徑 (.pth)，若無則導出隨機初始化模型')
    parser.add_argument('--num_classes', type=int, default=11, 
                        help='分類類別總數')
    
    # 2. 導出設定
    parser.add_argument('--output', type=str, default='output_model.onnx', 
                        help='輸出的 ONNX 文件名')
    parser.add_argument('--opset', type=int, default=13, 
                        help='ONNX Opset 版本 (建議 >= 12)')
    parser.add_argument('--device', type=str, default='cpu', 
                        choices=['cpu', 'cuda'], help='導出設備')
    
    # 3. 輸入形狀配置 (針對多輸入模型自動解析)
    parser.add_argument('--iq_len', type=int, default=128, help='IQ 信號長度')
    parser.add_argument('--stft_f', type=int, default=64, help='STFT 頻率維度')
    
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)

    # --- 第一步：動態實例化模型 ---
    print(f"[*] 正在從 models 模塊中加載類: {args.model}")
    try:
        # 使用 getattr 從 models 包中動態獲取類名
        model_class = getattr(models, args.model)
        model = model_class(num_classes=args.num_classes)
    except AttributeError:
        print(f"錯誤: 在 models 中找不到類 '{args.model}'。請檢查 __init__.py 是否正確導入。")
        sys.exit(1)

    # --- 第二步：加載權重 (如有) ---
    if args.weights and os.path.exists(args.weights):
        print(f"[*] 正在加載權重: {args.weights}")
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("[!] 未指定權重或路徑無效，將導出隨機初始化的模型。")

    # --- 第三步：根據模型類型準備輸入形狀 ---
    # 這裡體現泛用性：針對不同的模型類，自動匹配 Dummy Input 形狀
    if args.model == 'GatedFusionFormer':
        # 多輸入形狀: (IQ, STFT, STD)
        input_shapes = (
            (1, 2, args.iq_len),               # IQ
            (1, 1, args.stft_f, args.iq_len),  # STFT
            (1, 2, args.iq_len)                # STD
        )
    else:
        # 單輸入形狀 (如 CNN2, ModRecNet): (B, C, T)
        input_shapes = ((1, 2, args.iq_len),)

    # --- 第四步：調用 Exporter ---
    exporter = ModelExporter(model, device=args.device)
    
    # 自動根據模型名稱微調輸出路徑
    output_path = args.output if args.output != 'output_model.onnx' else f"{args.model}_v1.onnx"
    
    try:
        exporter.export_onnx(
            file_path=output_path,
            input_shapes=input_shapes,
            opset_version=args.opset
        )
        print(f"\n[成功] 模型已保存至: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"\n[失敗] 導出過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()
