import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


"""

# 重構說明：

-  **`nn.Sequential` 的使用**：
    `features` 部分變得非常整齊。當需要修改某個層時，只需修改這一處即可，`forward` 函數不需要改動。
    
-  **`self._get_num_features_out()`**：
    這是現代 PyTorch 模型（特別是自定義模型）的常見模式。
    它在 `__init__` 中運行一次，利用 dummy input 計算卷積層輸出的最終通道數。
    這意味現在可以隨意修改 `self.features` 中的 `kernel_size`, `out_channels` 或增加層數，
    **不需要手動去計算分類器的 `fc1 = nn.Linear(?, 512)` 裡的那個問號**，程序會自動搞定。
    
-  **ONNX/Inference 友好**：
    `nn.Sequential` 結構在導出為 ONNX 格式進行部署時，往往比純 `forward` 硬編碼的結構具有更好的穩定性和兼容性。
    
-  **`create_model` 優化**：
    使用 `MODEL_REGISTRY` 字典，使得添加新模型（如 `CNN2`）只需在字典中註冊即可。同時支持了 `**kwargs`，提高了參數傳遞的靈活性。
    
"""

class ModRecNet(nn.Module):
    """用於調制識別 (Modulation Recognition) 的網絡模型。

    採用 nn.Sequential 結構化設計，並具備自動計算分類器輸入維度的功能。
    """
    def __init__(self, num_classes: int, input_channels: int = 2):
        super(ModRecNet, self).__init__()
        
        # 1. 特徵提取器 (Feature Extractor)
        # 使用 Sequential 將 卷積+BN+ReLU+Pool 封裝，代碼更簡潔且有利於 ONNX 導出
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), # inplace=True 可節省顯存
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Layer 3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # 2. 全局池化與扁平化 (Global Pooling & Flattening)
        # 將不同長度的序列輸出統一為固定的特徵維度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 3. 分類器 (Classifier)
        # 計算特徵提取器的輸出通道數 (這裡是 256)
        num_features_out = self._get_num_features_out()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features_out, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def _get_num_features_out(self) -> int:
        """【自動化】通過模擬前向傳播來計算 features 的輸出通道數。
        
        這可以避免手動硬編碼分類器的輸入維度，增強代碼健壯性。
        """
        # 建立一個與真實數據通道數相同的 dummy input (batch_size=1)
        # 序列長度隨意 (例如 128)，因為 adaptive_pool 會處理
        dummy_input = torch.zeros(1, self.features[0].in_channels, 128)
        with torch.no_grad():
            dummy_out = self.features(dummy_input)
        return dummy_out.size(1) # 返回通道數 (Channels)

    def forward(self, x: torch.Tensor, features_only: bool = False) -> torch.Tensor:
        # 通過特徵提取器
        x = self.features(x)
        
        # 全局適應性池化 (N, C, L) -> (N, C, 1)
        x = self.adaptive_pool(x)
        
        # 扁平化 (N, C, 1) -> (N, C) 為全連接層做準備
        features = torch.flatten(x, 1)
        
        if features_only:
            # 返回扁平化後的特徵 (通常用於特徵可視化或度量學習)
            return features
            
        # 通過分類器獲得 Logits
        logits = self.classifier(features)
        return logits


def create_model(model_name: str, num_classes: int, device: Optional[torch.device] = None, **kwargs) -> nn.Module:
    """根據名稱創建模型實例的通用工廠函數。

    優化：
    1. 使用簡化的字符串對應類。
    2. 支持動態 device 分配。
    3. 支持通過 **kwargs 傳遞額外參數 (例如 input_channels)。
    """
    # 將所有模型類註冊在字典中
    MODEL_REGISTRY = {
        'modrecnet': ModRecNet,
        # 'cnn2': CNN2, # 如果有 CNN2 也加在這裡
    }

    # 標準化模型名稱 (不區分大小寫)
    model_name_lower = model_name.lower()
    
    print(f"[{create_model.__name__}] 正嘗試創建模型: '{model_name_lower}'...")
    
    if model_name_lower not in MODEL_REGISTRY:
        raise ValueError(f"❌ 未知的模型名稱: '{model_name}'。可用模型: {list(MODEL_REGISTRY.keys())}")
    
    # 獲取類別並實例化
    model_cls = MODEL_REGISTRY[model_name_lower]
    model = model_cls(num_classes=num_classes, **kwargs)
    
    # 自動獲取類名打印
    actual_class_name = model.__class__.__name__
    print(f"✅ 模型 '{actual_class_name}' 實例化成功。")

    if device:
        model.to(device)
        print(f"🚀 模型已移動至設備: {device}")
    
    return model

# --- 使用示例 ---
if __name__ == '__main__':
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10
    
    # 1. 創建模型
    model = create_model('modrecnet', num_classes, device=device)
    
    # 2. 生成模擬數據 (Batch_size, Channels, Length)
    # 假設 IQ 信號通道數為 2，長度為 128
    dummy_data = torch.randn(8, 2, 128).to(device)
    
    print(f"\n輸入數據形狀: {dummy_data.shape}")
    
    # 3. 前向傳播 (完整模式)
    output = model(dummy_data)
    print(f"完整模式輸出形狀 (Logits): {output.shape} (即 [Batch, NumClasses])")
    
    # 4. 前向傳播 (僅特徵模式)
    features = model(dummy_data, features_only=True)
    print(f"僅特徵模式輸出形狀: {features.shape} (即 [Batch, LastConvChannels])")
    
    # 5. 查看模型結構 (可選，安裝 torchsummary 後可用)
    # try:
    #     from torchsummary import summary
    #     summary(model, input_size=(2, 128))
    # except ImportError:
    #     print("\n未安裝 torchsummary，跳過結構摘要打印。")
    #     print(model) # 默認打印
