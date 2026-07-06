"""
config.py — 全局共享配置

所有實驗腳本皆從此模塊讀取路徑、類別名稱與模型超參數，
修改此檔案即可一次性更新全部實驗的設定。
"""

import argparse
import torch


# --- 數據集 ---
CLASSES = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4',
           'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'AM-SSB', 'WBFM']
NUM_CLASSES = len(CLASSES)

# --- 模型超參數 (須與訓練時一致) ---
MODEL_CFG = dict(
    embed_dim=96,
    num_classes=NUM_CLASSES,
    stft_time_steps=128,
    depth=4,
    num_heads=4,
)

# --- 可復現性 ---
SEED = 29

# --- 推理設定 ---
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 分析常數 ---
HIGH_SNR = 18                    # 「高 SNR」閾值，用於易混淆類別分析
TSNE_SNRS = (-8, 12)             # t-SNE 可視化的 SNR 值 (低 / 高)
TSNE_MAX_SAMPLES = 2000          # t-SNE 每個 SNR 的最大樣本數


def base_parser(description: str) -> argparse.ArgumentParser:
    """建立含公共參數 (模型路徑、資料路徑) 的基礎 ArgumentParser。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--weights", type=str, required=True,
        help="訓練好的模型權重路徑 (.pth)"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="RML2016.10a_dict.pkl 資料集路徑"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"推理批次大小 (預設: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="圖表輸出目錄 (預設: outputs)"
    )
    return parser
