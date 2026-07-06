"""
05_cnn_vs_transformer.py — CNN Backbone vs Transformer 性能對比

功能：
  定義「禁用注意力」的可驗證版模型 (VerifiableGFF)，模擬純 CNN 行為，
  並與完整 Transformer 版本在相同測試集上進行對比，繪製：
    - 整體 Accuracy vs SNR 對比折線圖
    - 兩者的參數量對比橫條圖

獨立執行：
  python 05_cnn_vs_transformer.py --weights <model.pth> --data <RML2016.10a_dict.pkl>
"""

import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn

from config import MODEL_CFG, SEED, base_parser
from model import (ConvEncoder_IQ, GatingNetwork, GatedFusionFormer,
                   LayerNorm, Squeeze)
try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath
from utils import build_loader, ensure_dir, load_model, load_rml_data, set_seed


# ---------------------------------------------------------------------------
# 支援「注意力開關」的 Transformer 塊，用於隔離 CNN 貢獻
# ---------------------------------------------------------------------------

class VerifiableTransformerBlock(nn.Module):
    """與 FusionTransformerBlock 相同，但 forward 接受 disable_attn 旗標。"""

    def __init__(self, dim: int, num_heads: int = 4,
                 mlp_ratio: float = 4., drop: float = 0.1,
                 drop_path: float = 0.):
        super().__init__()
        self.norm1        = LayerNorm(dim, data_format="channels_first")
        self.conv_encoder = ConvEncoder_IQ(dim=dim, drop_path=drop_path,
                                           drop_rate=drop)
        self.norm2 = LayerNorm(dim, data_format="channels_first")
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                           dropout=drop, batch_first=True)
        self.norm3 = LayerNorm(dim, data_format="channels_first")
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(dim, hidden, 1), nn.GELU(), nn.Dropout(drop),
            nn.Conv1d(hidden, dim, 1), nn.Dropout(drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor,
                disable_attn: bool = False) -> torch.Tensor:
        x = x + self.drop_path(self.conv_encoder(self.norm1(x)))
        if not disable_attn:
            x_t = self.norm2(x).permute(0, 2, 1)
            attn_out, _ = self.attn(x_t, x_t, x_t)
            x = x + self.drop_path(attn_out.permute(0, 2, 1))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class VerifiableGFF(nn.Module):
    """GatedFusionFormer 的可驗證版，用於隔離注意力貢獻。"""

    def __init__(self, embed_dim: int = 96, num_classes: int = 11,
                 stft_time_steps: int = 128, depth: int = 4,
                 num_heads: int = 4):
        super().__init__()
        self.embed_iq = nn.Conv1d(2, embed_dim, kernel_size=7,
                                  padding=3, bias=False)
        self.embed_stft = nn.Sequential(
            nn.Conv2d(1, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2), nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, stft_time_steps)),
            Squeeze(2),
        )
        self.embed_std     = nn.Conv1d(2, embed_dim, kernel_size=7,
                                       padding=3, bias=False)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(stft_time_steps)
        self.gating_network = GatingNetwork(input_dim=embed_dim)
        self.blocks = nn.ModuleList([
            VerifiableTransformerBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.norm    = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear(embed_dim, num_classes)

    def forward(self, x_iq, x_stft, x_std,
                disable_attn: bool = False):
        feat_iq   = self.adaptive_pool(self.embed_iq(x_iq))
        feat_stft = self.embed_stft(x_stft)
        feat_std  = self.adaptive_pool(self.embed_std(x_std))

        weights = self.gating_network(feat_iq, feat_stft, feat_std)
        w_iq, w_stft, w_std = (weights[:, i:i+1] for i in range(3))
        fused = (feat_iq   * w_iq.unsqueeze(-1) +
                 feat_stft * w_stft.unsqueeze(-1) +
                 feat_std  * w_std.unsqueeze(-1))

        x = fused
        for blk in self.blocks:
            x = blk(x, disable_attn=disable_attn)
        x = self.avgpool(self.norm(x)).squeeze(-1)
        return self.head(x), weights


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _infer(model, loader, device, disable_attn: bool = False):
    all_preds, all_labels, all_snrs = [], [], []
    with torch.no_grad():
        for iq, stft, std, labels, snr_vals in tqdm(
                loader, desc=f"  attn={'on' if not disable_attn else 'off'}",
                leave=False):
            iq, stft, std = iq.to(device), stft.to(device), std.to(device)
            logits, _ = model(iq, stft, std, disable_attn=disable_attn)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snr_vals.numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_snrs)


def run(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)

    print("Loading data ...")
    X, Y, Z = load_rml_data(args.data)
    loader = build_loader(X, Y, Z, batch_size=args.batch_size)
    snrs = sorted(np.unique(Z).tolist())

    # 加載標準模型，再複製參數到 VerifiableGFF
    print("Loading model ...")
    std_model = load_model(args.weights, device=device)
    std_state = std_model.state_dict()

    ver_model = VerifiableGFF(**MODEL_CFG).to(device)
    # 將相同名稱的參數直接複製 (backbone → blocks)
    ver_state = ver_model.state_dict()
    for k, v in std_state.items():
        # backbone.0.xxx → blocks.0.xxx
        k_ver = k.replace("backbone.", "blocks.", 1) if k.startswith("backbone.") else k
        if k_ver in ver_state and ver_state[k_ver].shape == v.shape:
            ver_state[k_ver] = v
    ver_model.load_state_dict(ver_state)
    ver_model.eval()

    n_transformer = _count_params(std_model)
    n_cnn         = _count_params(ver_model)
    print(f"Transformer params : {n_transformer:,}")
    print(f"CNN (attn-off) params : {n_cnn:,}")

    # 推理
    print("\nRunning Transformer (full) ...")
    preds_t, labels_t, snrs_t = _infer(ver_model, loader, device, disable_attn=False)
    print("Running CNN-only (attention disabled) ...")
    preds_c, labels_c, snrs_c = _infer(ver_model, loader, device, disable_attn=True)

    # --- 1. Accuracy vs SNR 對比 ---
    def snr_acc(preds, labels, snr_arr):
        return [accuracy_score(labels[snr_arr == s], preds[snr_arr == s])
                for s in snrs]

    plt.figure(figsize=(12, 7))
    plt.plot(snrs, snr_acc(preds_t, labels_t, snrs_t),
             marker="o", linewidth=2, label="Transformer (full)")
    plt.plot(snrs, snr_acc(preds_c, labels_c, snrs_c),
             marker="s", linewidth=2, linestyle="--", label="CNN-only (attn off)")
    plt.title("CNN Backbone vs. Transformer: Accuracy vs. SNR")
    plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy")
    plt.ylim(0, 1.05); plt.xticks(snrs, rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cnn_vs_transformer_accuracy.png"), dpi=150)
    plt.close()
    print("Saved: cnn_vs_transformer_accuracy.png")

    # --- 2. 整體準確率對比 ---
    acc_t = accuracy_score(labels_t, preds_t)
    acc_c = accuracy_score(labels_c, preds_c)
    labels_bar  = ["Transformer\n(full)", "CNN-only\n(attn off)"]
    values_bar  = [acc_t, acc_c]
    colors_bar  = ["steelblue", "salmon"]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels_bar, values_bar, color=colors_bar, width=0.5)
    for bar, val in zip(bars, values_bar):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.005, f"{val:.4f}", ha="center", fontsize=11)
    plt.ylim(0, 1.1)
    plt.title("Overall Accuracy: CNN vs. Transformer")
    plt.ylabel("Accuracy"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cnn_vs_transformer_overall.png"), dpi=150)
    plt.close()
    print("Saved: cnn_vs_transformer_overall.png")


if __name__ == "__main__":
    parser = base_parser("GFF CNN vs Transformer 對比")
    run(parser.parse_args())
