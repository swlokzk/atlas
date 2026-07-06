"""
01_evaluate.py — 基礎模型評估

功能：
  1. 加載模型與 RML2016.10a 資料集
  2. 對全部資料執行推理，收集預測結果
  3. 輸出：
     - 原始計數混淆矩陣
     - 歸一化混淆矩陣 (百分比)
     - 整體 Accuracy vs SNR 折線圖

獨立執行：
  python 01_evaluate.py --weights <model.pth> --data <RML2016.10a_dict.pkl>
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import torch

from config import CLASSES, SEED, base_parser
from utils import build_loader, ensure_dir, load_model, load_rml_data, set_seed


def run(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)

    # 1. 加載資料與模型
    print("Loading data ...")
    X, Y, Z = load_rml_data(args.data)
    loader = build_loader(X, Y, Z, batch_size=args.batch_size)
    snrs = sorted(np.unique(Z).tolist())

    print("Loading model ...")
    model = load_model(args.weights, device=device)

    # 2. 推理
    all_preds, all_labels, all_snrs = [], [], []
    with torch.no_grad():
        for iq, stft, std, labels, snr_vals in tqdm(loader, desc="Inference"):
            iq, stft, std = iq.to(device), stft.to(device), std.to(device)
            logits, _ = model(iq, stft, std)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snr_vals.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_snrs   = np.array(all_snrs)

    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall accuracy: {overall_acc:.4f}")

    # 3a. 混淆矩陣 (計數)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix (Counts)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_counts.png"), dpi=150)
    plt.close()
    print("Saved: confusion_matrix_counts.png")

    # 3b. 混淆矩陣 (歸一化)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                vmin=0, vmax=1)
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_normalized.png"), dpi=150)
    plt.close()
    print("Saved: confusion_matrix_normalized.png")

    # 3c. Accuracy vs SNR
    per_snr_acc = {
        snr: accuracy_score(all_labels[all_snrs == snr],
                            all_preds[all_snrs == snr])
        for snr in snrs
    }
    plt.figure(figsize=(10, 6))
    plt.plot(list(per_snr_acc.keys()), list(per_snr_acc.values()),
             marker="o", linestyle="-", label="GatedFusionFormer v3.0")
    plt.title("Recognition Accuracy vs. SNR")
    plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy")
    plt.ylim(0, 1.05); plt.xticks(snrs, rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_snr.png"), dpi=150)
    plt.close()
    print("Saved: accuracy_vs_snr.png")


if __name__ == "__main__":
    parser = base_parser("GFF 基礎模型評估")
    run(parser.parse_args())
