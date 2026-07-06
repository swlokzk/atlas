"""
04_gating_weights.py — 門控網絡權重分析

功能：
  對每個 SNR 等級，統計 GatingNetwork 輸出的平均融合權重，
  繪製「模態重要性 vs SNR」折線圖，直觀呈現模型在不同信噪比下
  對各模態的依賴程度。

獨立執行：
  python 04_gating_weights.py --weights <model.pth> --data <RML2016.10a_dict.pkl>
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch

from config import SEED, base_parser
from utils import build_loader, ensure_dir, load_model, load_rml_data, set_seed


def run(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)

    print("Loading data ...")
    X, Y, Z = load_rml_data(args.data)
    loader = build_loader(X, Y, Z, batch_size=args.batch_size)

    print("Loading model ...")
    model = load_model(args.weights, device=device)

    # 收集每個樣本的門控權重與對應 SNR
    all_weights = []
    all_snrs    = []

    with torch.no_grad():
        for iq, stft, std, _, snr_vals in tqdm(loader, desc="Extracting gating weights"):
            iq, stft, std = iq.to(device), stft.to(device), std.to(device)
            _, weights = model(iq, stft, std)
            all_weights.append(weights.cpu().numpy())
            all_snrs.extend(snr_vals.numpy())

    all_weights = np.vstack(all_weights)   # (N, 3)
    all_snrs    = np.array(all_snrs)
    snrs        = sorted(np.unique(all_snrs).tolist())

    # 每個 SNR 的平均權重
    iq_avg, stft_avg, std_avg = [], [], []
    for snr in snrs:
        mask = all_snrs == snr
        avg  = all_weights[mask].mean(axis=0)
        iq_avg.append(avg[0])
        stft_avg.append(avg[1])
        std_avg.append(avg[2])

    # 繪圖
    plt.figure(figsize=(12, 7))
    plt.plot(snrs, iq_avg,   marker="o", linewidth=2, label="IQ Modality")
    plt.plot(snrs, stft_avg, marker="s", linewidth=2, label="STFT Modality")
    plt.plot(snrs, std_avg,  marker="^", linewidth=2, label="S-TD Modality")
    plt.title("Modal Importance vs. SNR (Gating Weights)")
    plt.xlabel("SNR (dB)"); plt.ylabel("Average Gating Weight")
    plt.ylim(0, 1.0); plt.xticks(snrs, rotation=45)
    plt.grid(True, linestyle="--", alpha=0.7); plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gating_weights_vs_snr.png"), dpi=150)
    plt.close()
    print("Saved: gating_weights_vs_snr.png")


if __name__ == "__main__":
    parser = base_parser("GFF 門控網絡權重分析")
    run(parser.parse_args())
