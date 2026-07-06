"""
03_ablation.py — 模態消融實驗

功能：
  1. 單模態消融 (IQ / STFT / S-TD / All)：
     對每種配置計算 Accuracy vs SNR，並繪製對比折線圖。
  2. 成對模態消融 (IQ+STFT / IQ+S-TD / STFT+S-TD)：
     對比各成對配置的 Accuracy vs SNR。
  3. 高 SNR 下各配置易混淆類別比較。

獨立執行：
  python 03_ablation.py --weights <model.pth> --data <RML2016.10a_dict.pkl>
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch

from config import CLASSES, SEED, base_parser, HIGH_SNR
from utils import build_loader, ensure_dir, load_model, load_rml_data, set_seed


def _infer(model, loader, device, active_modalities):
    all_preds, all_labels, all_snrs = [], [], []
    with torch.no_grad():
        for iq, stft, std, labels, snr_vals in tqdm(
                loader, desc=f"  {'+'.join(active_modalities)}", leave=False):
            iq, stft, std = iq.to(device), stft.to(device), std.to(device)
            logits, _ = model(iq, stft, std, active_modalities=active_modalities)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snr_vals.numpy())
    return (np.array(all_preds), np.array(all_labels), np.array(all_snrs))


def _snr_accuracy(preds, labels, snrs, snr_list):
    return [
        accuracy_score(labels[snrs == s], preds[snrs == s])
        for s in snr_list
    ]


def _plot_snr_curves(results, snrs, title, fname, out_dir):
    plt.figure(figsize=(12, 7))
    for name, (preds, labels, snr_arr) in results.items():
        plt.plot(snrs, _snr_accuracy(preds, labels, snr_arr, snrs),
                 marker=".", label=name)
    plt.title(title)
    plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy")
    plt.ylim(0, 1.05); plt.xticks(snrs, rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def run(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)

    print("Loading data ...")
    X, Y, Z = load_rml_data(args.data)
    loader = build_loader(X, Y, Z, batch_size=args.batch_size)
    snrs = sorted(np.unique(Z).tolist())

    print("Loading model ...")
    model = load_model(args.weights, device=device)

    # --- 1. 單模態消融 ---
    print("\n[1/2] Single-modality ablation ...")
    single_configs = {
        "IQ Only":        ["iq"],
        "STFT Only":      ["stft"],
        "S-TD Only":      ["std"],
        "All Modalities": ["iq", "stft", "std"],
    }
    single_results = {
        name: _infer(model, loader, device, mods)
        for name, mods in single_configs.items()
    }
    _plot_snr_curves(
        single_results, snrs,
        "Single-Modality Ablation: Accuracy vs. SNR",
        "ablation_single_modality.png", out_dir
    )

    # --- 2. 成對模態消融 ---
    print("\n[2/2] Pairwise modality ablation ...")
    pair_configs = {
        "IQ + STFT":   ["iq", "stft"],
        "IQ + S-TD":   ["iq", "std"],
        "STFT + S-TD": ["stft", "std"],
        "All":         ["iq", "stft", "std"],
    }
    pair_results = {
        name: _infer(model, loader, device, mods)
        for name, mods in pair_configs.items()
    }
    _plot_snr_curves(
        pair_results, snrs,
        "Pairwise Modality Ablation: Accuracy vs. SNR",
        "ablation_pairwise_modality.png", out_dir
    )

    # --- 3. 高 SNR 易混淆類別對比 ---
    configs_to_compare = ["IQ Only", "All Modalities"]
    confusion_focus_keys = ["QAM16/64", "AM-DSB/WBFM", "8PSK/QPSK", "CPFSK/GFSK"]
    pair_map = [
        (("QAM16", "QAM64"), ("QAM64", "QAM16")),
        (("AM-DSB", "WBFM"), ("WBFM", "AM-DSB")),
        (("8PSK",  "QPSK"),  ("QPSK", "8PSK")),
        (("CPFSK", "GFSK"),  ("GFSK", "CPFSK")),
    ]

    fig, axes = plt.subplots(1, len(configs_to_compare),
                             figsize=(10 * len(configs_to_compare), 6))
    fig.suptitle(f"Confused Categories at SNR = {HIGH_SNR} dB", fontsize=15)

    for ax, name in zip(axes, configs_to_compare):
        preds, labels, snr_arr = single_results[name]
        mask = snr_arr == HIGH_SNR
        ep = [(CLASSES[labels[mask][i]], CLASSES[preds[mask][i]])
              for i in np.where(labels[mask] != preds[mask])[0]]
        counts = [ep.count(a) + ep.count(b) for a, b in pair_map]
        ax.bar(confusion_focus_keys, counts,
               color=plt.cm.Paired(np.arange(len(counts))))
        for k, v in enumerate(counts):
            ax.text(k, v + 2, str(v), ha="center")
        ax.set_title(name); ax.set_ylabel("# Misclassifications")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_confused_categories.png"), dpi=150)
    plt.close()
    print("Saved: ablation_confused_categories.png")


if __name__ == "__main__":
    parser = base_parser("GFF 模態消融實驗")
    run(parser.parse_args())
