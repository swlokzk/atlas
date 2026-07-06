"""
02_deep_analysis.py — 深度性能分析

功能：
  1. Per-class Accuracy vs SNR 折線圖
  2. 高 SNR 下易混淆類別柱狀圖
  3. t-SNE 特徵空間可視化（低、高 SNR 各一幅）

獨立執行：
  python 02_deep_analysis.py --weights <model.pth> --data <RML2016.10a_dict.pkl>
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset

from config import CLASSES, SEED, base_parser, HIGH_SNR, TSNE_SNRS, TSNE_MAX_SAMPLES
from dataset import RMLDataset
from utils import build_loader, ensure_dir, load_model, load_rml_data, set_seed


def _run_inference(model, loader, device):
    all_preds, all_labels, all_snrs = [], [], []
    with torch.no_grad():
        for iq, stft, std, labels, snr_vals in tqdm(loader, desc="Inference"):
            iq, stft, std = iq.to(device), stft.to(device), std.to(device)
            logits, _ = model(iq, stft, std)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snr_vals.numpy())
    return (np.array(all_preds), np.array(all_labels), np.array(all_snrs))


def run(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)

    print("Loading data ...")
    X, Y, Z = load_rml_data(args.data)
    full_dataset = RMLDataset(X, Y, Z)
    loader = DataLoader(full_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    snrs = sorted(np.unique(Z).tolist())

    print("Loading model ...")
    model = load_model(args.weights, device=device)

    all_preds, all_labels, all_snrs = _run_inference(model, loader, device)

    # 1. Per-class Accuracy vs SNR
    print("Plotting per-class accuracy vs SNR ...")
    plt.figure(figsize=(13, 8))
    for idx, cls_name in enumerate(CLASSES):
        per_snr = []
        for snr in snrs:
            mask = (all_snrs == snr) & (all_labels == idx)
            per_snr.append(
                accuracy_score(all_labels[mask], all_preds[mask])
                if mask.any() else 0.0
            )
        plt.plot(snrs, per_snr, marker=".", label=cls_name)
    plt.title("Per-class Accuracy vs. SNR")
    plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy")
    plt.ylim(0, 1.05); plt.xticks(snrs, rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_accuracy_vs_snr.png"), dpi=150)
    plt.close()
    print("Saved: per_class_accuracy_vs_snr.png")

    # 2. 高 SNR 易混淆類別
    mask_high = all_snrs == HIGH_SNR
    errors = np.where(all_labels[mask_high] != all_preds[mask_high])[0]
    err_pairs = [
        (CLASSES[all_labels[mask_high][i]], CLASSES[all_preds[mask_high][i]])
        for i in errors
    ]
    confusion_focus = {
        "QAM16/QAM64":  err_pairs.count(("QAM16", "QAM64"))  + err_pairs.count(("QAM64",  "QAM16")),
        "AM-DSB/WBFM":  err_pairs.count(("AM-DSB", "WBFM"))  + err_pairs.count(("WBFM",   "AM-DSB")),
        "8PSK/QPSK":    err_pairs.count(("8PSK",  "QPSK"))   + err_pairs.count(("QPSK",   "8PSK")),
        "CPFSK/GFSK":   err_pairs.count(("CPFSK", "GFSK"))   + err_pairs.count(("GFSK",   "CPFSK")),
    }
    plt.figure(figsize=(9, 5))
    bars = plt.bar(confusion_focus.keys(), confusion_focus.values(),
                   color=plt.cm.Paired(np.arange(len(confusion_focus))))
    for i, v in enumerate(confusion_focus.values()):
        plt.text(i, v + 2, str(v), ha="center")
    plt.title(f"Most Confused Categories at SNR = {HIGH_SNR} dB")
    plt.ylabel("# Misclassifications")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confused_categories_high_snr.png"), dpi=150)
    plt.close()
    print("Saved: confused_categories_high_snr.png")

    # 3. t-SNE
    print("Running t-SNE visualization ...")
    features_buf = []

    def _hook(module, inp, out):
        features_buf.append(inp[0].detach())

    handle = model.head.register_forward_hook(_hook)
    snrs_to_vis = list(TSNE_SNRS)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("t-SNE Feature Space", fontsize=16)

    for ax, snr in zip(axes, snrs_to_vis):
        indices = np.where(all_snrs == snr)[0]
        if len(indices) > TSNE_MAX_SAMPLES:
            indices = np.random.choice(indices, TSNE_MAX_SAMPLES, replace=False)
        subset_loader = DataLoader(
            Subset(full_dataset, indices),
            batch_size=args.batch_size, shuffle=False
        )
        features_buf.clear()
        with torch.no_grad():
            for iq, stft, std, _, _ in tqdm(subset_loader,
                                             desc=f"Feature extraction SNR={snr}"):
                model(iq.to(device), stft.to(device), std.to(device))

        feats  = torch.cat(features_buf).cpu().numpy()
        labels = all_labels[indices]
        feats2d = TSNE(n_components=2, perplexity=30,
                       learning_rate="auto", init="pca",
                       random_state=42).fit_transform(feats)
        ax.scatter(feats2d[:, 0], feats2d[:, 1],
                   c=labels, cmap="tab10", alpha=0.8, s=15)
        ax.set_title(f"SNR = {snr} dB")
        ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=plt.cm.tab10(j / 10.),
                       label=CLASSES[j], markersize=8)
            for j in range(len(CLASSES))
        ]
        ax.legend(handles=handles, title="Classes",
                  loc="upper right", fontsize="small")

    handle.remove()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "tsne_visualization.png"), dpi=150)
    plt.close()
    print("Saved: tsne_visualization.png")


if __name__ == "__main__":
    parser = base_parser("GFF 深度性能分析")
    run(parser.parse_args())
