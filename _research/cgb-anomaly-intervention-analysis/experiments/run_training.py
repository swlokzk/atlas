"""Minimal runner for training experiment.

Usage: python experiments/run_training.py --config configs/anomaly_transformer.yaml
"""
import argparse
from copy import deepcopy
from pathlib import Path

from src.config import load_config
from src.data.loader import find_paired_files
from src.data.processing import process_pair
from src.data.features import prepare_pair_feature_artifacts
from src.data.sequence import create_sequences
from src.data.dataset import build_dataloaders
from src.train.loop import train_epoch, evaluate_model_simple
from src.models.anomaly_transformer import AnomalyTransformerConfig, build_abnormal_transformer
from src.models.tranad import TranADConfig, build_tranad
from src.models.patchtst import PatchTSTConfig, build_patchtst
from src.models.dada import DADAConfig, build_dada
from src.models.ts2vec import TS2VecConfig, build_ts2vec
from src.models.timesnet import TimesNetConfig, build_timesnet
from src.models.dann import DANNConfig, build_dann


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_experiment_config(config_path: str) -> dict:
    config_file = Path(config_path).resolve()
    model_config = load_config(str(config_file))
    base_config = {}
    base_reference = model_config.get("base_config")
    if base_reference:
        base_path = (config_file.parent / base_reference).resolve()
        base_config = load_config(str(base_path))
    return deep_merge(base_config, model_config)


def build_model_from_config(feature_dim: int, model_cfg: dict, device: str):
    model_name = str(model_cfg.get("name", "abnormal_transformer")).lower()
    if model_name in {"abnormal_transformer", "anomaly_transformer", "transformer"}:
        config = AnomalyTransformerConfig(
            d_model=model_cfg.get("d_model", 64),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 2),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1),
            device=device,
            window_size=model_cfg.get("window_size", 60),
        )
        return build_abnormal_transformer(feature_dim, config=config, device=device)
    if model_name == "tranad":
        config = TranADConfig(
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 3),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1),
            device=device,
            adversarial_weight=model_cfg.get("adversarial_weight", 1.0),
        )
        return build_tranad(feature_dim, config=config, device=device)
    if model_name == "patchtst":
        config = PatchTSTConfig(
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 3),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1),
            device=device,
            patch_len=model_cfg.get("patch_len", 8),
            stride=model_cfg.get("stride", 4),
        )
        return build_patchtst(feature_dim, config=config, device=device)
    if model_name == "dada":
        config = DADAConfig(
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 3),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1),
            device=device,
            bottleneck_dim=model_cfg.get("bottleneck_dim", 64),
        )
        return build_dada(feature_dim, config=config, device=device)
    if model_name == "ts2vec":
        config = TS2VecConfig(
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 3),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1),
            device=device,
            projection_dim=model_cfg.get("projection_dim", 128),
        )
        return build_ts2vec(feature_dim, config=config, device=device)
    if model_name == "timesnet":
        config = TimesNetConfig(
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 3),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1),
            device=device,
            num_kernels=model_cfg.get("num_kernels", 6),
        )
        return build_timesnet(feature_dim, config=config, device=device)
    if model_name == "dann":
        config = DANNConfig(
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 3),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1),
            device=device,
            domain_loss_weight=model_cfg.get("domain_loss_weight", 1.0),
        )
        return build_dann(feature_dim, config=config, device=device)
    raise ValueError(f"Unsupported model name: {model_name}")


def main(config_path: str):
    cfg = load_experiment_config(config_path)
    data_cfg = cfg.get("data", {})
    feature_cfg = cfg.get("features", {})
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    project_root = Path(__file__).resolve().parents[1]
    path_cfg = cfg.get("paths", {})
    data_dir = project_root / path_cfg.get("raw", data_cfg.get("raw", cfg.get("DATA_DIR", "data/raw")))
    pairs = find_paired_files(data_dir)
    if not pairs:
        print("No data pairs found in:", data_dir)
        return
    pair_name = data_cfg.get("pair_name")
    pair = next((candidate for candidate in pairs if candidate.get("name") == pair_name), pairs[0])
    info = process_pair(
        pair,
        data_cfg.get("time_column") or cfg.get("TIME_COLUMN"),
        data_cfg.get("target_column") or cfg.get("TARGET_COLUMN"),
        derived_feature_columns=data_cfg.get("derived_feature_columns", feature_cfg.get("derived", [])),
        categorical_columns=data_cfg.get("categorical_columns", feature_cfg.get("categorical", [])),
    )
    if not info.get("valid"):
        print("Pair invalid")
        return
    processed_info = prepare_pair_feature_artifacts(info, training_cfg)
    X_train_seq, y_train_seq = create_sequences(
        processed_info["X_train_transformed_features"],
        processed_info["y_train_scaled"],
        int(processed_info["sequence_length"] or training_cfg.get("sequence_length", 16)),
    )
    X_exam_seq, y_exam_seq = create_sequences(
        processed_info["X_exam_transformed_features"],
        processed_info["y_exam_scaled"],
        int(processed_info["sequence_length"] or training_cfg.get("sequence_length", 16)),
    )
    train_loader, val_loader = build_dataloaders(
        X_train_seq,
        y_train_seq,
        int(processed_info["batch_size"] or training_cfg.get("batch_size", 32)),
        X_exam_seq,
        y_exam_seq,
    )

    device = str(training_cfg.get("device", model_cfg.get("device", "cpu")))
    model = build_model_from_config(int(processed_info["feature_count_transformed"]), model_cfg, device)
    import torch.nn as nn
    import torch.optim as optim

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(training_cfg.get("learning_rate", cfg.get("LEARNING_RATE", 1e-3))))

    for epoch in range(int(training_cfg.get("num_epochs", cfg.get("NUM_EPOCHS", 1)))):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = None
        if val_loader:
            val_loss = evaluate_model_simple(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}", f"val_loss={val_loss:.4f}" if val_loss else "")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/anomaly_transformer.yaml")
    args = p.parse_args()
    main(args.config)
