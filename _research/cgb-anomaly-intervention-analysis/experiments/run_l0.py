"""L0 single-domain anomaly detection experiment built around TranAD."""

from __future__ import annotations

import argparse
import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.data.features import prepare_pair_feature_artifacts
from src.data.loader import find_paired_files
from src.data.processing import process_pair
from src.data.sequence import create_sequences
from src.evaluate import compute_anomaly_metrics, compute_calibration_metrics, compute_stability_metrics, create_synthetic_test_set, measure_efficiency
from src.explain import analyze_residuals, calculate_permutation_importance, compute_gradient_importance, predict_with_mc_dropout
from src.models.tranad import TranADConfig, build_tranad
from src.utils import set_seed


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_l0_config(config_path: str) -> dict:
    config_file = Path(config_path).resolve()
    model_config = load_config(str(config_file))
    base_config = {}
    base_reference = model_config.get("base_config")
    if base_reference:
        base_path = (config_file.parent / base_reference).resolve()
        base_config = load_config(str(base_path))
    return deep_merge(base_config, model_config)


def resolve_device(device_name: str) -> str:
    normalized = str(device_name or "cpu").lower()
    if normalized in {"cuda", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return normalized


def resolve_project_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def build_sequence_loader(X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def infer_feature_names(processed_info: dict) -> list[str]:
    feature_preprocessor = processed_info.get("feature_preprocessor")
    if feature_preprocessor is not None and hasattr(feature_preprocessor, "get_feature_names_out"):
        return [str(name).split("__", 1)[-1] for name in feature_preprocessor.get_feature_names_out()]
    transformed_count = int(processed_info.get("feature_count_transformed", 0))
    feature_columns = list(processed_info.get("feature_columns", []))
    if transformed_count == len(feature_columns):
        return feature_columns
    return [f"feature_{index}" for index in range(transformed_count)]


def select_pair(pairs: list[dict], pair_name: str | None) -> dict:
    if pair_name:
        for pair in pairs:
            if pair.get("name") == pair_name:
                return pair
        raise FileNotFoundError(f"Data pair not found: {pair_name}")
    if not pairs:
        raise FileNotFoundError("No train/exam pairs were found")
    return pairs[0]


def prepare_l0_data(config: dict) -> Tuple[np.ndarray, np.ndarray, list[str], str]:
    project_root = Path(__file__).resolve().parents[1]
    path_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    feature_cfg = config.get("features", {})
    training_cfg = config.get("training", {})

    raw_dir = data_cfg.get("raw_dir") or path_cfg.get("raw", "data/raw")
    data_dir = resolve_project_path(project_root, raw_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {data_dir}")

    pairs = find_paired_files(str(data_dir))
    pair = select_pair(pairs, data_cfg.get("pair_name"))
    pair_name = str(pair.get("name"))
    LOGGER.info("Using pair: %s", pair_name)

    info = process_pair(
        pair,
        data_cfg.get("time_column") or config.get("TIME_COLUMN") or config.get("data", {}).get("time_column"),
        data_cfg.get("target_column") or config.get("TARGET_COLUMN") or config.get("data", {}).get("target_column"),
        derived_feature_columns=data_cfg.get("derived_feature_columns", feature_cfg.get("derived", [])),
        categorical_columns=data_cfg.get("categorical_columns", feature_cfg.get("categorical", [])),
    )
    if not info.get("valid"):
        raise ValueError(f"Selected pair is invalid: {pair_name}")

    processed_info = prepare_pair_feature_artifacts(info, training_cfg)
    sequence_length = int(processed_info.get("sequence_length") or training_cfg.get("sequence_length", 16))
    X_train_seq, _ = create_sequences(
        processed_info["X_train_transformed_features"],
        processed_info["y_train_scaled"],
        sequence_length,
    )
    X_exam_seq, _ = create_sequences(
        processed_info["X_exam_transformed_features"],
        processed_info["y_exam_scaled"],
        sequence_length,
    )
    if len(X_train_seq) == 0 or len(X_exam_seq) == 0:
        raise ValueError("Not enough samples to build sequences. Reduce sequence_length or inspect the input data.")

    return X_train_seq, X_exam_seq, infer_feature_names(processed_info), pair_name


def build_negative_batch(batch: torch.Tensor, noise_scale: float) -> torch.Tensor:
    if batch.size(0) == 1:
        negative = torch.roll(batch, shifts=1, dims=1)
    else:
        order = torch.randperm(batch.size(0), device=batch.device)
        negative = batch[order]
    if noise_scale > 0:
        negative = negative + noise_scale * torch.randn_like(negative)
    return negative


def train_l0_model(X_train: np.ndarray, config: dict, seed: int = 42) -> Tuple[torch.nn.Module, dict]:
    set_seed(seed)
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    device = resolve_device(training_cfg.get("device", model_cfg.get("device", "cpu")))
    batch_size = int(training_cfg.get("batch_size", 64))
    train_loader = build_sequence_loader(X_train, batch_size=batch_size, shuffle=True)

    model = build_tranad(
        feature_dim=int(X_train.shape[-1]),
        config=TranADConfig(
            d_model=int(model_cfg.get("d_model", 128)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("num_layers", 3)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            device=device,
            adversarial_weight=float(model_cfg.get("adversarial_weight", 0.5)),
            reconstruction_weight=float(model_cfg.get("reconstruction_weight", 1.0)),
            discriminator_hidden=int(model_cfg.get("discriminator_hidden", 64)),
        ),
        device=device,
    )

    generator_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("discriminator.")]
    discriminator_parameters = list(model.discriminator.parameters())
    generator_optimizer = optim.Adam(generator_parameters, lr=float(training_cfg.get("learning_rate", 1e-3)))
    discriminator_optimizer = optim.Adam(
        discriminator_parameters,
        lr=float(training_cfg.get("discriminator_learning_rate", training_cfg.get("learning_rate", 1e-3))),
    )

    history = {"generator_loss": [], "discriminator_loss": []}
    noise_scale = float(training_cfg.get("negative_noise_scale", 0.05))

    for epoch in range(int(training_cfg.get("num_epochs", 20))):
        model.train()
        generator_loss_total = 0.0
        discriminator_loss_total = 0.0
        batches = 0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            negative_batch = build_negative_batch(x_batch, noise_scale)

            discriminator_optimizer.zero_grad()
            with torch.no_grad():
                normal_latent = model.encode(x_batch)
                negative_latent = model.encode(negative_batch)
            normal_probability = model.discriminator(normal_latent.detach())
            negative_probability = model.discriminator(negative_latent.detach())
            discriminator_loss, _ = model.loss_fn.discriminator_loss(normal_probability, negative_probability)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            output = model(x_batch)
            generator_loss, _ = model.loss_fn.generator_loss(x_batch, output["reconstruction"], output["normal_probability"])
            generator_loss.backward()
            generator_optimizer.step()

            generator_loss_total += float(generator_loss.item())
            discriminator_loss_total += float(discriminator_loss.item())
            batches += 1

        history["generator_loss"].append(generator_loss_total / max(1, batches))
        history["discriminator_loss"].append(discriminator_loss_total / max(1, batches))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            LOGGER.info(
                "Epoch %s/%s generator_loss=%.6f discriminator_loss=%.6f",
                epoch + 1,
                int(training_cfg.get("num_epochs", 20)),
                history["generator_loss"][-1],
                history["discriminator_loss"][-1],
            )

    return model, history


def collect_outputs(model: torch.nn.Module, X: np.ndarray, batch_size: int) -> dict:
    loader = build_sequence_loader(X, batch_size=batch_size, shuffle=False)
    scores = []
    reconstructions = []
    normal_probability = []
    model.eval()
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(next(model.parameters()).device)
            output = model(x_batch)
            scores.append(model.compute_anomaly_score(output, x_batch).detach().cpu().numpy())
            reconstructions.append(output["reconstruction"].detach().cpu().numpy())
            normal_probability.append(output["normal_probability"].detach().cpu().numpy().reshape(-1))
    return {
        "raw_scores": np.concatenate(scores, axis=0),
        "reconstruction": np.concatenate(reconstructions, axis=0),
        "normal_probability": np.concatenate(normal_probability, axis=0),
    }


def normalize_scores(raw_scores: np.ndarray, reference_scores: np.ndarray) -> np.ndarray:
    score_min = float(np.min(reference_scores))
    score_max = float(np.max(reference_scores))
    scaled = (raw_scores - score_min) / (score_max - score_min + 1e-8)
    return np.clip(scaled, 0.0, 1.0)


def build_score_predictor(model: torch.nn.Module, reference_scores: np.ndarray, batch_size: int):
    def predict_fn(X: np.ndarray) -> np.ndarray:
        outputs = collect_outputs(model, X, batch_size=batch_size)
        return normalize_scores(outputs["raw_scores"], reference_scores)

    return predict_fn


def maybe_compute_permutation_importance(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Iterable[str],
    reference_scores: np.ndarray,
    threshold: float,
    batch_size: int,
) -> dict:
    if len(X) == 0:
        return {}
    predict_fn = build_score_predictor(model, reference_scores, batch_size=batch_size)
    metric_fn = lambda y_true, y_score: compute_anomaly_metrics(y_true, y_score, threshold)["auroc"]
    importances = calculate_permutation_importance(predict_fn, X, y, metric_fn)
    return {name: float(score) for name, score in zip(feature_names, importances)}


def dry_run(config: dict) -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = resolve_project_path(project_root, config.get("data", {}).get("raw_dir") or config.get("paths", {}).get("raw", "data/raw"))
    print("L0 configuration loaded")
    print(f"Raw data dir: {raw_dir}")
    if raw_dir.exists():
        pairs = find_paired_files(str(raw_dir))
        print(f"Discovered pairs: {len(pairs)}")
        if pairs:
            print("Available pairs:", ", ".join(pair.get("name", "") for pair in pairs))
    else:
        print("Raw data dir does not exist yet")


def main(config_path: str) -> dict:
    config = load_l0_config(config_path)
    project_root = Path(__file__).resolve().parents[1]
    output_cfg = config.get("output", {})
    report_path = resolve_project_path(project_root, output_cfg.get("report_path", "results/l0_report.json"))
    figures_dir = resolve_project_path(project_root, output_cfg.get("figures_dir", "results/l0_figures"))
    model_path = resolve_project_path(project_root, output_cfg.get("model_path", "results/l0_model.pt"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    base_seed = int(config.get("seed", 42))
    set_seed(base_seed)

    X_train_seq, X_exam_seq, feature_names, pair_name = prepare_l0_data(config)
    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    batch_size = int(training_cfg.get("batch_size", 64))

    start_time = time.perf_counter()
    model, history = train_l0_model(X_train_seq, config, seed=base_seed)
    training_time = time.perf_counter() - start_time

    train_outputs = collect_outputs(model, X_train_seq, batch_size=batch_size)
    exam_outputs = collect_outputs(model, X_exam_seq, batch_size=batch_size)
    train_scores = normalize_scores(train_outputs["raw_scores"], train_outputs["raw_scores"])
    exam_scores = normalize_scores(exam_outputs["raw_scores"], train_outputs["raw_scores"])

    threshold_value = evaluation_cfg.get("anomaly_threshold")
    if threshold_value is None:
        threshold = float(np.quantile(train_scores, float(evaluation_cfg.get("threshold_quantile", 0.95))))
    else:
        threshold = float(threshold_value)
    exam_labels = (exam_scores >= threshold).astype(int)

    synthetic_metrics = {}
    calibration_metrics = {}
    feature_importance = {}
    synthetic_labels = np.array([], dtype=int)
    synthetic_scores = np.array([], dtype=float)

    if evaluation_cfg.get("synthetic_validation", True):
        X_synthetic, synthetic_labels = create_synthetic_test_set(
            X_exam_seq,
            anomaly_ratio=float(evaluation_cfg.get("synthetic_anomaly_ratio", 0.1)),
            types=evaluation_cfg.get("synthetic_types", ["spike", "level_shift", "trend_change"]),
            random_seed=base_seed,
        )
        synthetic_outputs = collect_outputs(model, X_synthetic, batch_size=batch_size)
        synthetic_scores = normalize_scores(synthetic_outputs["raw_scores"], train_outputs["raw_scores"])
        synthetic_metrics = compute_anomaly_metrics(synthetic_labels, synthetic_scores, threshold)
        calibration_metrics = compute_calibration_metrics(synthetic_labels, synthetic_scores)
        feature_importance = maybe_compute_permutation_importance(
            model,
            X_synthetic,
            synthetic_labels,
            feature_names,
            train_outputs["raw_scores"],
            threshold,
            batch_size,
        )

    stability_results = []
    for seed in evaluation_cfg.get("stability_seeds", []):
        if int(seed) == base_seed:
            if synthetic_metrics:
                stability_results.append(synthetic_metrics)
            continue
        seed_model, _ = train_l0_model(X_train_seq, config, seed=int(seed))
        if synthetic_labels.size:
            seed_train_outputs = collect_outputs(seed_model, X_train_seq, batch_size=batch_size)
            synthetic_outputs = collect_outputs(seed_model, X_synthetic, batch_size=batch_size)
            synthetic_scores_seed = normalize_scores(synthetic_outputs["raw_scores"], seed_train_outputs["raw_scores"])
            stability_results.append(compute_anomaly_metrics(synthetic_labels, synthetic_scores_seed, threshold))

    stability_metrics = compute_stability_metrics(stability_results)

    exam_loader = build_sequence_loader(X_exam_seq, batch_size=batch_size, shuffle=False)
    mc_mean, mc_std = predict_with_mc_dropout(
        model,
        exam_loader,
        device=str(next(model.parameters()).device),
        n_samples=int(evaluation_cfg.get("mc_samples", 20)),
    )
    gradient_importance = compute_gradient_importance(
        model,
        torch.from_numpy(X_exam_seq[: min(len(X_exam_seq), batch_size)]).float(),
        device=str(next(model.parameters()).device),
    )
    residual_report = analyze_residuals(X_exam_seq, exam_outputs["reconstruction"])
    efficiency_metrics = measure_efficiency(
        model,
        sample_batch=torch.from_numpy(X_exam_seq[: min(len(X_exam_seq), batch_size)]).float(),
        wall_time_seconds=training_time,
    )

    torch.save(model.state_dict(), model_path)

    results = {
        "experiment": "L0 (Single-Domain Anomaly Detection)",
        "data": {
            "pair_name": pair_name,
            "n_train_sequences": int(len(X_train_seq)),
            "n_exam_sequences": int(len(X_exam_seq)),
            "feature_count": int(X_train_seq.shape[-1]),
        },
        "training": {
            "seed": base_seed,
            "threshold": threshold,
            "history": history,
        },
        "anomaly_detection": {
            "exam_anomaly_ratio": float(np.mean(exam_labels)),
            "exam_score_mean": float(np.mean(exam_scores)),
            "exam_score_std": float(np.std(exam_scores)),
        },
        "synthetic_validation": synthetic_metrics,
        "calibration": calibration_metrics,
        "stability": stability_metrics,
        "efficiency": efficiency_metrics,
        "uncertainty": {
            "mean_score": float(np.mean(mc_mean)),
            "mean_uncertainty": float(np.mean(mc_std)),
        },
        "residuals": residual_report,
        "feature_importance": feature_importance,
        "gradient_importance": {name: float(score) for name, score in zip(feature_names, gradient_importance)},
        "artifacts": {
            "report_path": str(report_path),
            "model_path": str(model_path),
            "figures_dir": str(figures_dir),
        },
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Saved L0 report to %s", report_path)
    LOGGER.info("Saved model checkpoint to %s", model_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/l0.yaml")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    configuration = load_l0_config(arguments.config)
    if arguments.dry_run:
        dry_run(configuration)
    else:
        main(arguments.config)