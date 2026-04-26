"""
Training pipeline for Arabic ABSA with class imbalance handling and pseudo labels.

Supported models:
1. MARBERT: UBC-NLP/MARBERT
2. AraBERT: aubmindlab/bert-base-arabertv02
3. XLM-RoBERTa: xlm-roberta-base

Default model: MARBERT
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import (
    ABDataset,
    ASPECT_SENTIMENT_LABELS,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALIDATION_PATH,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    OUTPUTS_ROOT,
    build_pos_weight_tensor,
    compute_class_distribution,
    infer_column_mapping,
    load_dataframe,
    resolve_input_path,
)
from preprocess import ArabicPreprocessor


DEFAULT_MODELS = {
    "marbert": "UBC-NLP/MARBERT",
    "arabert": "aubmindlab/bert-base-arabertv02",
    "xlmr": "xlm-roberta-base",
}

DEFAULT_MODEL_ALIAS = "marbert"
DEFAULT_MODEL_NAME = DEFAULT_MODELS[DEFAULT_MODEL_ALIAS]

DEFAULT_CONFIG = {
    "max_length": 128,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "early_stopping_patience": 2,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "dropout": 0.1,
    "seed": 42,
}

DEFAULT_PSEUDO_LABEL_PATH = OUTPUTS_ROOT / "pseudo_labeled.csv"
LEGACY_PSEUDO_LABEL_PATH = OUTPUTS_ROOT / "pseudo_labeled.json"
DEFAULT_PSEUDO_LABEL_WEIGHT = 0.3
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT
TRAINING_MANIFEST_FILENAME = "training_manifest.pkl"
TRAINING_STATE_FILENAME = "training_state.pkl"
DEFAULT_TRAINING_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / TRAINING_MANIFEST_FILENAME
DEFAULT_TRAINING_STATE_PATH = DEFAULT_OUTPUT_DIR / TRAINING_STATE_FILENAME


def str2bool(value: Any) -> bool:
    """Parse flexible CLI booleans."""
    if isinstance(value, bool):
        return value

    lowered = str(value).strip().lower()

    if lowered in {"1", "true", "yes", "y", "on"}:
        return True

    if lowered in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Could not interpret boolean value: {value}")


def save_json(data: Mapping[str, Any], output_path: Path) -> None:
    """Save UTF-8 JSON with Arabic support."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def resolve_pseudo_label_input_path(path: Optional[Path]) -> Path:
    """Resolve pseudo-label paths while preserving compatibility with legacy JSON files."""
    resolved_path = resolve_input_path(path, path or DEFAULT_PSEUDO_LABEL_PATH)
    if resolved_path is not None and resolved_path.exists():
        return resolved_path

    requested_path = Path(path) if path is not None else DEFAULT_PSEUDO_LABEL_PATH
    legacy_requested = requested_path == DEFAULT_PSEUDO_LABEL_PATH or requested_path == LEGACY_PSEUDO_LABEL_PATH
    if legacy_requested:
        legacy_path = resolve_input_path(LEGACY_PSEUDO_LABEL_PATH, LEGACY_PSEUDO_LABEL_PATH)
        if legacy_path is not None and legacy_path.exists():
            return legacy_path

    return resolved_path or requested_path


def summarize_distribution_for_logging(distribution: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a compact, readable summary of class/aspect coverage."""
    aspect_distribution = dict(distribution.get("aspect_distribution", {}))
    top_aspects = sorted(
        aspect_distribution.items(),
        key=lambda item: (-float(item[1]), item[0]),
    )[:5]
    summary: Dict[str, Any] = {
        "num_samples": distribution.get("num_samples"),
        "top_aspects": top_aspects,
        "sentiment_distribution": distribution.get("sentiment_distribution", {}),
    }
    if "effective_num_samples" in distribution:
        summary["effective_num_samples"] = distribution.get("effective_num_samples")
    return summary


def print_distribution_summary(label: str, distribution: Mapping[str, Any]) -> None:
    """Print a concise distribution summary for training logs."""
    summary = summarize_distribution_for_logging(distribution)
    print(f"{label}:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def normalize_training_config(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Normalize config types so saved manifests compare cleanly across CLI runs."""
    normalized = dict(DEFAULT_CONFIG)

    for key, value in dict(config or {}).items():
        if key in DEFAULT_CONFIG:
            default_value = DEFAULT_CONFIG[key]

            if isinstance(default_value, bool):
                normalized[key] = bool(value)
            elif isinstance(default_value, int):
                normalized[key] = int(value)
            elif isinstance(default_value, float):
                normalized[key] = float(value)
            else:
                normalized[key] = value
        else:
            normalized[key] = value

    return normalized


def build_file_signature(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Capture lightweight file metadata to detect training-data changes."""
    if path is None:
        return None

    resolved_path = resolve_input_path(path, path) or path

    if not resolved_path.exists():
        return {
            "path": str(resolved_path.resolve()),
            "exists": False,
        }

    stat_result = resolved_path.stat()

    return {
        "path": str(resolved_path.resolve()),
        "exists": True,
        "size_bytes": int(stat_result.st_size),
        "modified_time_ns": int(stat_result.st_mtime_ns),
    }


def load_training_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load a saved training manifest from disk."""
    with manifest_path.open("rb") as handle:
        manifest = pickle.load(handle)

    if not isinstance(manifest, dict):
        raise ValueError(f"Training manifest must be a dictionary: {manifest_path}")

    return manifest


def load_model_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    """Load saved checkpoint metadata without constructing the model."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary: {checkpoint_path}")

    return checkpoint


def load_training_state(state_path: Path) -> Dict[str, Any]:
    """Load an in-progress resumable training state from disk."""
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")

    if not isinstance(state, dict):
        raise ValueError(f"Training state must be a dictionary: {state_path}")

    return state


def capture_random_state() -> Dict[str, Any]:
    """Capture NumPy and Torch RNG state so resumed training stays reproducible."""
    state: Dict[str, Any] = {
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    return state


def restore_random_state(state: Mapping[str, Any]) -> None:
    """Restore NumPy and Torch RNG state from a saved checkpoint."""
    numpy_rng_state = state.get("numpy_rng_state")
    if numpy_rng_state is not None:
        np.random.set_state(numpy_rng_state)

    torch_rng_state = state.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)

    cuda_rng_state_all = state.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_rng_state_all is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state_all)


def move_optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move optimizer state tensors onto the active device after resuming."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def cleanup_training_state(output_dir: Path) -> None:
    """Remove the rolling resume state once training finishes cleanly."""
    state_path = output_dir / TRAINING_STATE_FILENAME

    if state_path.exists():
        state_path.unlink()


def resolve_model_name(model_name: Optional[str]) -> str:
    """
    Resolve supported model aliases to Hugging Face model names.

    Supported aliases:
    - marbert
    - arabert
    - xlmr
    """
    if not model_name:
        return DEFAULT_MODEL_NAME

    normalized_name = str(model_name).strip()

    if not normalized_name:
        return DEFAULT_MODEL_NAME

    return DEFAULT_MODELS.get(normalized_name.lower(), normalized_name)


def infer_model_family(model_name: str) -> str:
    """Infer model family for checkpoint metadata."""
    resolved_name = resolve_model_name(model_name).lower()

    if "marbert" in resolved_name:
        return "marbert"

    if "arabert" in resolved_name:
        return "arabert"

    if "xlm-roberta" in resolved_name:
        return "xlmr"

    return "custom"


def save_training_manifest(
    output_dir: Path,
    checkpoint_path: Path,
    model_name: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    threshold: float,
    train_path: Path,
    validation_path: Path,
    use_pseudo_labels: bool = False,
    pseudo_label_path: Optional[Path] = None,
) -> Path:
    """Save reusable training metadata alongside the checkpoint."""
    manifest_path = output_dir / TRAINING_MANIFEST_FILENAME

    payload = {
        "manifest_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_status": "completed",
        "model_name": resolve_model_name(model_name),
        "model_family": infer_model_family(model_name),
        "config": normalize_training_config(config),
        "metrics": dict(metrics),
        "best_threshold": float(threshold),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "tokenizer_dir": str((output_dir / "tokenizer").resolve()),
        "label_mapping_path": str((output_dir / "label_mapping.json").resolve()),
        "class_distribution_path": str((output_dir / "class_distribution.json").resolve()),
        "train_signature": build_file_signature(train_path),
        "validation_signature": build_file_signature(validation_path),
        "use_pseudo_labels": bool(use_pseudo_labels),
        "pseudo_label_signature": (
            build_file_signature(pseudo_label_path) if use_pseudo_labels else None
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("wb") as handle:
        pickle.dump(payload, handle)

    return manifest_path


def save_training_state(
    state_path: Path,
    model: "ABSAModel",
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    model_name: str,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    train_path: Path,
    validation_path: Path,
    next_epoch: int,
    best_f1: float,
    best_threshold: float,
    best_metrics: Mapping[str, Any],
    best_threshold_history: Mapping[str, Any],
    stagnant_epochs: int,
    class_distribution: Mapping[str, Any],
    use_pseudo_labels: bool = False,
    pseudo_label_path: Optional[Path] = None,
) -> Path:
    """Persist rolling state needed to resume an interrupted training run."""
    payload = {
        "state_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_status": "in_progress",
        "model_name": resolve_model_name(model_name),
        "model_family": infer_model_family(model_name),
        "config": normalize_training_config(config),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "train_signature": build_file_signature(train_path),
        "validation_signature": build_file_signature(validation_path),
        "use_pseudo_labels": bool(use_pseudo_labels),
        "pseudo_label_signature": (
            build_file_signature(pseudo_label_path) if use_pseudo_labels else None
        ),
        "next_epoch": int(next_epoch),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "best_metrics": dict(best_metrics),
        "best_threshold_history": dict(best_threshold_history),
        "stagnant_epochs": int(stagnant_epochs),
        "class_distribution": dict(class_distribution),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        **capture_random_state(),
    }

    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, state_path)

    return state_path


def collect_training_artifact_mismatch_reasons(
    artifact: Mapping[str, Any],
    checkpoint_path: Path,
    model_name: str,
    config: Mapping[str, Any],
    train_path: Path,
    validation_path: Path,
    use_pseudo_labels: bool = False,
    pseudo_label_path: Optional[Path] = None,
) -> List[str]:
    """Collect compatibility blockers for saved manifests or resume states."""
    reasons: List[str] = []

    resolved_checkpoint_path = resolve_input_path(checkpoint_path, checkpoint_path) or checkpoint_path

    if not resolved_checkpoint_path.exists():
        reasons.append(f"Checkpoint not found at {resolved_checkpoint_path}.")

    expected_model_name = resolve_model_name(model_name)

    if str(artifact.get("model_name")) != expected_model_name:
        reasons.append("Model name changed.")

    saved_config = normalize_training_config(artifact.get("config", {}))
    current_config = normalize_training_config(config)

    if saved_config != current_config:
        reasons.append("Training configuration changed.")

    if artifact.get("train_signature") != build_file_signature(train_path):
        reasons.append("Training dataset changed.")

    if artifact.get("validation_signature") != build_file_signature(validation_path):
        reasons.append("Validation dataset changed.")

    saved_uses_pseudo_labels = bool(artifact.get("use_pseudo_labels", False))

    if saved_uses_pseudo_labels != bool(use_pseudo_labels):
        reasons.append("Pseudo-label usage changed.")
    elif use_pseudo_labels and artifact.get("pseudo_label_signature") != build_file_signature(
        pseudo_label_path
    ):
        reasons.append("Pseudo-label dataset changed.")

    saved_checkpoint_path = artifact.get("checkpoint_path")

    if saved_checkpoint_path and Path(saved_checkpoint_path).resolve() != resolved_checkpoint_path.resolve():
        reasons.append("Checkpoint path changed.")

    return reasons


def is_training_manifest_compatible(
    manifest: Mapping[str, Any],
    checkpoint_path: Path,
    model_name: str,
    config: Mapping[str, Any],
    train_path: Path,
    validation_path: Path,
    use_pseudo_labels: bool = False,
    pseudo_label_path: Optional[Path] = None,
) -> Tuple[bool, List[str]]:
    """Check whether saved training artifacts can be reused safely."""
    reasons = collect_training_artifact_mismatch_reasons(
        artifact=manifest,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        config=config,
        train_path=train_path,
        validation_path=validation_path,
        use_pseudo_labels=use_pseudo_labels,
        pseudo_label_path=pseudo_label_path,
    )

    return not reasons, reasons


def is_training_state_compatible(
    training_state: Mapping[str, Any],
    checkpoint_path: Path,
    model_name: str,
    config: Mapping[str, Any],
    train_path: Path,
    validation_path: Path,
    use_pseudo_labels: bool = False,
    pseudo_label_path: Optional[Path] = None,
) -> Tuple[bool, List[str]]:
    """Check whether an in-progress training state can be resumed safely."""
    reasons = collect_training_artifact_mismatch_reasons(
        artifact=training_state,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        config=config,
        train_path=train_path,
        validation_path=validation_path,
        use_pseudo_labels=use_pseudo_labels,
        pseudo_label_path=pseudo_label_path,
    )

    return not reasons, reasons


def ensure_trained_model(
    train_path: Path = DEFAULT_TRAIN_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    use_pseudo_labels: bool = False,
    pseudo_label_path: Optional[Path] = None,
    pseudo_label_weight: float = DEFAULT_PSEUDO_LABEL_WEIGHT,
    force_retrain: bool = False,
    allow_checkpoint_fallback: bool = False,
) -> Dict[str, Any]:
    """Reuse compatible saved training artifacts or retrain from scratch."""
    resolved_train_path = resolve_input_path(train_path, DEFAULT_TRAIN_PATH) or DEFAULT_TRAIN_PATH
    resolved_validation_path = (
        resolve_input_path(validation_path, DEFAULT_VALIDATION_PATH) or DEFAULT_VALIDATION_PATH
    )
    resolved_output_dir = resolve_input_path(output_dir, DEFAULT_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR
    resolved_pseudo_label_path = resolve_pseudo_label_input_path(pseudo_label_path)

    resolved_model_name = resolve_model_name(model_name)

    checkpoint_path = resolved_output_dir / "model.pt"
    manifest_path = resolved_output_dir / TRAINING_MANIFEST_FILENAME
    state_path = resolved_output_dir / TRAINING_STATE_FILENAME
    effective_config = normalize_training_config(config)

    retrain_reasons: List[str] = []
    manifest_allows_checkpoint_fallback = not manifest_path.exists()

    if force_retrain:
        retrain_reasons.append("Force retrain requested.")

    elif manifest_path.exists():
        try:
            manifest = load_training_manifest(manifest_path)
        except (OSError, pickle.PickleError, ValueError) as exc:
            retrain_reasons.append(f"Could not load training manifest: {exc}")
            manifest_allows_checkpoint_fallback = True
        else:
            reusable, reuse_blockers = is_training_manifest_compatible(
                manifest=manifest,
                checkpoint_path=checkpoint_path,
                model_name=resolved_model_name,
                config=effective_config,
                train_path=resolved_train_path,
                validation_path=resolved_validation_path,
                use_pseudo_labels=use_pseudo_labels,
                pseudo_label_path=resolved_pseudo_label_path if use_pseudo_labels else None,
            )

            if reusable:
                cleanup_training_state(resolved_output_dir)

                return {
                    "reused_existing_training": True,
                    "compatibility_verified": True,
                    "model_name": str(manifest.get("model_name", resolved_model_name)),
                    "model_family": infer_model_family(str(manifest.get("model_name", resolved_model_name))),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "training_manifest_path": str(manifest_path.resolve()),
                    "training_state_path": str(state_path.resolve()),
                    "best_threshold": float(manifest.get("best_threshold", 0.5)),
                    "metrics": dict(manifest.get("metrics", {})),
                    "config": dict(manifest.get("config", effective_config)),
                    "retrain_reasons": [],
                }

            retrain_reasons.extend(reuse_blockers)
            manifest_allows_checkpoint_fallback = False

    else:
        retrain_reasons.append(f"Training manifest not found at {manifest_path}.")

    if (
        not force_retrain
        and allow_checkpoint_fallback
        and manifest_allows_checkpoint_fallback
        and checkpoint_path.exists()
        and not state_path.exists()
    ):
        checkpoint_metadata = load_model_checkpoint_metadata(checkpoint_path)

        checkpoint_model_name = str(
            checkpoint_metadata.get("model_name", resolved_model_name)
        )

        return {
            "reused_existing_training": True,
            "compatibility_verified": False,
            "model_name": checkpoint_model_name,
            "model_family": infer_model_family(checkpoint_model_name),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "training_manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else None,
            "training_state_path": str(state_path.resolve()),
            "best_threshold": float(checkpoint_metadata.get("best_threshold", 0.5)),
            "metrics": dict(checkpoint_metadata.get("metrics", {})),
            "config": dict(checkpoint_metadata.get("config", effective_config)),
            "retrain_reasons": retrain_reasons,
        }

    train_df = load_dataframe(resolved_train_path)
    val_df = load_dataframe(resolved_validation_path)
    original_train_distribution = compute_class_distribution(train_df)

    pseudo_label_count = 0
    if use_pseudo_labels:
        if not resolved_pseudo_label_path.exists():
            raise FileNotFoundError(
                f"Pseudo-label path does not exist: {resolved_pseudo_label_path}"
            )

        pseudo_df = load_pseudo_label_dataframe(resolved_pseudo_label_path)
        pseudo_label_count = int(len(pseudo_df))
        train_df = merge_pseudo_labels(
            train_df,
            pseudo_df,
            pseudo_label_weight=float(pseudo_label_weight),
        )

        print(f"Loaded {pseudo_label_count} pseudo-labeled rows from {resolved_pseudo_label_path}.")
        print_distribution_summary("Labeled-only distribution", original_train_distribution)
        if pseudo_label_count > 0:
            print_distribution_summary(
                "Pseudo-label-only distribution",
                compute_class_distribution(pseudo_df.assign(sample_weight=float(pseudo_label_weight))),
            )
        print_distribution_summary("Combined training distribution", compute_class_distribution(train_df))

    model, metrics, threshold = train_model(
        train_df=train_df,
        val_df=val_df,
        model_name=resolved_model_name,
        config=effective_config,
        output_dir=resolved_output_dir,
        train_path=resolved_train_path,
        validation_path=resolved_validation_path,
        use_pseudo_labels=use_pseudo_labels,
        pseudo_label_path=resolved_pseudo_label_path if use_pseudo_labels else None,
    )

    del model

    checkpoint_path = resolved_output_dir / "model.pt"

    manifest_path = save_training_manifest(
        output_dir=resolved_output_dir,
        checkpoint_path=checkpoint_path,
        model_name=resolved_model_name,
        config=effective_config,
        metrics=metrics,
        threshold=threshold,
        train_path=resolved_train_path,
        validation_path=resolved_validation_path,
        use_pseudo_labels=use_pseudo_labels,
        pseudo_label_path=resolved_pseudo_label_path if use_pseudo_labels else None,
    )

    cleanup_training_state(resolved_output_dir)

    return {
        "reused_existing_training": False,
        "compatibility_verified": True,
        "model_name": resolved_model_name,
        "model_family": infer_model_family(resolved_model_name),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "training_manifest_path": str(manifest_path.resolve()),
        "training_state_path": str(state_path.resolve()),
        "best_threshold": float(threshold),
        "metrics": metrics,
        "config": effective_config,
        "pseudo_labeled_samples": pseudo_label_count,
        "pseudo_label_weight": float(pseudo_label_weight),
        "retrain_reasons": retrain_reasons,
    }


def set_seed(seed: int) -> None:
    """Set reproducible seeds across NumPy and Torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute standard multi-label metrics from sigmoid probabilities."""
    pred_binary = (predictions >= threshold).astype(int)
    labels_binary = (labels >= 0.5).astype(int)

    return {
        "micro_precision": float(
            precision_score(labels_binary, pred_binary, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(labels_binary, pred_binary, average="micro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(labels_binary, pred_binary, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(labels_binary, pred_binary, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels_binary, pred_binary, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels_binary, pred_binary, average="macro", zero_division=0)
        ),
        "subset_accuracy": float(
            accuracy_score(labels_binary, pred_binary)
        ),
    }


def tune_global_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    search_space: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    """Tune a global threshold during training for checkpoint selection."""
    thresholds = list(search_space or np.arange(0.25, 0.76, 0.05))

    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}
    all_results: Dict[str, Dict[str, float]] = {}
    best_f1 = -1.0

    for threshold in thresholds:
        metrics = compute_metrics(predictions, labels, float(threshold))
        all_results[f"{threshold:.2f}"] = metrics

        if metrics["micro_f1"] > best_f1:
            best_f1 = metrics["micro_f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics, all_results


class ABSAModel(nn.Module):
    """Transformer encoder with a linear multi-label classification head."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        load_pretrained: bool = True,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        resolved_model_name = resolve_model_name(model_name)

        if config_dict:
            config_payload = dict(config_dict)
            model_type = config_payload.pop("model_type", None)

            if model_type:
                self.config = AutoConfig.for_model(model_type, **config_payload)
            else:
                self.config = AutoConfig.from_pretrained(resolved_model_name)
        else:
            self.config = AutoConfig.from_pretrained(resolved_model_name)

        self.transformer = (
            AutoModel.from_pretrained(resolved_model_name)
            if load_pretrained
            else AutoModel.from_config(self.config)
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.model_name = resolved_model_name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if token_type_ids is not None:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        return self.classifier(pooled_output)


def write_label_mapping(
    output_dir: Path,
    label_names: Sequence[str] = ASPECT_SENTIMENT_LABELS,
) -> Path:
    """Save label metadata alongside a checkpoint for downstream evaluation."""
    output_path = output_dir / "label_mapping.json"

    save_json(
        {
            "label_names": list(label_names),
            "label_to_idx": {
                label: index for index, label in enumerate(label_names)
            },
            "idx_to_label": {
                str(index): label for index, label in enumerate(label_names)
            },
        },
        output_path,
    )

    return output_path


def load_absa_model(
    checkpoint_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> ABSAModel:
    """Load a fine-tuned ABSA model checkpoint for inference."""
    if checkpoint_path is None:
        raise NotImplementedError(
            "A fine-tuned ABSA checkpoint is required. Train the model first, then pass "
            "the resulting checkpoint path to load_absa_model()."
        )

    resolved_checkpoint = resolve_input_path(checkpoint_path, checkpoint_path) or checkpoint_path

    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved_checkpoint}")

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(resolved_checkpoint, map_location=resolved_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(resolved_checkpoint, map_location=resolved_device)

    resolved_model_name = resolve_model_name(model_name or checkpoint.get("model_name"))

    model = ABSAModel(
        resolved_model_name,
        num_labels=len(ASPECT_SENTIMENT_LABELS),
        dropout=float(checkpoint.get("config", {}).get("dropout", DEFAULT_CONFIG["dropout"])),
        load_pretrained=False,
        config_dict=checkpoint.get("transformer_config"),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(resolved_device)
    model.eval()

    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Optional[Any],
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> Tuple[float, float]:
    """Run one training epoch."""
    model.train()

    total_loss = 0.0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        sample_weight = batch.get("sample_weight")
        if sample_weight is not None:
            sample_weight = sample_weight.to(device).view(-1, 1)
        else:
            sample_weight = torch.ones((labels.size(0), 1), device=device, dtype=labels.dtype)

        token_type_ids = batch.get("token_type_ids")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss_matrix = criterion(logits, labels)
        loss = (loss_matrix * sample_weight).mean() / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item() * gradient_accumulation_steps:.4f}"
            }
        )

    if len(dataloader) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()

    average_loss = total_loss / max(len(dataloader), 1)
    learning_rate = optimizer.param_groups[0]["lr"]

    return average_loss, learning_rate


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], np.ndarray, Optional[np.ndarray]]:
    """Collect validation probabilities and metrics."""
    model.eval()

    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            token_type_ids = batch.get("token_type_ids")

            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)

            all_predictions.append(torch.sigmoid(logits).cpu().numpy())

            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else None

    metrics: Dict[str, float] = {}

    if labels is not None:
        metrics = compute_metrics(predictions, labels, threshold=threshold)

    return metrics, predictions, labels


def load_pseudo_label_dataframe(pseudo_label_path: Path) -> pd.DataFrame:
    """Load pseudo labels from CSV/JSON and normalize them to the training schema."""
    pseudo_df = load_dataframe(pseudo_label_path)

    if pseudo_df.empty:
        return pseudo_df

    pseudo_df = pseudo_df.copy()

    if "aspects" not in pseudo_df.columns and "predicted_aspects" in pseudo_df.columns:
        pseudo_df["aspects"] = pseudo_df["predicted_aspects"]

    if "aspect_sentiments" not in pseudo_df.columns and "predicted_sentiments" in pseudo_df.columns:
        pseudo_df["aspect_sentiments"] = pseudo_df["predicted_sentiments"]

    if "confidence" not in pseudo_df.columns and "confidence_score" in pseudo_df.columns:
        pseudo_df["confidence"] = pseudo_df["confidence_score"]

    if "source" not in pseudo_df.columns and "pseudo_label_source" in pseudo_df.columns:
        pseudo_df["source"] = pseudo_df["pseudo_label_source"]

    if "aspects" not in pseudo_df.columns or "aspect_sentiments" not in pseudo_df.columns:
        raise ValueError(
            "Pseudo-label file must include either 'aspects'/'aspect_sentiments' or "
            "'predicted_aspects'/'predicted_sentiments' columns."
        )

    pseudo_df["aspects"] = pseudo_df["aspects"].apply(
        lambda value: json.dumps(value, ensure_ascii=False)
        if isinstance(value, list)
        else value
    )
    pseudo_df["aspect_sentiments"] = pseudo_df["aspect_sentiments"].apply(
        lambda value: json.dumps(value, ensure_ascii=False)
        if isinstance(value, dict)
        else value
    )

    if "review_id" not in pseudo_df.columns:
        pseudo_df["review_id"] = np.arange(1, len(pseudo_df) + 1)

    if "review_text" not in pseudo_df.columns:
        raise ValueError("Pseudo-label file is missing the required 'review_text' column.")

    return pseudo_df


def merge_pseudo_labels(
    train_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
    pseudo_label_weight: float = DEFAULT_PSEUDO_LABEL_WEIGHT,
) -> pd.DataFrame:
    """Append pseudo-labeled rows while preserving metadata and sample weights."""
    if pseudo_df.empty:
        merged_empty = train_df.copy()
        if "sample_weight" not in merged_empty.columns:
            merged_empty["sample_weight"] = 1.0
        if "label_origin" not in merged_empty.columns:
            merged_empty["label_origin"] = "gold"
        return merged_empty

    train_mapping = infer_column_mapping(train_df, require_labels=True)
    pseudo_mapping = infer_column_mapping(pseudo_df, require_labels=True)

    merged = train_df.copy()
    if "sample_weight" not in merged.columns:
        merged["sample_weight"] = 1.0
    else:
        merged["sample_weight"] = pd.to_numeric(merged["sample_weight"], errors="coerce").fillna(1.0)
    if "label_origin" not in merged.columns:
        merged["label_origin"] = "gold"

    combined_columns = list(merged.columns)
    for column_name in pseudo_df.columns:
        if column_name not in combined_columns:
            combined_columns.append(column_name)

    aligned_rows = pd.DataFrame(columns=combined_columns)
    aligned_rows[train_mapping.review_id] = pseudo_df[pseudo_mapping.review_id]
    aligned_rows[train_mapping.review_text] = pseudo_df[pseudo_mapping.review_text]
    aligned_rows[train_mapping.aspects] = pseudo_df[pseudo_mapping.aspects]
    aligned_rows[train_mapping.aspect_sentiments] = pseudo_df[pseudo_mapping.aspect_sentiments]

    for column_name in combined_columns:
        if column_name in aligned_rows.columns and aligned_rows[column_name].notna().any():
            continue
        if column_name == "sample_weight":
            aligned_rows[column_name] = float(pseudo_label_weight)
        elif column_name == "label_origin":
            aligned_rows[column_name] = "pseudo"
        elif column_name in pseudo_df.columns:
            aligned_rows[column_name] = pseudo_df[column_name]
        elif column_name in merged.columns:
            aligned_rows[column_name] = np.nan

    merged = pd.concat([merged[combined_columns], aligned_rows[combined_columns]], ignore_index=True)

    return merged


def save_checkpoint(
    model: ABSAModel,
    tokenizer_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    threshold: float,
    threshold_history: Mapping[str, Any],
    class_distribution: Mapping[str, Any],
) -> Path:
    """Persist the trained model and metadata."""
    checkpoint_path = output_dir / "model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "metrics": dict(metrics),
            "best_threshold": float(threshold),
            "threshold": float(threshold),
            "threshold_history": dict(threshold_history),
            "model_name": model.model_name,
            "model_family": infer_model_family(model.model_name),
            "tokenizer_name": model.model_name,
            "tokenizer_dir_name": tokenizer_dir.name,
            "transformer_config": model.config.to_dict(),
            "label_names": ASPECT_SENTIMENT_LABELS,
            "label_to_idx": LABEL_TO_IDX,
            "idx_to_label": IDX_TO_LABEL,
            "class_distribution": dict(class_distribution),
        },
        checkpoint_path,
    )

    return checkpoint_path


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_path: Optional[Path] = None,
    validation_path: Optional[Path] = None,
    use_pseudo_labels: bool = False,
    pseudo_label_path: Optional[Path] = None,
    resume_from_state: bool = True,
) -> Tuple[ABSAModel, Dict[str, float], float]:
    """Train the model and save the best checkpoint."""
    cfg = normalize_training_config(config)
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    resolved_model_name = resolve_model_name(model_name)

    print(f"Using model: {resolved_model_name}")
    print(f"Model family: {infer_model_family(resolved_model_name)}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
    preprocessor = ArabicPreprocessor()

    train_dataset = ABDataset(
        train_df,
        tokenizer,
        max_length=int(cfg["max_length"]),
        preprocessor=preprocessor,
    )

    val_dataset = ABDataset(
        val_df,
        tokenizer,
        max_length=int(cfg["max_length"]),
        preprocessor=preprocessor,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "model.pt"
    state_path = output_dir / TRAINING_STATE_FILENAME
    tokenizer_dir = output_dir / "tokenizer"

    tokenizer.save_pretrained(tokenizer_dir)
    write_label_mapping(output_dir)

    class_distribution = compute_class_distribution(train_df)
    save_json(class_distribution, output_dir / "class_distribution.json")
    print_distribution_summary("Training distribution used for loss weighting", class_distribution)

    pos_weight = build_pos_weight_tensor(class_distribution).to(device)

    model = ABSAModel(
        resolved_model_name,
        num_labels=len(ASPECT_SENTIMENT_LABELS),
        dropout=float(cfg["dropout"]),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        foreach=False,
    )

    total_steps = max(
        1,
        len(train_loader) * int(cfg["num_epochs"]) // int(cfg["gradient_accumulation_steps"]),
    )

    warmup_steps = int(total_steps * float(cfg["warmup_ratio"]))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    resolved_train_path = (
        resolve_input_path(train_path, train_path)
        if train_path is not None
        else None
    )

    resolved_validation_path = (
        resolve_input_path(validation_path, validation_path)
        if validation_path is not None
        else None
    )

    resolved_pseudo_label_path = (
        resolve_pseudo_label_input_path(pseudo_label_path)
        if pseudo_label_path is not None
        else None
    )

    resume_state_enabled = (
        resume_from_state
        and resolved_train_path is not None
        and resolved_validation_path is not None
    )

    best_f1 = -1.0
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}
    best_threshold_history: Dict[str, Dict[str, float]] = {}
    stagnant_epochs = 0
    start_epoch = 0

    if resume_state_enabled and state_path.exists():
        try:
            training_state = load_training_state(state_path)
        except (OSError, RuntimeError, ValueError, pickle.PickleError) as exc:
            print(f"Could not load resume state from {state_path}: {exc}")
        else:
            reusable, blockers = is_training_state_compatible(
                training_state=training_state,
                checkpoint_path=checkpoint_path,
                model_name=resolved_model_name,
                config=cfg,
                train_path=resolved_train_path,
                validation_path=resolved_validation_path,
                use_pseudo_labels=use_pseudo_labels,
                pseudo_label_path=resolved_pseudo_label_path if use_pseudo_labels else None,
            )

            if reusable:
                model.load_state_dict(training_state["model_state_dict"])
                optimizer.load_state_dict(training_state["optimizer_state_dict"])
                move_optimizer_to_device(optimizer, device)

                scheduler_state_dict = training_state.get("scheduler_state_dict")

                if scheduler_state_dict:
                    scheduler.load_state_dict(scheduler_state_dict)

                best_f1 = float(training_state.get("best_f1", -1.0))
                best_threshold = float(training_state.get("best_threshold", 0.5))
                best_metrics = dict(training_state.get("best_metrics", {}))
                best_threshold_history = dict(training_state.get("best_threshold_history", {}))
                stagnant_epochs = int(training_state.get("stagnant_epochs", 0))
                start_epoch = int(training_state.get("next_epoch", 0))

                restore_random_state(training_state)

                print(
                    f"Resuming training from epoch {start_epoch + 1}/{cfg['num_epochs']} "
                    f"using {state_path}"
                )
            else:
                print("Ignoring incompatible resume state:")

                for blocker in blockers:
                    print(f"- {blocker}")

    if not resume_state_enabled and resume_from_state and state_path.exists():
        print(
            f"Resume state exists at {state_path}, but dataset paths were not provided. "
            "Starting from scratch."
        )

    if start_epoch >= int(cfg["num_epochs"]):
        print("Requested epoch count is already covered by the saved training state. Finalizing checkpoint.")

    for epoch in range(start_epoch, int(cfg["num_epochs"])):
        print(f"\nEpoch {epoch + 1}/{cfg['num_epochs']}")

        train_loss, learning_rate = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
        )

        print(f"Train loss: {train_loss:.4f} | LR: {learning_rate:.2e}")

        _, predictions, labels = evaluate_model(
            model,
            val_loader,
            device=device,
            threshold=0.5,
        )

        if labels is None:
            raise ValueError("Validation labels are required for training-time model selection.")

        epoch_threshold, epoch_metrics, threshold_history = tune_global_threshold(
            predictions,
            labels,
        )

        print(
            f"Validation micro-F1: {epoch_metrics['micro_f1']:.4f} "
            f"at threshold {epoch_threshold:.2f}"
        )

        if epoch_metrics["micro_f1"] > best_f1:
            best_f1 = epoch_metrics["micro_f1"]
            best_threshold = epoch_threshold
            best_metrics = dict(epoch_metrics)
            best_threshold_history = threshold_history
            stagnant_epochs = 0

            checkpoint_path = save_checkpoint(
                model=model,
                tokenizer_dir=tokenizer_dir,
                output_dir=output_dir,
                config=cfg,
                metrics=best_metrics,
                threshold=best_threshold,
                threshold_history=best_threshold_history,
                class_distribution=class_distribution,
            )

            print(f"Saved new best checkpoint to {checkpoint_path}")

        else:
            stagnant_epochs += 1

            print(
                f"No improvement for {stagnant_epochs} epoch(s); "
                f"best micro-F1 remains {best_f1:.4f}."
            )

            if stagnant_epochs >= int(cfg["early_stopping_patience"]):
                print("Early stopping triggered.")

        if resume_state_enabled:
            save_training_state(
                state_path=state_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                model_name=resolved_model_name,
                config=cfg,
                checkpoint_path=checkpoint_path,
                train_path=resolved_train_path,
                validation_path=resolved_validation_path,
                next_epoch=epoch + 1,
                best_f1=best_f1,
                best_threshold=best_threshold,
                best_metrics=best_metrics,
                best_threshold_history=best_threshold_history,
                stagnant_epochs=stagnant_epochs,
                class_distribution=class_distribution,
                use_pseudo_labels=use_pseudo_labels,
                pseudo_label_path=resolved_pseudo_label_path if use_pseudo_labels else None,
            )

        if stagnant_epochs >= int(cfg["early_stopping_patience"]):
            break

    if checkpoint_path.exists():
        checkpoint_metadata = load_model_checkpoint_metadata(checkpoint_path)
        model.load_state_dict(checkpoint_metadata["model_state_dict"])

    final_metrics, _, _ = evaluate_model(
        model,
        val_loader,
        device=device,
        threshold=float(best_threshold),
    )

    checkpoint_path = save_checkpoint(
        model=model,
        tokenizer_dir=tokenizer_dir,
        output_dir=output_dir,
        config=cfg,
        metrics=final_metrics,
        threshold=float(best_threshold),
        threshold_history=best_threshold_history,
        class_distribution=class_distribution,
    )

    cleanup_training_state(output_dir)

    print(f"Final checkpoint saved to {checkpoint_path}")

    return model, final_metrics, float(best_threshold)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Train the Arabic ABSA model.")

    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)

    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_ALIAS,
        help=(
            "Model alias or Hugging Face model name. "
            "Supported aliases: marbert, arabert, xlmr. "
            "Default: marbert."
        ),
    )

    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_CONFIG["warmup_ratio"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULT_CONFIG["gradient_accumulation_steps"],
    )
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_CONFIG["max_grad_norm"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=DEFAULT_CONFIG["early_stopping_patience"],
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])

    parser.add_argument("--use_pseudo_labels", type=str2bool, default=False)
    parser.add_argument("--pseudo_label_path", type=Path, default=DEFAULT_PSEUDO_LABEL_PATH)
    parser.add_argument("--pseudo_label_weight", type=float, default=DEFAULT_PSEUDO_LABEL_WEIGHT)

    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force_retrain", type=str2bool, default=False)
    parser.add_argument("--allow_checkpoint_fallback", type=str2bool, default=True)

    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    train_path = resolve_input_path(args.train_path, DEFAULT_TRAIN_PATH) or DEFAULT_TRAIN_PATH
    validation_path = (
        resolve_input_path(args.validation_path, DEFAULT_VALIDATION_PATH)
        or DEFAULT_VALIDATION_PATH
    )
    output_dir = resolve_input_path(args.output_dir, DEFAULT_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR
    pseudo_label_path = resolve_pseudo_label_input_path(args.pseudo_label_path)

    resolved_model_name = resolve_model_name(args.model_name)

    training_result = ensure_trained_model(
        train_path=train_path,
        validation_path=validation_path,
        model_name=resolved_model_name,
        config={
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "dropout": args.dropout,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed,
            "pseudo_label_weight": args.pseudo_label_weight,
        },
        output_dir=output_dir,
        use_pseudo_labels=args.use_pseudo_labels,
        pseudo_label_path=pseudo_label_path,
        pseudo_label_weight=args.pseudo_label_weight,
        force_retrain=args.force_retrain,
        allow_checkpoint_fallback=args.allow_checkpoint_fallback,
    )

    print(
        json.dumps(
            {
                **training_result,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
