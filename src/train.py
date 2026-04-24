"""
Training pipeline for Arabic ABSA with class imbalance handling and pseudo labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
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
    DATASET_ROOT,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALIDATION_PATH,
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
    "marbertv2": "UBC-NLP/MARBERTv2",
    "arabert": "aubmindlab/bert-base-arabertv02",
    "arabertv02": "aubmindlab/bert-base-arabertv02",
    "arabertv3": "aubmindlab/bert-base-arabertv3",
}

DEFAULT_CONFIG = {
    "max_length": 256,
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

DEFAULT_PSEUDO_LABEL_PATH = OUTPUTS_ROOT / "pseudo_labeled.json"
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT


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


def resolve_model_name(model_name: Optional[str]) -> str:
    """Resolve model aliases to a concrete Hugging Face identifier."""
    if not model_name:
        return DEFAULT_MODELS["marbertv2"]
    return DEFAULT_MODELS.get(model_name, model_name)


def infer_model_family(model_name: str) -> str:
    """Infer a short model family tag for metadata and ensemble discovery."""
    lowered = model_name.lower()
    if "marbert" in lowered:
        return "marbert"
    if "arabert" in lowered:
        return "arabert"
    return "custom"


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
        "micro_precision": float(precision_score(labels_binary, pred_binary, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(labels_binary, pred_binary, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(labels_binary, pred_binary, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(labels_binary, pred_binary, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels_binary, pred_binary, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels_binary, pred_binary, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(labels_binary, pred_binary)),
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
        if config_dict:
            config_payload = dict(config_dict)
            model_type = config_payload.pop("model_type", None)
            if model_type:
                self.config = AutoConfig.for_model(model_type, **config_payload)
            else:
                self.config = AutoConfig.from_pretrained(model_name)
        else:
            self.config = AutoConfig.from_pretrained(model_name)

        self.transformer = (
            AutoModel.from_pretrained(model_name)
            if load_pretrained
            else AutoModel.from_config(self.config)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.model_name = model_name

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
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels) / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

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
    """Load pseudo labels from JSON and normalize to the training schema."""
    pseudo_df = load_dataframe(pseudo_label_path)
    if pseudo_df.empty:
        return pseudo_df

    if "aspects" not in pseudo_df.columns and "prediction" in pseudo_df.columns:
        raise ValueError(
            "Pseudo-label file is missing direct 'aspects' and 'aspect_sentiments' columns."
        )

    pseudo_df = pseudo_df.copy()
    if "aspects" in pseudo_df.columns:
        pseudo_df["aspects"] = pseudo_df["aspects"].apply(
            lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
        )
    if "aspect_sentiments" in pseudo_df.columns:
        pseudo_df["aspect_sentiments"] = pseudo_df["aspect_sentiments"].apply(
            lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
        )
    return pseudo_df


def merge_pseudo_labels(
    train_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Append pseudo-labeled rows while preserving the original training schema."""
    if pseudo_df.empty:
        return train_df

    train_mapping = infer_column_mapping(train_df, require_labels=True)
    pseudo_mapping = infer_column_mapping(pseudo_df, require_labels=True)
    merged = train_df.copy()

    aligned_rows = pd.DataFrame(
        {
            train_mapping.review_id: pseudo_df[pseudo_mapping.review_id],
            train_mapping.review_text: pseudo_df[pseudo_mapping.review_text],
            train_mapping.aspects: pseudo_df[pseudo_mapping.aspects],
            train_mapping.aspect_sentiments: pseudo_df[pseudo_mapping.aspect_sentiments],
        }
    )
    merged = pd.concat([merged, aligned_rows], ignore_index=True)
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
            "threshold_history": dict(threshold_history),
            "model_name": model.model_name,
            "model_family": infer_model_family(model.model_name),
            "tokenizer_name": model.model_name,
            "tokenizer_dir_name": tokenizer_dir.name,
            "transformer_config": model.config.to_dict(),
            "label_names": ASPECT_SENTIMENT_LABELS,
            "label_to_idx": LABEL_TO_IDX,
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
) -> Tuple[ABSAModel, Dict[str, float], float]:
    """Train the model and save the best checkpoint."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    resolved_model_name = resolve_model_name(model_name)
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
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    class_distribution = compute_class_distribution(train_df)
    save_json(class_distribution, output_dir / "class_distribution.json")
    pos_weight = build_pos_weight_tensor(class_distribution).to(device)

    model = ABSAModel(
        resolved_model_name,
        num_labels=len(ASPECT_SENTIMENT_LABELS),
        dropout=float(cfg["dropout"]),
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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

    best_f1 = -1.0
    best_state_dict = None
    best_threshold = DEFAULT_CONFIG["seed"] * 0.0 + 0.5
    best_metrics: Dict[str, float] = {}
    best_threshold_history: Dict[str, Dict[str, float]] = {}
    stagnant_epochs = 0

    for epoch in range(int(cfg["num_epochs"])):
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

        _, predictions, labels = evaluate_model(model, val_loader, device=device, threshold=0.5)
        if labels is None:
            raise ValueError("Validation labels are required for training-time model selection.")

        epoch_threshold, epoch_metrics, threshold_history = tune_global_threshold(predictions, labels)
        print(
            f"Validation micro-F1: {epoch_metrics['micro_f1']:.4f} "
            f"at threshold {epoch_threshold:.2f}"
        )

        if epoch_metrics["micro_f1"] > best_f1:
            best_f1 = epoch_metrics["micro_f1"]
            best_state_dict = copy.deepcopy(model.state_dict())
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
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_metrics, _, _ = evaluate_model(model, val_loader, device=device, threshold=float(best_threshold))
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
    print(f"Final checkpoint saved to {checkpoint_path}")
    return model, final_metrics, float(best_threshold)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Train the Arabic ABSA model.")
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--model_name", default=DEFAULT_MODELS["marbertv2"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_CONFIG["warmup_ratio"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_CONFIG["max_grad_norm"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--early_stopping_patience", type=int, default=DEFAULT_CONFIG["early_stopping_patience"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--use_pseudo_labels", type=str2bool, default=False)
    parser.add_argument("--pseudo_label_path", type=Path, default=DEFAULT_PSEUDO_LABEL_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    train_path = resolve_input_path(args.train_path, DEFAULT_TRAIN_PATH) or DEFAULT_TRAIN_PATH
    validation_path = resolve_input_path(args.validation_path, DEFAULT_VALIDATION_PATH) or DEFAULT_VALIDATION_PATH
    output_dir = resolve_input_path(args.output_dir, DEFAULT_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR
    pseudo_label_path = resolve_input_path(args.pseudo_label_path, DEFAULT_PSEUDO_LABEL_PATH)

    train_df = load_dataframe(train_path)
    val_df = load_dataframe(validation_path)
    if args.use_pseudo_labels:
        if not pseudo_label_path or not pseudo_label_path.exists():
            raise FileNotFoundError(
                f"Pseudo-label path does not exist: {pseudo_label_path}"
            )
        pseudo_df = load_pseudo_label_dataframe(pseudo_label_path)
        train_df = merge_pseudo_labels(train_df, pseudo_df)
        print(f"Loaded {len(pseudo_df)} pseudo-labeled rows.")

    model, metrics, threshold = train_model(
        train_df=train_df,
        val_df=val_df,
        model_name=args.model_name,
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
        },
        output_dir=output_dir,
    )
    del model
    print(
        json.dumps(
            {
                "model_name": resolve_model_name(args.model_name),
                "best_threshold": threshold,
                "metrics": metrics,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
