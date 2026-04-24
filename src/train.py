"""
Training Module for Arabic ABSA
================================
Handles model training with MARBERTv2 or AraBERT.
"""

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import ABDataset, ASPECT_SENTIMENT_LABELS
from preprocess import ArabicPreprocessor


DEFAULT_MODELS = {
    "marbert": "UBC-NLP/MARBERTv2",
    "arabert": "aubmindlab/bert-base-arabertv02",
    "arabertv3": "aubmindlab/bert-base-arabertv3",
}

DEFAULT_CONFIG = {
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "seed": 42,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
DEFAULT_TRAIN_PATH = DATASET_ROOT / "DeepX_train.xlsx"
DEFAULT_VALIDATION_PATH = DATASET_ROOT / "DeepX_validation.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


class ABSAModel(nn.Module):
    """Multi-label classification model for Arabic ABSA."""

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
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


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    pred_binary = (predictions >= threshold).astype(int)
    labels_binary = (labels >= 0.5).astype(int)

    micro_precision = precision_score(labels_binary, pred_binary, average="micro", zero_division=0)
    micro_recall = recall_score(labels_binary, pred_binary, average="micro", zero_division=0)
    micro_f1 = f1_score(labels_binary, pred_binary, average="micro", zero_division=0)

    macro_precision = precision_score(labels_binary, pred_binary, average="macro", zero_division=0)
    macro_recall = recall_score(labels_binary, pred_binary, average="macro", zero_division=0)
    macro_f1 = f1_score(labels_binary, pred_binary, average="macro", zero_division=0)

    subset_accuracy = accuracy_score(labels_binary, pred_binary)

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "subset_accuracy": subset_accuracy,
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

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

    avg_loss = total_loss / len(dataloader)
    learning_rate = optimizer.param_groups[0]["lr"]
    return avg_loss, learning_rate


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], np.ndarray, Optional[np.ndarray]]:
    """Evaluate the model and collect predictions and labels."""
    model.eval()
    all_predictions = []
    all_labels = []

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

    all_predictions_array = np.concatenate(all_predictions, axis=0)
    all_labels_array = None
    metrics: Dict[str, float] = {}

    if all_labels:
        all_labels_array = np.concatenate(all_labels, axis=0)
        metrics = compute_metrics(all_predictions_array, all_labels_array, threshold)

    return metrics, all_predictions_array, all_labels_array


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = "marbert",
    config: Optional[Dict] = None,
    output_dir: str = "outputs",
    device: Optional[torch.device] = None,
) -> Tuple[ABSAModel, Dict, float]:
    """Train the full model and save the best checkpoint."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg["seed"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    resolved_model_name = DEFAULT_MODELS.get(model_name, model_name)
    print(f"Loading tokenizer: {resolved_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
    preprocessor = ArabicPreprocessor()

    print("Creating datasets...")
    train_dataset = ABDataset(
        train_df,
        tokenizer,
        max_length=cfg["max_length"],
        preprocessor=preprocessor,
    )
    val_dataset = ABDataset(
        val_df,
        tokenizer,
        max_length=cfg["max_length"],
        preprocessor=preprocessor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    print(f"Creating model: {resolved_model_name}")
    model = ABSAModel(resolved_model_name, len(ASPECT_SENTIMENT_LABELS)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    total_steps = max(
        1,
        len(train_loader) * cfg["num_epochs"] // cfg["gradient_accumulation_steps"],
    )
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("\nStarting training...")
    best_f1 = -1.0
    best_model_state = None
    best_threshold = 0.5

    for epoch in range(cfg["num_epochs"]):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{cfg['num_epochs']}")
        print(f"{'=' * 50}")

        train_loss, learning_rate = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            cfg["gradient_accumulation_steps"],
            cfg["max_grad_norm"],
        )
        print(f"Train Loss: {train_loss:.4f} | LR: {learning_rate:.2e}")

        metrics, val_predictions, val_labels = evaluate(model, val_loader, device)
        print(f"Validation Micro F1: {metrics['micro_f1']:.4f}")
        print(f"Validation Micro Precision: {metrics['micro_precision']:.4f}")
        print(f"Validation Micro Recall: {metrics['micro_recall']:.4f}")

        print("\nTuning threshold...")
        best_epoch_f1 = -1.0
        best_epoch_threshold = 0.5
        if val_labels is None:
            raise ValueError("Validation labels were not available for threshold tuning.")

        for threshold in np.arange(0.1, 0.9, 0.05):
            threshold_metrics = compute_metrics(val_predictions, val_labels, threshold=threshold)
            if threshold_metrics["micro_f1"] > best_epoch_f1:
                best_epoch_f1 = threshold_metrics["micro_f1"]
                best_epoch_threshold = float(threshold)

        print(
            f"Best threshold for epoch: {best_epoch_threshold:.2f} "
            f"(F1: {best_epoch_f1:.4f})"
        )

        if best_model_state is None or best_epoch_f1 > best_f1:
            best_f1 = best_epoch_f1
            best_model_state = copy.deepcopy(model.state_dict())
            best_threshold = best_epoch_threshold
            print(f"New best model! F1: {best_f1:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n{'=' * 50}")
    print("Final Evaluation")
    print(f"{'=' * 50}")
    final_metrics, _, _ = evaluate(model, val_loader, device, threshold=best_threshold)

    print(f"Final Micro F1: {final_metrics['micro_f1']:.4f}")
    print(f"Final Micro Precision: {final_metrics['micro_precision']:.4f}")
    print(f"Final Micro Recall: {final_metrics['micro_recall']:.4f}")
    print(f"Best Threshold: {best_threshold:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "metrics": final_metrics,
            "threshold": best_threshold,
            "model_name": resolved_model_name,
        },
        checkpoint_path,
    )
    print(f"Model saved to: {checkpoint_path}")

    return model, final_metrics, best_threshold


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for training."""
    parser = argparse.ArgumentParser(description="Train the Arabic ABSA model.")
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--model_name", default=DEFAULT_MODELS["marbert"])
    parser.add_argument("--fallback_model_name", default=DEFAULT_MODELS["arabert"])
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
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def should_retry_with_fallback(exc: Exception) -> bool:
    """Return True when the failure likely came from model size/download issues."""
    message = str(exc).lower()
    retry_tokens = (
        "out of memory",
        "cuda",
        "cublas",
        "download",
        "timeout",
        "timed out",
        "connection",
        "ssl",
        "repository",
        "403",
        "404",
        "not found",
        "max retries exceeded",
    )
    return any(token in message for token in retry_tokens)


def run_from_args(args: argparse.Namespace) -> Tuple[Dict[str, float], float, str]:
    """Load data and run training from parsed CLI arguments."""
    train_df = pd.read_excel(args.train_path)
    val_df = pd.read_excel(args.validation_path)

    config = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
    }

    model_candidates = [args.model_name]
    if args.fallback_model_name and args.fallback_model_name != args.model_name:
        model_candidates.append(args.fallback_model_name)

    last_error: Optional[Exception] = None
    for index, candidate in enumerate(model_candidates):
        try:
            _, metrics, threshold = train_model(
                train_df=train_df,
                val_df=val_df,
                model_name=candidate,
                config=config,
                output_dir=str(args.output_dir),
            )
            return metrics, threshold, candidate
        except Exception as exc:  # pragma: no cover - runtime fallback path
            last_error = exc
            is_last_candidate = index == len(model_candidates) - 1
            if is_last_candidate or not should_retry_with_fallback(exc):
                raise
            next_candidate = model_candidates[index + 1]
            print(
                f"Model '{candidate}' failed with {type(exc).__name__}: {exc}\n"
                f"Retrying with fallback model '{next_candidate}'."
            )

    raise RuntimeError(f"Training failed: {last_error}")


def main():
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    metrics, threshold, model_name = run_from_args(args)
    print(f"Training completed with model: {model_name}")
    print(json.dumps({"threshold": threshold, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
