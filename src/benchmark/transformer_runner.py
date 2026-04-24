"""Transformer benchmark runner for AraBERT and MARBERT."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from benchmark.evaluate_model import ModelPredictionBundle
from benchmark.metrics import decode_prediction, records_from_dataframe, tune_aspect_thresholds
from dataset import (
    VALID_ASPECTS,
    VALID_SENTIMENTS,
    coerce_review_id,
    infer_column_mapping,
    parse_json_column,
    parse_sentiment_dict,
)
from preprocess import ArabicPreprocessor


TRANSFORMER_MODELS = {
    "arabert": "aubmindlab/bert-base-arabertv02",
    "marbert": "UBC-NLP/MARBERT",
}


@dataclass
class TransformerTrainingConfig:
    """Training configuration for benchmark transformer models."""

    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    dropout: float = 0.2
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    seed: int = 42
    sentiment_loss_weight: float = 1.0
    threshold_granularity: str = "aspect"
    threshold_tuning_passes: int = 2


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_targets(aspects: Sequence[str], aspect_sentiments: Mapping[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Build aspect and sentiment targets for one review."""
    aspect_targets = np.zeros(len(VALID_ASPECTS), dtype=np.float32)
    sentiment_targets = np.full(len(VALID_ASPECTS), -100, dtype=np.int64)

    for aspect in aspects:
        if aspect not in VALID_ASPECTS:
            continue
        aspect_index = VALID_ASPECTS.index(aspect)
        aspect_targets[aspect_index] = 1.0
        sentiment = str(aspect_sentiments.get(aspect, "neutral"))
        sentiment_targets[aspect_index] = VALID_SENTIMENTS.index(sentiment) if sentiment in VALID_SENTIMENTS else VALID_SENTIMENTS.index("neutral")

    return aspect_targets, sentiment_targets


class TransformerABSADataset(Dataset):
    """Dataset for transformer-based multi-task ABSA."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Any,
        max_length: int,
        preprocessor: Optional[ArabicPreprocessor] = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor or ArabicPreprocessor()
        self.mapping = infer_column_mapping(self.dataframe, require_labels=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[index]
        review_text = str(row[self.mapping.review_text])
        normalized_text = self.preprocessor.normalize(review_text)
        aspects = parse_json_column(row[self.mapping.aspects]) if self.mapping.aspects else []
        aspect_sentiments = parse_sentiment_dict(row[self.mapping.aspect_sentiments]) if self.mapping.aspect_sentiments else {}
        aspect_targets, sentiment_targets = build_targets(aspects, aspect_sentiments)

        encoding = self.tokenizer(
            normalized_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: Dict[str, Any] = {
            "review_id": row[self.mapping.review_id],
            "review_text": review_text,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "aspect_targets": torch.tensor(aspect_targets, dtype=torch.float32),
            "sentiment_targets": torch.tensor(sentiment_targets, dtype=torch.long),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        return item


class TransformerABSAModel(nn.Module):
    """Shared transformer encoder with aspect and sentiment heads."""

    def __init__(self, model_name: str, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.aspect_head = nn.Linear(hidden_size, len(VALID_ASPECTS))
        self.sentiment_head = nn.Linear(hidden_size, len(VALID_ASPECTS) * len(VALID_SENTIMENTS))
        self.model_name = model_name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_type_ids is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        aspect_logits = self.aspect_head(pooled)
        sentiment_logits = self.sentiment_head(pooled).view(-1, len(VALID_ASPECTS), len(VALID_SENTIMENTS))
        return aspect_logits, sentiment_logits


def build_aspect_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    """Build a positive-class weight vector for aspect detection."""
    gold_records = records_from_dataframe(train_df)
    aspect_matrix = np.zeros((len(gold_records), len(VALID_ASPECTS)), dtype=np.float32)
    for row_index, record in enumerate(gold_records):
        for aspect in record["aspects"]:
            aspect_matrix[row_index, VALID_ASPECTS.index(aspect)] = 1.0

    positives = aspect_matrix.sum(axis=0)
    negatives = max(len(train_df), 1) - positives
    pos_weight = (negatives + 1.0) / (positives + 1.0)
    pos_weight = np.clip(pos_weight, 1.0, 20.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_losses(
    aspect_logits: torch.Tensor,
    sentiment_logits: torch.Tensor,
    aspect_targets: torch.Tensor,
    sentiment_targets: torch.Tensor,
    aspect_criterion: nn.Module,
    sentiment_criterion: nn.Module,
    sentiment_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the joint multi-task loss."""
    aspect_loss = aspect_criterion(aspect_logits, aspect_targets)
    sentiment_loss = sentiment_criterion(
        sentiment_logits.view(-1, len(VALID_SENTIMENTS)),
        sentiment_targets.view(-1),
    )
    total_loss = aspect_loss + (sentiment_loss_weight * sentiment_loss)
    return total_loss, {
        "aspect_loss": float(aspect_loss.detach().cpu().item()),
        "sentiment_loss": float(sentiment_loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
    }


def train_epoch(
    model: TransformerABSAModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    aspect_criterion: nn.Module,
    sentiment_criterion: nn.Module,
    config: TransformerTrainingConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run a single training epoch."""
    model.train()
    optimizer.zero_grad()
    losses: List[Dict[str, float]] = []

    progress_bar = tqdm(dataloader, desc="Training transformer", leave=False)
    for step, batch in enumerate(progress_bar):
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        aspect_logits, sentiment_logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            token_type_ids,
        )
        total_loss, loss_items = compute_losses(
            aspect_logits,
            sentiment_logits,
            batch["aspect_targets"].to(device),
            batch["sentiment_targets"].to(device),
            aspect_criterion,
            sentiment_criterion,
            config.sentiment_loss_weight,
        )
        total_loss = total_loss / config.gradient_accumulation_steps
        total_loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        losses.append(loss_items)
        progress_bar.set_postfix({"loss": f"{loss_items['total_loss']:.4f}"})

    if len(dataloader) % config.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return {
        "aspect_loss": round(float(np.mean([item["aspect_loss"] for item in losses])), 6),
        "sentiment_loss": round(float(np.mean([item["sentiment_loss"] for item in losses])), 6),
        "total_loss": round(float(np.mean([item["total_loss"] for item in losses])), 6),
    }


def collect_validation_outputs(
    model: TransformerABSAModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Collect validation probabilities and references."""
    model.eval()
    review_ids: List[Any] = []
    review_texts: List[str] = []
    aspect_probabilities: List[np.ndarray] = []
    sentiment_probabilities: List[np.ndarray] = []
    aspect_targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating transformer", leave=False):
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            aspect_logits, sentiment_logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                token_type_ids,
            )
            aspect_probs = torch.sigmoid(aspect_logits).cpu().numpy()
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1).cpu().numpy()

            aspect_probabilities.extend(aspect_probs)
            sentiment_probabilities.extend(sentiment_probs)
            aspect_targets.extend(batch["aspect_targets"].cpu().numpy())
            review_ids.extend(batch["review_id"])
            review_texts.extend(batch["review_text"])

    return {
        "review_ids": review_ids,
        "review_texts": review_texts,
        "aspect_probabilities": np.asarray(aspect_probabilities, dtype=np.float32),
        "sentiment_probabilities": np.asarray(sentiment_probabilities, dtype=np.float32),
        "aspect_targets": np.asarray(aspect_targets, dtype=np.float32),
    }


def build_predictions(
    review_ids: Sequence[Any],
    review_texts: Sequence[str],
    aspect_probabilities: np.ndarray,
    sentiment_probabilities: np.ndarray,
    thresholds: Mapping[str, float],
) -> List[Dict[str, Any]]:
    """Decode probability tensors into submission-compatible predictions."""
    predictions: List[Dict[str, Any]] = []
    for review_id, review_text, aspect_probs, sentiment_probs in zip(
        review_ids,
        review_texts,
        aspect_probabilities,
        sentiment_probabilities,
    ):
        aspects, aspect_sentiments = decode_prediction(aspect_probs, sentiment_probs, thresholds)
        predictions.append(
            {
                "review_id": coerce_review_id(review_id),
                "review_text": review_text,
                "aspects": aspects,
                "aspect_sentiments": aspect_sentiments,
            }
        )
    return predictions


def save_checkpoint(
    model: TransformerABSAModel,
    tokenizer: Any,
    output_dir: Path,
    config: TransformerTrainingConfig,
    thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, str]:
    """Persist the best model checkpoint and tokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model.model_name,
            "config": vars(config),
            "thresholds": dict(thresholds or {}),
        },
        checkpoint_path,
    )
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(config), handle, ensure_ascii=False, indent=2)
    return {
        "checkpoint": str(checkpoint_path),
        "tokenizer_dir": str(tokenizer_dir),
    }


def run_transformer_benchmark(
    model_key: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: Path,
    config: Optional[TransformerTrainingConfig] = None,
    device: Optional[torch.device] = None,
) -> ModelPredictionBundle:
    """Train and evaluate one transformer model for the benchmark."""
    if model_key not in TRANSFORMER_MODELS:
        raise ValueError(f"Unsupported transformer model key: {model_key}")

    cfg = config or TransformerTrainingConfig()
    set_seed(cfg.seed)
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = TRANSFORMER_MODELS[model_key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    preprocessor = ArabicPreprocessor()
    train_dataset = TransformerABSADataset(train_df, tokenizer, cfg.max_length, preprocessor)
    validation_dataset = TransformerABSADataset(validation_df, tokenizer, cfg.max_length, preprocessor)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = TransformerABSAModel(model_name=model_name, dropout=cfg.dropout).to(resolved_device)
    aspect_criterion = nn.BCEWithLogitsLoss(pos_weight=build_aspect_pos_weight(train_df).to(resolved_device))
    sentiment_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        foreach=False,
    )
    total_steps = max(1, (len(train_loader) * cfg.num_epochs) // max(cfg.gradient_accumulation_steps, 1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_state_dict = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_validation_outputs: Optional[Dict[str, Any]] = None
    stagnant_epochs = 0
    training_start = time.perf_counter()

    for epoch in range(cfg.num_epochs):
        loss_summary = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            aspect_criterion=aspect_criterion,
            sentiment_criterion=sentiment_criterion,
            config=cfg,
            device=resolved_device,
        )
        validation_outputs = collect_validation_outputs(model, validation_loader, resolved_device)
        validation_pred = (validation_outputs["aspect_probabilities"] >= 0.5).astype(int)
        validation_score = float(
            f1_score(validation_outputs["aspect_targets"].astype(int), validation_pred, average="micro", zero_division=0)
        )
        print(
            json.dumps(
                {
                    "model": model_key,
                    "epoch": epoch + 1,
                    "loss": loss_summary,
                    "validation_aspect_micro_f1@0.5": round(validation_score, 6),
                },
                ensure_ascii=False,
            )
        )

        if validation_score > best_score:
            best_score = validation_score
            best_state_dict = copy.deepcopy(model.state_dict())
            best_validation_outputs = validation_outputs
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= cfg.early_stopping_patience:
                break

    training_seconds = time.perf_counter() - training_start
    model.load_state_dict(best_state_dict)

    inference_start = time.perf_counter()
    final_outputs = collect_validation_outputs(model, validation_loader, resolved_device)
    inference_seconds = time.perf_counter() - inference_start

    gold_records = records_from_dataframe(validation_df)
    thresholds, tuned_micro_f1 = tune_aspect_thresholds(
        final_outputs["aspect_probabilities"],
        gold_records,
        granularity=cfg.threshold_granularity,
        num_passes=cfg.threshold_tuning_passes,
    )
    predictions = build_predictions(
        final_outputs["review_ids"],
        final_outputs["review_texts"],
        final_outputs["aspect_probabilities"],
        final_outputs["sentiment_probabilities"],
        thresholds,
    )

    checkpoint_paths = save_checkpoint(model, tokenizer, output_dir / "checkpoint", cfg, thresholds=thresholds)
    metadata = {
        "base_model_name": model_name,
        "best_validation_aspect_micro_f1@0.5": round(best_score, 6),
        "tuned_validation_aspect_micro_f1": round(tuned_micro_f1, 6),
        "checkpoint_paths": checkpoint_paths,
    }
    if best_validation_outputs is not None:
        metadata["best_epoch_validation_shape"] = {
            "aspect_probabilities": list(best_validation_outputs["aspect_probabilities"].shape),
            "sentiment_probabilities": list(best_validation_outputs["sentiment_probabilities"].shape),
        }

    return ModelPredictionBundle(
        model_name=model_key,
        model_family="transformer",
        predictions=predictions,
        inference_seconds=inference_seconds,
        review_ids=list(final_outputs["review_ids"]),
        aspect_probabilities=final_outputs["aspect_probabilities"],
        sentiment_probabilities=final_outputs["sentiment_probabilities"],
        thresholds=thresholds,
        training_seconds=training_seconds,
        metadata=metadata,
    )
