"""BiLSTM baseline runner for the Arabic ABSA benchmark."""

from __future__ import annotations

import copy
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
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


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class BiLSTMConfig:
    """Training configuration for the BiLSTM baseline."""

    max_length: int = 128
    batch_size: int = 32
    embedding_dim: int = 300
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 8
    early_stopping_patience: int = 2
    seed: int = 42
    sentiment_loss_weight: float = 1.0
    threshold_granularity: str = "aspect"
    threshold_tuning_passes: int = 2
    min_token_frequency: int = 1
    embedding_path: Optional[str] = None


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
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


class Vocabulary:
    """Simple whitespace-token vocabulary built from the training data."""

    def __init__(self, stoi: Optional[Dict[str, int]] = None) -> None:
        self.stoi = stoi or {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.itos = [token for token, _ in sorted(self.stoi.items(), key=lambda item: item[1])]

    @classmethod
    def build(
        cls,
        texts: Sequence[str],
        preprocessor: ArabicPreprocessor,
        min_frequency: int = 1,
    ) -> "Vocabulary":
        counter: Counter = Counter()
        for text in texts:
            counter.update(preprocessor.tokenize(text))

        stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token, count in counter.items():
            if count >= min_frequency and token not in stoi:
                stoi[token] = len(stoi)
        return cls(stoi)

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, tokens: Sequence[str], max_length: int) -> Tuple[List[int], int]:
        indices = [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokens[:max_length]]
        length = len(indices)
        if length < max_length:
            indices.extend([self.stoi[PAD_TOKEN]] * (max_length - length))
        return indices, max(length, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stoi": self.stoi,
        }


class BiLSTMABSADataset(Dataset):
    """Dataset for the BiLSTM benchmark model."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocabulary: Vocabulary,
        preprocessor: ArabicPreprocessor,
        max_length: int,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.mapping = infer_column_mapping(self.dataframe, require_labels=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[index]
        review_text = str(row[self.mapping.review_text])
        normalized_tokens = self.preprocessor.tokenize(review_text)
        token_ids, sequence_length = self.vocabulary.encode(normalized_tokens, self.max_length)

        aspects = parse_json_column(row[self.mapping.aspects]) if self.mapping.aspects else []
        aspect_sentiments = parse_sentiment_dict(row[self.mapping.aspect_sentiments]) if self.mapping.aspect_sentiments else {}
        aspect_targets, sentiment_targets = build_targets(aspects, aspect_sentiments)

        return {
            "review_id": row[self.mapping.review_id],
            "review_text": review_text,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "sequence_length": torch.tensor(sequence_length, dtype=torch.long),
            "aspect_targets": torch.tensor(aspect_targets, dtype=torch.float32),
            "sentiment_targets": torch.tensor(sentiment_targets, dtype=torch.long),
        }


class BiLSTMABSAModel(nn.Module):
    """Bidirectional LSTM baseline with aspect and sentiment heads."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        embedding_matrix: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.aspect_head = nn.Linear(hidden_size * 2, len(VALID_ASPECTS))
        self.sentiment_head = nn.Linear(hidden_size * 2, len(VALID_ASPECTS) * len(VALID_SENTIMENTS))

    def forward(self, token_ids: torch.Tensor, sequence_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths=sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
        pooled = self.dropout(pooled)
        aspect_logits = self.aspect_head(pooled)
        sentiment_logits = self.sentiment_head(pooled).view(-1, len(VALID_ASPECTS), len(VALID_SENTIMENTS))
        return aspect_logits, sentiment_logits


def build_aspect_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    """Build positive-class weights for aspect BCE loss."""
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


def maybe_load_pretrained_embeddings(
    vocabulary: Vocabulary,
    embedding_path: Optional[str],
    embedding_dim: int,
) -> Tuple[Optional[np.ndarray], int, bool]:
    """Load pretrained embeddings when a vector file is provided."""
    if not embedding_path:
        return None, embedding_dim, False

    path = Path(embedding_path)
    if not path.exists():
        return None, embedding_dim, False

    matched_vectors: Dict[int, np.ndarray] = {}
    inferred_dim: Optional[int] = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle):
            parts = line.rstrip().split()
            if len(parts) <= 2:
                continue
            token, vector_values = parts[0], parts[1:]
            try:
                vector = np.asarray([float(value) for value in vector_values], dtype=np.float32)
            except ValueError:
                continue
            if inferred_dim is None:
                inferred_dim = int(vector.shape[0])
            if vector.shape[0] != inferred_dim:
                continue
            token_index = vocabulary.stoi.get(token)
            if token_index is not None:
                matched_vectors[token_index] = vector
            if line_number > 2_000_000:
                break

    if inferred_dim is None:
        return None, embedding_dim, False

    matrix = np.random.normal(loc=0.0, scale=0.02, size=(len(vocabulary), inferred_dim)).astype(np.float32)
    matrix[vocabulary.stoi[PAD_TOKEN]] = 0.0
    for token_index, vector in matched_vectors.items():
        matrix[token_index] = vector
    return matrix, inferred_dim, bool(matched_vectors)


def compute_losses(
    aspect_logits: torch.Tensor,
    sentiment_logits: torch.Tensor,
    aspect_targets: torch.Tensor,
    sentiment_targets: torch.Tensor,
    aspect_criterion: nn.Module,
    sentiment_criterion: nn.Module,
    sentiment_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the BiLSTM joint loss."""
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
    model: BiLSTMABSAModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    aspect_criterion: nn.Module,
    sentiment_criterion: nn.Module,
    config: BiLSTMConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run one BiLSTM training epoch."""
    model.train()
    losses: List[Dict[str, float]] = []
    progress_bar = tqdm(dataloader, desc="Training BiLSTM", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()
        aspect_logits, sentiment_logits = model(
            batch["token_ids"].to(device),
            batch["sequence_length"].to(device),
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
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss_items)
        progress_bar.set_postfix({"loss": f"{loss_items['total_loss']:.4f}"})

    return {
        "aspect_loss": round(float(np.mean([item["aspect_loss"] for item in losses])), 6),
        "sentiment_loss": round(float(np.mean([item["sentiment_loss"] for item in losses])), 6),
        "total_loss": round(float(np.mean([item["total_loss"] for item in losses])), 6),
    }


def collect_validation_outputs(
    model: BiLSTMABSAModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Collect validation probabilities from the BiLSTM model."""
    model.eval()
    review_ids: List[Any] = []
    review_texts: List[str] = []
    aspect_probabilities: List[np.ndarray] = []
    sentiment_probabilities: List[np.ndarray] = []
    aspect_targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating BiLSTM", leave=False):
            aspect_logits, sentiment_logits = model(
                batch["token_ids"].to(device),
                batch["sequence_length"].to(device),
            )
            aspect_probabilities.extend(torch.sigmoid(aspect_logits).cpu().numpy())
            sentiment_probabilities.extend(torch.softmax(sentiment_logits, dim=-1).cpu().numpy())
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
    """Decode BiLSTM outputs into submission-compatible predictions."""
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
    model: BiLSTMABSAModel,
    vocabulary: Vocabulary,
    output_dir: Path,
    config: BiLSTMConfig,
    thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, str]:
    """Persist the BiLSTM model and vocabulary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(config),
            "thresholds": dict(thresholds or {}),
        },
        checkpoint_path,
    )
    vocab_path = output_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as handle:
        json.dump(vocabulary.to_dict(), handle, ensure_ascii=False, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(config), handle, ensure_ascii=False, indent=2)
    return {
        "checkpoint": str(checkpoint_path),
        "vocab": str(vocab_path),
    }


def run_bilstm_benchmark(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: Path,
    config: Optional[BiLSTMConfig] = None,
    device: Optional[torch.device] = None,
) -> ModelPredictionBundle:
    """Train and evaluate the BiLSTM baseline."""
    cfg = config or BiLSTMConfig()
    set_seed(cfg.seed)
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = ArabicPreprocessor()
    vocabulary = Vocabulary.build(
        [str(text) for text in train_df[infer_column_mapping(train_df, require_labels=True).review_text].tolist()],
        preprocessor=preprocessor,
        min_frequency=cfg.min_token_frequency,
    )

    embedding_matrix, actual_embedding_dim, used_pretrained_embeddings = maybe_load_pretrained_embeddings(
        vocabulary,
        cfg.embedding_path,
        cfg.embedding_dim,
    )

    train_dataset = BiLSTMABSADataset(train_df, vocabulary, preprocessor, cfg.max_length)
    validation_dataset = BiLSTMABSADataset(validation_df, vocabulary, preprocessor, cfg.max_length)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = BiLSTMABSAModel(
        vocabulary_size=len(vocabulary),
        embedding_dim=actual_embedding_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        embedding_matrix=embedding_matrix,
    ).to(resolved_device)

    aspect_criterion = nn.BCEWithLogitsLoss(pos_weight=build_aspect_pos_weight(train_df).to(resolved_device))
    sentiment_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_state_dict = copy.deepcopy(model.state_dict())
    best_score = -1.0
    stagnant_epochs = 0
    training_start = time.perf_counter()

    for epoch in range(cfg.num_epochs):
        loss_summary = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
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
                    "model": "bilstm",
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

    checkpoint_paths = save_checkpoint(model, vocabulary, output_dir / "checkpoint", cfg, thresholds=thresholds)
    metadata = {
        "used_pretrained_embeddings": used_pretrained_embeddings,
        "embedding_path": cfg.embedding_path,
        "embedding_dim": actual_embedding_dim,
        "vocab_size": len(vocabulary),
        "best_validation_aspect_micro_f1@0.5": round(best_score, 6),
        "tuned_validation_aspect_micro_f1": round(tuned_micro_f1, 6),
        "checkpoint_paths": checkpoint_paths,
    }

    return ModelPredictionBundle(
        model_name="bilstm",
        model_family="bilstm",
        predictions=predictions,
        inference_seconds=inference_seconds,
        review_ids=list(final_outputs["review_ids"]),
        aspect_probabilities=final_outputs["aspect_probabilities"],
        sentiment_probabilities=final_outputs["sentiment_probabilities"],
        thresholds=thresholds,
        training_seconds=training_seconds,
        metadata=metadata,
    )
