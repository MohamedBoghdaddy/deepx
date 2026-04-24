"""
Dataset Module for Arabic ABSA
==============================
Handles data loading, preprocessing, and multi-label encoding.

The task is formulated as multi-label classification:
- 9 aspects × 3 sentiments = 27 possible labels
- Each review can have multiple aspect-sentiment pairs
"""

import json
import ast
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from preprocess import ArabicPreprocessor


# Define valid aspects and sentiments
VALID_ASPECTS = [
    "food", "service", "price", "cleanliness", 
    "delivery", "ambiance", "app_experience", "general", "none"
]

VALID_SENTIMENTS = ["positive", "negative", "neutral"]

# Generate all possible aspect-sentiment labels
ASPECT_SENTIMENT_LABELS = [
    f"{aspect}_{sentiment}"
    for aspect in VALID_ASPECTS
    for sentiment in VALID_SENTIMENTS
]

# Create label to index mapping
LABEL_TO_IDX = {label: idx for idx, label in enumerate(ASPECT_SENTIMENT_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}


def parse_json_column(value: Any) -> List[str]:
    """
    Safely parse JSON-like string or list from DataFrame.
    
    Args:
        value: Raw value from DataFrame (str, list, or None)
        
    Returns:
        List of strings
    """
    if pd.isna(value):
        return []
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return list(parsed.keys())
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Try to parse as Python literal
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return list(parsed.keys())
        except (ValueError, SyntaxError):
            pass
    
    return []


def parse_sentiment_dict(value: Any) -> Dict[str, str]:
    """
    Safely parse JSON-like string or dict from DataFrame.
    
    Args:
        value: Raw value from DataFrame (str, dict, or None)
        
    Returns:
        Dictionary of aspect -> sentiment
    """
    if pd.isna(value):
        return {}
    
    if isinstance(value, dict):
        return value
    
    if isinstance(value, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Try to parse as Python literal
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass
    
    return {}


def create_multi_label_vector(
    aspects: List[str], 
    sentiments: Dict[str, str]
) -> np.ndarray:
    """
    Create multi-label binary vector from aspects and sentiments.
    
    Args:
        aspects: List of aspect names
        sentiments: Dictionary mapping aspect -> sentiment
        
    Returns:
        Binary numpy array of shape (27,)
    """
    label_vector = np.zeros(len(ASPECT_SENTIMENT_LABELS), dtype=np.float32)
    
    for aspect in aspects:
        if aspect not in VALID_ASPECTS:
            continue
        
        sentiment = sentiments.get(aspect, "neutral")
        if sentiment not in VALID_SENTIMENTS:
            sentiment = "neutral"
        
        label = f"{aspect}_{sentiment}"
        if label in LABEL_TO_IDX:
            label_vector[LABEL_TO_IDX[label]] = 1.0
    
    return label_vector


def decode_multi_label_vector(
    label_vector: np.ndarray,
    threshold: float = 0.5
) -> Tuple[List[str], Dict[str, str]]:
    """
    Decode binary label vector back to aspects and sentiments.
    
    Args:
        label_vector: Binary array of shape (27,)
        threshold: Threshold for considering a label as positive
        
    Returns:
        Tuple of (aspects list, sentiments dict)
    """
    aspects = []
    sentiments = {}
    
    # Get indices where prediction exceeds threshold
    positive_indices = np.where(label_vector >= threshold)[0]
    
    for idx in positive_indices:
        label = IDX_TO_LABEL[idx]
        aspect, sentiment = label.rsplit('_', 1)
        
        if aspect not in aspects:
            aspects.append(aspect)
            sentiments[aspect] = sentiment
    
    # Handle case with no aspects detected
    if not aspects:
        aspects = ["none"]
        sentiments = {"none": "neutral"}
    
    return aspects, sentiments


class ABDataset(Dataset):
    """PyTorch Dataset for Arabic ABSA."""
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        preprocessor: Optional[ArabicPreprocessor] = None,
        is_test: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            dataframe: Pandas DataFrame with review data
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            preprocessor: Arabic text preprocessor
            is_test: Whether this is test data (no labels)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor or ArabicPreprocessor()
        self.is_test = is_test
        
        # Validate required columns
        required_cols = ['review_id', 'review_text']
        if not is_test:
            required_cols.extend(['aspects', 'aspect_sentiments'])
        
        missing = [c for c in required_cols if c not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        row = self.dataframe.iloc[idx]
        
        # Get review text and preprocess
        review_text = str(row['review_text']) if pd.notna(row['review_text']) else ""
        review_text = self.preprocessor.normalize(review_text)
        
        # Tokenize
        encoding = self.tokenizer(
            review_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'review_id': int(row['review_id']),
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
        # Add token type IDs if available
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        
        # Add labels if not test data
        if not self.is_test:
            aspects = parse_json_column(row.get('aspects', []))
            sentiments = parse_sentiment_dict(row.get('aspect_sentiments', {}))
            label_vector = create_multi_label_vector(aspects, sentiments)
            item['labels'] = torch.from_numpy(label_vector)
        
        return item


def load_data(
    train_path: str,
    val_path: str,
    test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data files.
    
    Args:
        train_path: Path to training Excel file
        val_path: Path to validation Excel file
        test_path: Path to unlabeled Excel file
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = pd.read_excel(train_path)
    val_df = pd.read_excel(val_path)
    test_df = pd.read_excel(test_path)
    
    print(f"Loaded {len(train_df)} training samples")
    print(f"Loaded {len(val_df)} validation samples")
    print(f"Loaded {len(test_df)} test samples")
    
    return train_df, val_df, test_df


# Import torch at module level for label tensor creation
import torch


# For direct script execution
if __name__ == "__main__":
    # Test the dataset module
    print("Testing dataset module...")
    print(f"Valid aspects: {VALID_ASPECTS}")
    print(f"Valid sentiments: {VALID_SENTIMENTS}")
    print(f"Total labels: {len(ASPECT_SENTIMENT_LABELS)}")
    print(f"Labels: {ASPECT_SENTIMENT_LABELS}")
    
    # Test multi-label vector creation
    aspects = ["food", "service"]
    sentiments = {"food": "positive", "service": "negative"}
    vector = create_multi_label_vector(aspects, sentiments)
    print(f"\nTest vector: {vector}")
    print(f"Non-zero indices: {np.where(vector)[0]}")
    
    # Test decoding
    decoded_aspects, decoded_sentiments = decode_multi_label_vector(vector)
    print(f"Decoded aspects: {decoded_aspects}")
    print(f"Decoded sentiments: {decoded_sentiments}")