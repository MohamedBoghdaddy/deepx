# Arabic Aspect-Based Sentiment Analysis (ABSA)

A production-style Arabic ABSA system using transformer models for detecting aspects and predicting sentiments in Arabic restaurant/service reviews.

## Project Overview

This project implements a complete end-to-end Arabic Aspect-Based Sentiment Analysis system using:
- **Model**: UBC-NLP/MARBERTv2 (Arabic BERT)
- **Task Formulation**: Multi-label classification (27 labels = 9 aspects × 3 sentiments)
- **Loss**: BCEWithLogitsLoss with sigmoid outputs
- **Evaluation**: Micro F1 score

## Dataset

| File | Description | Size |
|------|-------------|------|
| `DeepX_train.xlsx` | Labeled training data | 1,971 samples |
| `DeepX_validation.xlsx` | Labeled validation data | 500 samples |
| `DeepX_unlabeled.xlsx` | Unlabeled test data | 7,047 samples |
| `sample_submission.json` | Expected output format | - |

### Data Schema

**Training/Validation columns:**
- `review_id`: Unique identifier
- `review_text`: Arabic review text
- `star_rating`: Rating (1-5)
- `date`: Review date
- `business_name`: Business name
- `business_category`: Category
- `platform`: Source platform
- `aspects`: List of detected aspects (JSON array)
- `aspect_sentiments`: Dict of aspect → sentiment (JSON object)

**Test columns:**
- Same as above but without `aspects` and `aspect_sentiments`

## Allowed Values

### Aspects
- `food` - Food quality
- `service` - Service quality
- `price` - Price/value
- `cleanliness` - Cleanliness
- `delivery` - Delivery experience
- `ambiance` - Ambiance/atmosphere
- `app_experience` - App experience (for delivery apps)
- `general` - General sentiment
- `none` - No aspect detected

### Sentiments
- `positive`
- `negative`
- `neutral`

## Project Structure

```
deepx/
├── src/
│   ├── preprocess.py    # Arabic text preprocessing
│   ├── dataset.py       # Data loading and encoding
│   ├── train.py         # Model training
│   ├── evaluate.py     # Evaluation and threshold tuning
│   ├── predict.py      # Prediction on unlabeled data
│   └── validator.py    # JSON validation
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- pandas>=2.0.0
- openpyxl>=3.1.0
- numpy>=1.24.0
- torch>=2.0.0
- transformers>=4.35.0
- arabert>=2.1.0
- scikit-learn>=1.3.0
- tqdm>=4.65.0
- scipy>=1.11.0

## Usage

### 1. Training

```python
from src.train import train_model
from src.dataset import load_data

# Load data
train_df, val_df, test_df = load_data(
    'dataset/DeepX_train.xlsx',
    'dataset/DeepX_validation.xlsx',
    'dataset/DeepX_unlabeled.xlsx'
)

# Train model
model, metrics, threshold = train_model(
    train_df,
    val_df,
    model_name='marbert',
    config={
        'max_length': 256,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 3
    },
    output_dir='outputs'
)
```

### 2. Prediction

```python
from src.predict import run_prediction
from src.dataset import load_data

# Load test data
_, _, test_df = load_data(
    'dataset/DeepX_train.xlsx',
    'dataset/DeepX_validation.xlsx',
    'dataset/DeepX_unlabeled.xlsx'
)

# Run prediction
predictions = run_prediction(
    test_df,
    model_path='outputs/model.pt',
    base_model_name='marbert',
    output_path='submission.json',
    threshold=0.5
)
```

### 3. Validation

```python
from src.validator import validate_submission, print_validation_report

# Validate submission
is_valid, report = validate_submission(
    'submission.json',
    sample_submission_path='dataset/sample_submission.json',
    test_df=test_df
)

# Print report
print_validation_report(report)
```

## Arabic Preprocessing

The preprocessing module handles:
- Removing Arabic diacritics (tashkeel)
- Normalizing Alef forms (أ, إ, آ → ا)
- Removing tatweel (ـ)
- Normalizing repeated characters
- Cleaning extra whitespace

## Model Architecture

```
Input Text → MARBERT/AraBERT → [CLS] Token → Dropout → Linear(768→27) → Sigmoid → Predictions
```

Each of the 27 output values represents the probability of an aspect-sentiment pair:
- `food_positive`, `food_negative`, `food_neutral`
- `service_positive`, `service_negative`, `service_neutral`
- ... (9 aspects × 3 sentiments)

## Output Format

```json
[
  {
    "review_id": 23,
    "aspects": ["service", "ambiance", "food"],
    "aspect_sentiments": {
      "service": "positive",
      "ambiance": "positive",
      "food": "negative"
    }
  }
]
```

## Notes

- Column names are automatically detected from the Excel files
- If no aspects are detected, the system outputs `["none"]` with `{"none": "neutral"}`
- Threshold is tuned on validation data for optimal F1 score
- The model uses GPU if available, otherwise falls back to CPU

## References

- [MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2)
- [AraBERT](https://huggingface.co/aubmindlab/bert-base-arabertv02)