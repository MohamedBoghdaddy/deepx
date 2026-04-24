# Arabic Aspect-Based Sentiment Analysis (ABSA)

A production-oriented Arabic ABSA pipeline for multi-label aspect and sentiment prediction on review data, now refined for multilingual competition inputs including Arabic, English, French, Italian, Franco-Arabic, and emoji-heavy text.

## Overview

- Task: multi-label classification with 27 labels = 9 aspects x 3 sentiments
- Loss: `BCEWithLogitsLoss`
- Primary metrics: micro F1 and macro F1
- Output: competition-ready `submission.json`
- Multilingual preprocessing: language detection, Franco-Arabic normalization, and emoji sentiment preservation

Supported model families:

- XLM-R: recommended default for mixed-language competition data
- AraBERT: stronger for Modern Standard Arabic and cleaner text
- MARBERT: better for dialect, social media, and informal Arabic

## Dataset

Files live in the sibling `dataset/` directory:

- `DeepX_train.xlsx`
- `DeepX_validation.xlsx`
- `DeepX_unlabeled.xlsx`
- `sample_submission.json`

Training and validation columns:

- `review_id`
- `review_text`
- `star_rating`
- `date`
- `business_name`
- `business_category`
- `platform`
- `aspects`
- `aspect_sentiments`

Unlabeled columns:

- same schema without `aspects` and `aspect_sentiments`

## Preprocessing

Transformer preprocessing is intentionally light and safe:

- normalize Arabic letter variants
- remove tashkeel
- remove tatweel
- remove URLs and mentions
- keep hashtags as text
- convert key emojis into `EMO_POS`, `EMO_NEG`, and `EMO_NEU`
- detect language with `lingua-language-detector`
- normalize Franco-Arabic phrases into Arabic script when Franco density is high
- do not remove stopwords
- do not apply stemming
- avoid aggressive punctuation stripping

Generate the Franco-Arabic seed lexicon:

```bash
python src/generate_franco_seed.py
```

Preprocess the train, validation, and unlabeled datasets into `data/processed/`:

```bash
python src/preprocess_data.py
```

## Training

Train by explicit model name:

```bash
python src/train.py --model_name xlm-roberta-base --epochs 1
python src/train.py --model_name aubmindlab/bert-base-arabertv02 --epochs 1
python src/train.py --model_name UBC-NLP/MARBERT --epochs 1
```

Train by model family:

```bash
python src/train.py --model_family arabert --epochs 1
python src/train.py --model_family marbert --epochs 1
```

Recommended first smoke tests:

```bash
python src/train.py --model_name aubmindlab/bert-base-arabertv02 --epochs 1 --output_dir outputs/arabert_smoke
python src/train.py --model_name UBC-NLP/MARBERT --epochs 1 --output_dir outputs/marbert_smoke
```

Recommended final training commands:

```bash
python src/train.py --model_name aubmindlab/bert-base-arabertv02 --epochs 5 --batch_size 8 --output_dir outputs/arabert_final
python src/train.py --model_name UBC-NLP/MARBERT --epochs 5 --batch_size 8 --output_dir outputs/marbert_final
```

Checkpoints include:

- model weights
- tokenizer files
- label mapping
- threshold config
- training config
- `training_manifest.pkl` for finished-run reuse checks
- `training_state.pkl` for interrupted-run resume

Run the full pipeline in one command:

```bash
python src/run_pipeline.py
python src/run_pipeline.py --model_name aubmindlab/bert-base-arabertv02 --epochs 5 --output_dir outputs/arabert_auto
```

Pipeline behavior:

- if a compatible finished training run exists, it skips training and goes straight to testing/prediction
- if an interrupted `training_state.pkl` exists, it resumes training from the saved epoch
- if neither exists, it trains from scratch and then runs testing/prediction

## Evaluation

Evaluate a saved checkpoint:

```bash
python src/evaluate.py --model_path outputs/arabert_smoke/model.pt
python src/evaluate.py --model_path outputs/marbert_smoke/model.pt
python src/evaluate.py --model_path outputs/model.pt --test_file ../dataset/DeepX_validation.xlsx --output_dir outputs
```

This writes:

- `outputs/validation_metrics.json`
- `outputs/per_class_metrics.json`
- `outputs/error_analysis.json`
- `outputs/evaluation_report.md`

## Prediction

Generate a submission from any saved checkpoint:

```bash
python src/predict.py --model_path outputs/arabert_smoke/model.pt --output_path submission.json
python src/predict.py --model_path outputs/marbert_smoke/model.pt --output_path submission.json
```

The prediction script loads the tokenizer and model metadata from the checkpoint, so no hardcoded model name is required.

## Validation

Validate the generated submission:

```bash
python src/validator.py submission.json sample_submission.json
```

## Python API

```python
from src.dataset import load_data
from src.train import train_model
from src.predict import run_prediction

train_df, val_df, test_df = load_data(
    "../dataset/DeepX_train.xlsx",
    "../dataset/DeepX_validation.xlsx",
    "../dataset/DeepX_unlabeled.xlsx",
)

model, metrics, threshold = train_model(
    train_df,
    val_df,
    model_name="aubmindlab/bert-base-arabertv02",
    config={"num_epochs": 1, "batch_size": 8},
    output_dir="outputs/arabert_api",
)

predictions = run_prediction(
    test_df=test_df,
    model_path="outputs/arabert_api/model.pt",
    base_model_name=None,
    output_path="submission.json",
)
```

## Notes

- CUDA is used automatically when available
- If no aspect is detected, the output falls back to `["none"]` with `{"none": "neutral"}`
- Evaluation and prediction can reload either AraBERT or MARBERT checkpoints

## References

- [AraBERT](https://huggingface.co/aubmindlab/bert-base-arabertv02)
- [MARBERT](https://huggingface.co/UBC-NLP/MARBERT)
- [MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2)
