# Unlabeled Data Usage for DeepX Arabic ABSA

## Why unlabeled data is not direct ground truth

`DeepX_unlabeled.xlsx` is valuable, but it must not be treated as if it already contains correct ABSA annotations.

- A 5-star review can still mention a negative aspect like `price`.
- A 1-star review can still contain a positive aspect like `service`.
- Star ratings are coarse overall signals, while ABSA needs aspect-specific sentiment.
- Automatically trusting unlabeled predictions as gold labels can amplify model mistakes.

This project therefore uses unlabeled data only through optional enhancement workflows:

- conservative cleaning
- high-confidence pseudo-labeling
- weak sentiment signals from `star_rating`
- aspect keyword mining
- active-learning sample selection
- robustness stress testing
- domain adaptation corpus preparation

## What was added

### 1. Cleaning and loading

The unlabeled pipeline now:

- loads `review_text`, `star_rating`, `business_category`, and `platform` when available
- removes empty reviews
- removes duplicate reviews using normalized text
- preserves Arabic, English, mixed-language, Franco-Arabic, and emoji-bearing reviews
- keeps a cleaned CSV at `outputs/clean_unlabeled.csv` or the pipeline-specific output directory

### 2. Pseudo-labeling

Pseudo labels are created only from confident model predictions.

- the model predicts aspects and sentiments on cleaned unlabeled reviews
- low-confidence predictions are excluded
- exported pseudo labels include metadata and remain separate from gold labels
- training can optionally consume them with a lower loss weight

### 3. Weak signal from `star_rating`

`star_rating` is converted into a weak overall signal only:

- `4` or `5` -> weak positive
- `1` or `2` -> weak negative
- `3` -> weak neutral or mixed

This weak signal is used for:

- analysis
- contradiction checks
- active-learning prioritization
- optional filtering logic

It does not overwrite aspect-level model predictions.

### 4. Aspect keyword mining

`src/mine_keywords.py` starts from seeded aspect terms and mines frequent neighboring words and phrases from the unlabeled reviews.

This helps:

- expand rules safely
- improve post-processing later
- support project presentation with data-driven qualitative analysis

### 5. Active learning

`src/active_learning.py` exports unlabeled reviews that are most useful for manual annotation.

It prioritizes:

- low-confidence cases
- mixed sentiment
- multi-aspect reviews
- short but meaningful reviews
- multilingual reviews
- Franco-Arabic reviews
- star-rating contradictions
- aspect-confusion cases such as `general` vs `ambiance` vs `app_experience`

### 6. Robustness testing

`src/unlabeled_stress_test.py` runs the trained model on the unlabeled set and reports:

- schema validity
- multilingual coverage
- Franco-Arabic coverage
- short-review behavior
- emoji behavior
- low-confidence rates
- common aspect confusions
- failure examples
- recommendations

### 7. Domain adaptation corpus

`outputs/domain_adaptation_corpus.txt` contains one cleaned review per line for possible future continued pretraining or masked language modeling.

This corpus:

- keeps Arabic, English, Franco-Arabic, and emoji signals
- removes only empty, duplicate, and noisy rows
- avoids destructive preprocessing

## Commands

### Train baseline on labeled data only

```bash
python src/train.py \
  --train_path DeepX_train.xlsx \
  --validation_path DeepX_validation.xlsx \
  --model_name UBC-NLP/MARBERT \
  --epochs 5 \
  --batch_size 8 \
  --max_length 128 \
  --output_dir outputs
```

### Generate pseudo labels

```bash
python src/pseudo_label.py \
  --model_dir outputs \
  --unlabeled_path DeepX_unlabeled.xlsx \
  --output_path outputs/pseudo_labeled.csv \
  --confidence_threshold 0.75
```

### Mine aspect keywords

```bash
python src/mine_keywords.py \
  --unlabeled_path DeepX_unlabeled.xlsx \
  --output_path outputs/aspect_keyword_report.json
```

### Select active-learning samples

```bash
python src/active_learning.py \
  --model_dir outputs \
  --unlabeled_path DeepX_unlabeled.xlsx \
  --output_path outputs/active_learning_samples.csv \
  --num_samples 200
```

### Run robustness stress test

```bash
python src/unlabeled_stress_test.py \
  --model_dir outputs \
  --unlabeled_path DeepX_unlabeled.xlsx \
  --output_path outputs/unlabeled_stress_report.json
```

### Retrain with pseudo labels at lower weight

```bash
python src/train.py \
  --train_path DeepX_train.xlsx \
  --validation_path DeepX_validation.xlsx \
  --model_name UBC-NLP/MARBERT \
  --epochs 5 \
  --batch_size 8 \
  --max_length 128 \
  --use_pseudo_labels true \
  --pseudo_label_path outputs/pseudo_labeled.csv \
  --pseudo_label_weight 0.3 \
  --output_dir outputs_marbert_pseudo
```

### Run the full unlabeled pipeline

```bash
python src/run_unlabeled_pipeline.py \
  --model_dir outputs \
  --unlabeled_path DeepX_unlabeled.xlsx \
  --confidence_threshold 0.75 \
  --num_active_samples 200
```

## Recommended workflow for leaderboard improvement

1. Train a baseline on labeled data only.
2. Run pseudo-labeling on the unlabeled set.
3. Keep only high-confidence pseudo labels.
4. Review a small sample manually before trusting the pseudo-label pool.
5. Retrain with real labels plus pseudo labels at low weight such as `0.3`.
6. Compare validation `Micro F1`, `Macro F1`, `Weighted F1`, `Precision`, `Recall`, `PR-AUC`, and per-class `F1`.
7. Use active-learning samples to plan future manual labeling.
8. Use the stress-test report to inspect failure cases and improve rules, preprocessing, and annotation coverage.

## Presentation notes

For a project presentation, the unlabeled-data story is:

- unlabeled reviews improved coverage without pretending they were gold labels
- pseudo labels were filtered by confidence
- `star_rating` was used as a weak signal only
- keyword mining exposed new aspect vocabulary
- active learning turned uncertainty into annotation priorities
- stress testing revealed where the model still struggles
- the final submission format stayed fully compatible with the original competition schema
