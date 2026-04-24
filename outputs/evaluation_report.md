# Arabic ABSA Evaluation Report

## Run Summary
- Model: `aubmindlab/bert-base-arabertv02`
- Threshold: `0.35`
- Validation samples: `500`
- Labels: `27`

## Overall Metrics
| Metric | Value |
| --- | ---: |
| Accuracy (exact-match / subset accuracy) | 0.5800 |
| Precision (micro) | 0.8059 |
| Recall (micro) | 0.7857 |
| Micro F1 | 0.7957 |
| Macro F1 | 0.4439 |
| Weighted F1 | 0.7767 |
| ROC-AUC (micro) | 0.9772 |
| ROC-AUC (macro) | 0.9426 |
| PR-AUC / Average Precision (micro) | 0.8660 |
| PR-AUC / Average Precision (macro) | 0.6476 |
| Mean Average Precision (mAP) | 0.6476 |
| Coverage Error | 2.7440 |

Accuracy alone is misleading for imbalanced ABSA because it is exact-match accuracy here: a sample is counted as correct only when every aspect-sentiment label is right. That makes it harsh on multi-label reviews and insensitive to whether the model is improving on rare classes.

Micro F1 pools all label decisions together, so it mostly reflects performance on frequent aspect-sentiment pairs. Macro F1 gives each class equal weight and exposes weakness on minority labels. Weighted F1 sits between them by weighting each class by support, which is useful when you want a more deployment-oriented summary without fully hiding imbalance.

PR-AUC / Average Precision is usually more informative than ROC-AUC for imbalanced ABSA because it focuses on the precision-recall tradeoff over the positive class. ROC-AUC can still look healthy when negatives dominate, while PR-AUC drops quickly when the model surfaces too many false alarms on rare aspects.

## Best Performing Classes
| Class | F1 | Precision | Recall | Support |
| --- | ---: | ---: | ---: | ---: |
| `service_positive` | 0.9342 | 0.9281 | 0.9404 | 151 |
| `general_positive` | 0.8527 | 0.8333 | 0.8730 | 63 |
| `app_experience_negative` | 0.8322 | 0.7561 | 0.9254 | 67 |
| `food_positive` | 0.8136 | 0.7385 | 0.9057 | 53 |
| `service_negative` | 0.8077 | 0.7925 | 0.8235 | 102 |

## Worst Performing Classes
| Class | F1 | Precision | Recall | Support |
| --- | ---: | ---: | ---: | ---: |
| `delivery_neutral` | 0.0000 | 0.0000 | 0.0000 | 1 |
| `ambiance_neutral` | 0.0000 | 0.0000 | 0.0000 | 2 |
| `price_neutral` | 0.0000 | 0.0000 | 0.0000 | 2 |
| `general_neutral` | 0.0000 | 0.0000 | 0.0000 | 3 |
| `delivery_positive` | 0.0000 | 0.0000 | 0.0000 | 6 |

Classes with zero positive support in this split: `service_neutral`, `cleanliness_neutral`, `none_positive`, `none_negative`

## Error Analysis
- Failed samples: `210` / `500`
- False-positive samples: `137`
- False-negative samples: `156`
- Completely wrong samples: `63`
- Low-confidence failed predictions: `68`
- High-confidence wrong predictions: `102`

## Common Failure Patterns
- Possible sarcasm or strong-emphasis cues appeared in `11` failed samples. These often flip literal polarity and make aspect extraction look deceptively easy while sentiment remains wrong.
- Possible dialect markers appeared in `86` failed samples. Dialectal phrasing can drift away from pretraining distribution and reduce confidence calibration.
- Long failed reviews (at least `30` words) appeared in `52` samples. Longer texts increase aspect overlap and make label interactions harder.
- Ambiguous or mixed sentiment signals appeared in `85` failed samples. These are typical ABSA pain points when praise and criticism coexist in the same review.

Detailed sample-level failures are saved in `error_analysis.json` for manual review.
