# Arabic ABSA Evaluation Report

## Run Summary
- Model: `aubmindlab/bert-base-arabertv02`
- Threshold range: `0.55` to `0.55`
- Validation samples: `500`
- Labels: `27`

## Overall Metrics
| Metric | Value |
| --- | ---: |
| Accuracy (exact-match / subset accuracy) | 0.5360 |
| Precision (micro) | 0.8737 |
| Precision (macro) | 0.4752 |
| Precision (weighted) | 0.8258 |
| Recall (micro) | 0.7000 |
| Recall (macro) | 0.3709 |
| Recall (weighted) | 0.7000 |
| F1 (micro) | 0.7773 |
| F1 (macro) | 0.4112 |
| F1 (weighted) | 0.7523 |
| ROC-AUC (micro) | 0.9766 |
| ROC-AUC (macro) | 0.9415 |
| PR-AUC / Average Precision (micro) | 0.8625 |
| PR-AUC / Average Precision (macro) | 0.6443 |
| Mean Average Precision (mAP) | 0.6443 |
| Coverage Error | 2.7620 |

Accuracy alone is misleading for imbalanced ABSA because exact-match accuracy only counts a review as correct when every aspect-sentiment label is correct. Multi-label reviews are therefore punished harshly, even when most aspects are right.

Micro F1 pools all label decisions together and is driven by frequent classes. Macro F1 gives each class equal weight and exposes weakness on rare labels. Weighted F1 sits between them by respecting class support without letting majority classes dominate completely.

PR-AUC is more informative than ROC-AUC for imbalanced ABSA because it focuses directly on the precision-recall tradeoff for the positive class. ROC-AUC can still look strong when negatives dominate, even if the model produces too many false alarms on rare aspects.

## Best Performing Classes
| Class | F1 | Precision | Recall | Support |
| --- | ---: | ---: | ---: | ---: |
| `service_positive` | 0.9329 | 0.9456 | 0.9205 | 151 |
| `app_experience_negative` | 0.8529 | 0.8406 | 0.8657 | 67 |
| `food_positive` | 0.8317 | 0.8750 | 0.7925 | 53 |
| `general_positive` | 0.8174 | 0.9038 | 0.7460 | 63 |
| `price_negative` | 0.8036 | 0.8654 | 0.7500 | 60 |

## Worst Performing Classes
| Class | F1 | Precision | Recall | Support |
| --- | ---: | ---: | ---: | ---: |
| `delivery_neutral` | 0.0000 | 0.0000 | 0.0000 | 1 |
| `price_neutral` | 0.0000 | 0.0000 | 0.0000 | 2 |
| `ambiance_neutral` | 0.0000 | 0.0000 | 0.0000 | 2 |
| `general_neutral` | 0.0000 | 0.0000 | 0.0000 | 3 |
| `delivery_positive` | 0.0000 | 0.0000 | 0.0000 | 6 |

Classes with zero positive support in this split: `service_neutral`, `cleanliness_neutral`, `none_positive`, `none_negative`

## Error Analysis
- Curated failed examples saved: `75`
- False-positive candidates: `21`
- False-negative candidates: `152`
- Completely wrong candidates: `97`
- Low-confidence candidates: `20`
- High-confidence wrong candidates: `125`

## Common Failure Patterns
- Possible sarcasm or irony cues appeared in `28` curated failures.
- Dialect or Franco-Arabic variation appeared in `2` curated failures.
- Mixed-script or mixed-language text appeared in `3` curated failures.
- Long reviews appeared in `12` curated failures.
- Ambiguous or mixed sentiment cues appeared in `1` curated failures.

Curated examples per saved error type:
`false_positive`=15, `false_negative`=15, `completely_wrong`=15, `low_confidence`=15, `high_conf_wrong`=15

Detailed sample-level failures are saved in `error_analysis.json` for manual review.
