| Method | Remove Precision | Must-Retain Recall | Pairwise agreement |
|---|---|---|---|
| Random | 0.425 | 0.575 | 0.500 |
| Untrained UniXCoder | 0.356 | 0.499 | 0.345 |
| Small LLM (Qwen2.5-Coder-1.5B) | 0.536 | 0.634 | 0.553 |
| 5.6sol | 0.862 | 0.947 | 0.907 |
| EEL | 0.821 | 0.903 | 0.863 |

Dataset labels vs engineers (151 surveyed hunks; engineer label = median of 3; Must Retain / Optional = keep, Remove = remove).
Dataset rule: a generated hunk that matches the reference patch is retained (T1/T2), otherwise not retained (T3).

| Dataset label | Hunks | Engineers agree | Engineers disagree | Disagreement |
|---|---|---|---|---|
| Retained, T1 | 33 | 28 keep | 5 remove | 15.2% |
| Retained, T2 | 46 | 46 keep | 0 remove | 0.0% |
| Retained, total | 79 | 74 keep | 5 remove | 6.3% |
| Not retained, T3 | 72 | 48 remove | 24 keep (19 Must Retain, 5 Optional) | 33.3% |
| All | 151 | 122 | 29 | 19.2% |
