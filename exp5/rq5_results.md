| Method | Remove Precision | Must-Retain Recall | Pairwise agreement |
|---|---|---|---|
| Random | 0.425 | 0.575 | 0.500 |
| REQ-only (untrained UniXCoder) | 0.356 | 0.499 | 0.345 |
| Small LLM (Qwen2.5-Coder-1.5B) | 0.536 | 0.634 | 0.553 |
| 5.6sol | 0.862 | 0.947 | 0.907 |
| EEL | 0.821 | 0.903 | 0.863 |
| Reference patch (upper bound) | 0.809 | 0.873 | 0.841 |

Default labels (generated hunk matches the reference patch → accept, otherwise → reject) vs engineers (median of 3):

| Default label | Engineers: Must Retain | Optional | Remove | Total |
|---|---|---|---|---|
| Accept (matches reference) | 69 | 5 | 5 | 79 |
| Reject (no match, T3) | 19 | 5 | 48 | 72 |

Accept/reject agreement: 122 / 151 = 0.808
