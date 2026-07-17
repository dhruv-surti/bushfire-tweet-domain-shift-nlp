# bushfire-tweet-domain-shift-nlp
# When Alerts Go Unheard: Domain Shift in NLP-Based Bushfire Tweet Classification

Research paper on domain shift in crisis NLP, comparing 6 model families for classifying actionable vs noise tweets during Australian bushfire emergencies.

**Author:** Dhruv Surti, Deakin University

## Problem

Emergency responders need to identify actionable tweets (evacuation orders, damage reports, urgent needs) during disasters. Most NLP crisis classifiers are trained on general disaster data and fail on wildfire-specific language (Australian agency acronyms, local place names, bushfire jargon). This project measures that failure and tests whether small amounts of domain-specific data can fix it.

## Key Findings

| Model | Type | Params | Train Data | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| M1: BART | Zero-shot | 406M | 0 | 0.422 | 0.444 | 0.403 |
| M4: BERT-NLI | Zero-shot | 22M | 0 | 0.563 | 0.394 | 0.987 |
| M5: Phi-1.5 | LLM zero-shot | 1.3B | 0 | 0.530 | 0.427 | 0.699 |
| M6: SetFit | Few-shot | 66M | 64 | 0.773 | 0.749 | 0.798 |
| M2: RoBERTa-Mixed | Fine-tuned | 125M | 5,000 | 0.873 | 0.843 | 0.906 |
| **M3: RoBERTa-Wildfire** | **Fine-tuned (proposed)** | **125M** | **6,792** | **0.886** | **0.852** | **0.923** |

- A 1.3B parameter LLM (M5) underperforms a 66M parameter SetFit model trained on just 64 samples, model scale does not fix domain mismatch.
- Domain-specific fine-tuning (M3) achieves the best F1 and recall, and the lowest total error count.
- Evaluated on a fixed 1,000-tweet wildfire holdout set from HumAID, with McNemar's test and bootstrap confidence intervals for statistical rigor.

## Repo Contents

```
├── README.md
├── paper/
│   └── bushfire_domain_shift_paper.pdf
└── notebook/
    └── research_paper_nlp.ipynb
```

## Dataset

[HumAID](https://crisisnlp.qcri.org/humaid_dataset) — Human-Annotated Disaster Incidents Data (Alam et al., 2021), accessed via HuggingFace `QCRI/HumAID-event-type`.

## Methods

Six models spanning zero-shot NLI, LLM prompting, few-shot (SetFit), and fine-tuned (RoBERTa) approaches. Full details, statistical tests, and figures are in the paper.

## Tech Stack

Python, PyTorch, HuggingFace Transformers & Datasets, Sentence-Transformers, scikit-learn, SciPy, Matplotlib/Seaborn.

## Citation

If you reference this work, please cite the paper included in `/paper`.

## Contact

Dhruv Surti | youhavereacheddhruv@gmail.com | [https://www.linkedin.com/in/dhruv-surti-73b782390](#)
