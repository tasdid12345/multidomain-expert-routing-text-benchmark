# Soft Routing with Global Fallback vs. Hard Routing to Domain Experts

This experiment is motivated by "Multidomain Expert Ensembles for Robust AI-Generated Text Detection With Backdoor-Aware Continual Learning" (2026), which combines multiple expert detectors with a domain classifier that routes each incoming text to a relevant expert. Routing errors in such systems can send text to the wrong expert with no recourse, so we ask whether softening the routing decision and falling back to a generalist model helps. We test this idea on a public text-classification task rather than AI-text detection.

**Question:** Does soft routing with a global fallback beat hard routing to a single domain expert when a domain classifier sends multi-domain text to experts?

**Dataset:** 20 Newsgroups, downloaded through scikit-learn's `fetch_20newsgroups` over HTTP. The 20 categories are grouped into 4 domains (sci, rec, comp, talk) that serve as the experts' domains. This is not the paper's own dataset.

**Baseline:** Hard routing. A logistic-regression domain router sends each test text to one per-domain expert, and that expert's label is final.

**Improved model:** Soft routing. Expert probabilities are weighted by router probabilities. When router confidence is below a threshold chosen by cross-validation on training data only, the model falls back to a global logistic regression.

Both models use the same TF-IDF features, the same split, and the same macro-F1 metric over 20 classes, with accuracy and router error as secondary metrics.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results are written to `results.json`. Charts are saved to `figures/`.

## Results (latest run)

On 20 Newsgroups (macro-F1 over 20 classes, 4,712 held-out test texts), soft routing with a global fallback reached 75.1% versus 73.6% for a standard hard-routing mixture of domain experts, a small single-run gain of 1.5 points.

| Metric | Value |
|---|---|
| baseline_score | 0.736 |
| improved_score | 0.7511 |
| improvement | 0.01507 |
| baseline_f1 | 0.736 |
| improved_f1 | 0.7511 |
| baseline_accuracy | 0.7413 |
| improved_accuracy | 0.7574 |
| router_error_rate | 0.1324 |
| fallback_rate | 0.4665 |
| selected_threshold | 0.9 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single run (logistic regression is deterministic given seed 42; no seed sweep needed). Deviation: the official train/test split is ~40% test, so the corpus was pooled and split 75/25 stratified with seed 42 to meet the 20-30% test-size rule. Test texts are single-domain, so the multi-domain setting is simulated only through router errors. Domain grouping is a fixed, hand-chosen mapping. Backdoor-aware continual learning from the paper is not reproduced. Headers, footers and quotes are removed to reduce metadata leakage.

## Reference

This project is motivated by the following paper:

> I Miller, D Lin (2026). *Multidomain Expert Ensembles for Robust AI-Generated Text Detection With Backdoor-Aware Continual Learning*. Computer. https://ieeexplore.ieee.org/abstract/document/11599500/

Paper author: Prof. Dan Lin, Vanderbilt University (dan.lin@vanderbilt.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
