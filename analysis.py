"""Hard routing to one domain expert (baseline) vs. soft routing with a global
fallback (improved) on 20 Newsgroups grouped into 4 domains. CPU only, seed 42."""
import json
import os
import socket
import sys
import time
from urllib.error import HTTPError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
np.random.seed(SEED)
SOURCE_URL = "https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset"
DOMAINS = ["sci", "rec", "comp", "talk"]
# Candidate fallback thresholds on router confidence; 1.01 = always use the global model.
THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
LABELS = list(range(20))


def domain_of(category):
    """Map a 20NG category name to one of the 4 domains."""
    head = category.split(".")[0]
    return {"sci": "sci", "rec": "rec", "comp": "comp", "misc": "rec",
            "talk": "talk", "alt": "talk", "soc": "talk"}[head]


def load_data():
    """Download 20 Newsgroups with a 60 s socket timeout and retry/backoff on 429/5xx."""
    socket.setdefaulttimeout(60)  # sklearn downloads via urllib; bounds each request
    for attempt in range(5):
        try:
            return fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        except HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise
            print(f"HTTP {e.code}; retrying (attempt {attempt + 1})")
        except OSError as e:  # URLError / timeouts
            print(f"network error ({e}); retrying (attempt {attempt + 1})")
        time.sleep(5 * 2 ** attempt)
    sys.exit("ERROR: could not download 20 Newsgroups")


def lr():
    """Shared logistic-regression settings for router, experts and global model."""
    return LogisticRegression(C=10, max_iter=500, random_state=SEED)


def fit_components(Xtr, ytr, dtr, Xte):
    """Fit router, per-domain experts and global model; return probabilities on Xte.
    R (n,4): router probs. E (n,4,20): expert probs embedded in the 20-class space
    (zero outside an expert's classes). G (n,20): global 20-way model probs."""
    n = Xte.shape[0]
    R = lr().fit(Xtr, dtr).predict_proba(Xte)
    E = np.zeros((n, len(DOMAINS), 20))
    for k in range(len(DOMAINS)):
        m = dtr == k
        expert = lr().fit(Xtr[m], ytr[m])
        E[:, k, expert.classes_] = expert.predict_proba(Xte)
    gl = lr().fit(Xtr, ytr)
    G = np.zeros((n, 20))
    G[:, gl.classes_] = gl.predict_proba(Xte)
    return R, E, G


def predict_hard(R, E):
    """Baseline: route each text to its single top domain; that expert's label is final."""
    k = R.argmax(1)
    return E[np.arange(len(k)), k].argmax(1)


def predict_soft(R, E, G, tau):
    """Improved: router-weighted blend of experts; global model when router confidence < tau."""
    S = np.einsum("nk,nkc->nc", R, E)
    use_global = R.max(1) < tau
    return np.where(use_global[:, None], G, S).argmax(1)


def main():
    data = load_data()
    texts = np.array(data.data, dtype=object)
    y_all = data.target
    dom_of_cat = np.array([DOMAINS.index(domain_of(c)) for c in data.target_names])
    d_all = dom_of_cat[y_all]
    print(f"loaded {len(y_all)} documents, 20 classes, 4 domains")

    # One stratified hold-out split (25% test) on the pooled corpus, seed 42.
    idx_tr, idx_te = train_test_split(np.arange(len(y_all)), test_size=0.25,
                                      stratify=y_all, random_state=SEED)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000, stop_words="english")
    Xtr = vec.fit_transform(texts[idx_tr])  # vocabulary fit on training texts only
    Xte = vec.transform(texts[idx_te])
    ytr, dtr = y_all[idx_tr], d_all[idx_tr]
    yte, dte = y_all[idx_te], d_all[idx_te]

    # 5-fold out-of-fold predictions on training data, used only to pick the threshold.
    R_oof = np.zeros((len(ytr), 4))
    E_oof = np.zeros((len(ytr), 4, 20))
    G_oof = np.zeros((len(ytr), 20))
    for i, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr)):
        print(f"CV fold {i + 1}/5")
        R_oof[b], E_oof[b], G_oof[b] = fit_components(Xtr[a], ytr[a], dtr[a], Xtr[b])
    cv_scores = {t: f1_score(ytr, predict_soft(R_oof, E_oof, G_oof, t), average="macro", labels=LABELS)
                 for t in THRESHOLDS}
    best_t = max(THRESHOLDS, key=lambda t: cv_scores[t])  # first maximum: deterministic
    print("CV macro-F1 per threshold:", {t: round(s, 4) for t, s in cv_scores.items()})
    print(f"selected threshold (CV on train only): {best_t}")

    # Refit on full training set and evaluate both models once on the test set.
    print("refitting on full training set")
    R, E, G = fit_components(Xtr, ytr, dtr, Xte)
    base_pred = predict_hard(R, E)
    imp_pred = predict_soft(R, E, G, best_t)

    def macro_f1(p):
        return float(f1_score(yte, p, average="macro", labels=LABELS))

    base_f1, imp_f1 = macro_f1(base_pred), macro_f1(imp_pred)
    base_acc, imp_acc = float(accuracy_score(yte, base_pred)), float(accuracy_score(yte, imp_pred))
    router_err = float(np.mean(R.argmax(1) != dte))
    fallback_rate = float(np.mean(R.max(1) < best_t))
    print(f"test macro-F1: baseline={base_f1:.4f} improved={imp_f1:.4f}")
    print(f"test accuracy: baseline={base_acc:.4f} improved={imp_acc:.4f}; router error={router_err:.4f}")

    results = {
        "question": "Does soft routing with a global fallback beat hard routing to a single domain expert when a domain classifier sends multi-domain text to experts?",
        "method": "Hard-routing mixture of 4 domain experts (20-class LR per domain) vs. router-probability-weighted soft blend with a global LR fallback when router confidence is below a CV-selected threshold; TF-IDF features; macro-F1 over 20 newsgroups.",
        "dataset": {
            "name": "20 Newsgroups (scikit-learn fetch_20newsgroups, subset=all, headers/footers/quotes removed)",
            "source_url": SOURCE_URL,
            "n_rows": int(len(y_all)),
            "n_train": int(len(idx_tr)),
            "n_test": int(len(idx_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 over 20 classes",
        "higher_is_better": True,
        "baseline": "Hard routing: logistic-regression domain router sends each test text to exactly one per-domain expert whose label is final.",
        "improved": "Soft routing: expert probabilities weighted by router probabilities, with fallback to a global logistic regression when router confidence is below a threshold chosen by 5-fold CV on training data.",
        "metrics": {
            "baseline_score": base_f1,
            "improved_score": imp_f1,
            "improvement": imp_f1 - base_f1,
            "baseline_f1": base_f1,
            "improved_f1": imp_f1,
            "baseline_accuracy": base_acc,
            "improved_accuracy": imp_acc,
            "router_error_rate": router_err,
            "fallback_rate": fallback_rate,
            "selected_threshold": float(best_t),
        },
        "notes": ("Single run (logistic regression is deterministic given seed 42; no seed sweep needed). "
                  "Deviation: the official train/test split is ~40% test, so the corpus was pooled and split 75/25 "
                  "stratified with seed 42 to meet the 20-30% test-size rule. Test texts are single-domain, so the "
                  "multi-domain setting is simulated only through router errors. Domain grouping is a fixed, "
                  "hand-chosen mapping. Backdoor-aware continual learning from the paper is not reproduced. "
                  "Headers, footers and quotes are removed to reduce metadata leakage."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    ax.bar(x - 0.2, [base_f1, base_acc], 0.4, label="hard routing (baseline)")
    ax.bar(x + 0.2, [imp_f1, imp_acc], 0.4, label="soft + global fallback (improved)")
    ax.set_xticks(x, ["macro-F1", "accuracy"])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/baseline_vs_improved.png", dpi=120)
    print("wrote results.json and figures/baseline_vs_improved.png")


if __name__ == "__main__":
    main()
