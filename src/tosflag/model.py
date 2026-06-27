"""ToSFlag model: a recall-first unfair-clause detector.

Lap 1 deliberately uses TF-IDF + Logistic Regression, not a transformer. Two
reasons, both defensible and both stated in the README:
  1. It runs end-to-end on a CPU in seconds with no GPU and no model download,
     so the repo is reproducible by anyone who clones it. (Same discipline as
     the strongest neighbor repo, which ships a cheap baseline as the real
     artifact and leaves the transformer optional.)
  2. The project's contribution is the METRIC DECISION, not the architecture.
     A heavier encoder would move the numbers but not the argument. Lap 2 swaps
     in legal-bert behind the same interface if the numbers justify it.

The thesis, made operational:
  For a consumer-protection tool, a false negative (a genuinely unfair clause
  the model calls fair) is borne by a consumer who never reads the ToS and now
  has false reassurance. A false positive (a fair clause flagged for review)
  costs a few seconds of a human skim. The costs are asymmetric, so the
  operating point is chosen to maximise recall on the unfair class subject to
  precision staying usable -- NOT to maximise accuracy or F1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import FeatureUnion, Pipeline


# How many false positives we are willing to tolerate per false negative avoided.
# 1 missed unfair clause is judged as costly as ~5 fair clauses sent to needless
# review. This single number encodes the whole thesis and is exposed, not hidden.
FN_TO_FP_COST_RATIO = 5.0


class StructuralFeatures(BaseEstimator, TransformerMixin):
    """Hand features that survive TF-IDF's lowercasing.

    Diagnosed from Lap-1 error analysis: the model missed ALL-CAPS warranty
    disclaimers ("THE SERVICE IS PROVIDED ON AN 'AS IS' BASIS") because the
    default TfidfVectorizer lowercases, collapsing "AS IS" into the stopword-ish
    tokens "as"/"is" and erasing the signal. Caps emphasis is itself a feature
    in legal drafting (limitation-of-liability blocks are conventionally
    capitalised), so we measure it directly.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            t = str(t)
            n = max(len(t), 1)
            letters = [c for c in t if c.isalpha()]
            upper_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
            rows.append(
                [
                    upper_ratio,                       # caps emphasis (the AS-IS fix)
                    1.0 if upper_ratio > 0.6 else 0.0,  # mostly-caps flag
                    min(len(t.split()) / 50.0, 1.0),    # length (headers are short)
                    1.0 if t.strip().endswith(":") else 0.0,  # header/stub marker
                ]
            )
        return csr_matrix(np.asarray(rows, dtype=float))


class _CapsTfidf(TransformerMixin, BaseEstimator):
    """A second TF-IDF that does NOT lowercase, so caps tokens get their own
    vocabulary entries (e.g. 'AS IS', 'WILL NOT BE LIABLE')."""

    def __init__(self):
        self.vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False
        )

    def fit(self, X, y=None):
        self.vec.fit(X)
        return self

    def transform(self, X):
        return self.vec.transform(X)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        # lowercased lexical signal (the original)
                        (
                            "tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=2,
                                sublinear_tf=True,
                                strip_accents="unicode",
                            ),
                        ),
                        # caps-preserving lexical signal (the AS-IS fix)
                        ("caps_tfidf", _CapsTfidf()),
                        # structural / emphasis features
                        ("structural", StructuralFeatures()),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    # Real data is ~4:1 fair:unfair (not 9:1 as the source papers
                    # report for CLAUDETTE). balanced weighting still needed so
                    # the model does not collapse to predicting "fair".
                    class_weight="balanced",
                    C=1.0,
                ),
            ),
        ]
    )


@dataclass
class Operating:
    """A chosen decision threshold and the metrics it produces on a split."""

    threshold: float
    precision: float
    recall: float
    f1: float
    flagged_rate: float  # share of clauses sent to human review at this threshold


def choose_threshold(
    y_true: np.ndarray, scores: np.ndarray, cost_ratio: float = FN_TO_FP_COST_RATIO
) -> Operating:
    """Pick the threshold that minimises expected cost under the asymmetry.

    Cost(t) = cost_ratio * FN(t) + FP(t). We sweep the precision-recall curve
    (which enumerates the achievable thresholds) and take the cost-minimising
    point. This is the thesis turned into one explicit objective rather than an
    appeal to "we care about recall."
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns one more P/R point than thresholds.
    precision, recall = precision[:-1], recall[:-1]
    pos = int(y_true.sum())
    neg = len(y_true) - pos

    best = None
    for t, p, r in zip(thresholds, precision, recall):
        if p == 0:
            continue
        fn = pos * (1 - r)
        # tp = pos*r ; fp = tp*(1-p)/p
        tp = pos * r
        fp = tp * (1 - p) / p if p > 0 else neg
        cost = cost_ratio * fn + fp
        if best is None or cost < best[0]:
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            flagged = (tp + fp) / len(y_true)
            best = (cost, Operating(float(t), float(p), float(r), float(f1), float(flagged)))
    if best is None:
        return Operating(0.5, 0.0, 0.0, 0.0, 0.0)
    return best[1]


def metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Operating:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    flagged = (tp + fp) / len(y_true)
    return Operating(threshold, precision, recall, f1, flagged)
