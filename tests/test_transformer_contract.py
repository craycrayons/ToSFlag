"""Verify the Lap 2 transformer scorer honours the Lap 1 interface contract,
without downloading legal-bert. Uses a stub embedder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from tosflag import data, model
from tosflag.transformer import TransformerScorer, EncoderConfig


class _StubScorer(TransformerScorer):
    def _embed(self, texts):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(len(texts), 8))
        for i, t in enumerate(texts):
            if any(k in t.lower() for k in ["waiver", "terminate", "not liable"]):
                X[i, 0] += 3.0
        return X


def _toy(n):
    import pandas as pd
    rng = np.random.default_rng(2)
    U = ["shall not constitute a waiver", "we may terminate at our sole discretion"]
    F = ["you may cancel anytime", "you retain ownership"]
    rows = []
    for _ in range(n):
        if rng.random() < 0.25:
            rows.append({"sentence": rng.choice(U), "unfairness_level": "clearly_unfair"})
        else:
            rows.append({"sentence": rng.choice(F), "unfairness_level": "clearly_fair"})
    return data._enrich(pd.DataFrame(rows))


def test_scorer_returns_valid_probabilities():
    df = _toy(200)
    s = _StubScorer(EncoderConfig(mode="probe")).fit(df["text"], df["y_binary"].values)
    scores = s.predict_proba_unfair(df["text"])
    assert scores.ndim == 1 and len(scores) == len(df)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_lap1_metric_code_reused_unchanged():
    df = _toy(200)
    s = _StubScorer(EncoderConfig(mode="probe")).fit(df["text"], df["y_binary"].values)
    scores = s.predict_proba_unfair(df["text"])
    op = model.choose_threshold(df["y_binary"].values, scores, model.FN_TO_FP_COST_RATIO)
    m = model.metrics_at(df["y_binary"].values, scores, op.threshold)
    assert 0.0 <= m.recall <= 1.0 and 0.0 <= m.precision <= 1.0
