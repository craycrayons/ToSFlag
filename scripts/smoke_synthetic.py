"""Generate synthetic ToS-shaped data with the real ~11% unfair skew, then run
the full pipeline through it. Verifies the code is correct; the NUMBERS here are
meaningless (synthetic), only the plausibility of the mechanics matters.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tosflag import data, model

rng = np.random.default_rng(0)

UNFAIR_TEMPLATES = [
    "we may terminate your account at any time at our sole discretion without notice",
    "we reserve the right to modify these terms without prior notice to you",
    "you waive the right to a jury trial and agree to binding arbitration",
    "we are not liable for any damages arising from use of the service",
    "we may remove your content at our sole discretion for any reason",
    "any disputes shall be resolved exclusively in courts of our choosing",
    "we limit our total liability to the amount you paid in the last month",
    "by continuing to use the service you accept the revised terms automatically",
]
FAIR_TEMPLATES = [
    "you may close your account at any time from the settings page",
    "we will notify you by email before any material change takes effect",
    "this clause does not affect your statutory consumer rights",
    "please review the current pricing before placing an order",
    "you retain ownership of the content you upload to the service",
    "we provide a thirty day refund window on annual subscriptions",
    "support is available during business hours via the help centre",
    "you can export your data in a standard format whenever you wish",
]


def make_split(n, unfair_pct):
    rows = []
    for _ in range(n):
        if rng.random() < unfair_pct:
            base = rng.choice(UNFAIR_TEMPLATES)
            level = rng.choice(["potentially_unfair", "clearly_unfair"], p=[0.6, 0.4])
        else:
            base = rng.choice(FAIR_TEMPLATES)
            level = "clearly_fair"
        # add mild noise so tfidf has something to chew on
        filler = " ".join(rng.choice(["hereby", "the", "service", "user", "agreement"], size=rng.integers(0, 4)))
        rows.append({"sentence": f"{base} {filler}".strip(), "unfairness_level": level})
    return pd.DataFrame(rows)


frames = {
    "train": data._enrich(make_split(2000, 0.11)),
    "validation": data._enrich(make_split(400, 0.11)),
    "test": data._enrich(make_split(800, 0.11)),
}

print("Distribution (synthetic, should be ~11% unfair):")
print(data.describe(frames).to_string(index=False))

pipe = model.build_pipeline()
pipe.fit(frames["train"]["text"], frames["train"]["y_binary"])

val_scores = pipe.predict_proba(frames["validation"]["text"])[:, 1]
op = model.choose_threshold(frames["validation"]["y_binary"].values, val_scores)
print(f"\nChosen threshold: {op.threshold:.3f} (val recall {op.recall:.3f}, precision {op.precision:.3f})")

test_scores = pipe.predict_proba(frames["test"]["text"])[:, 1]
test_op = model.metrics_at(frames["test"]["y_binary"].values, test_scores, op.threshold)
f1_t = model._best_f1_point if False else None  # noqa
print(f"TEST recall-first: recall={test_op.recall:.3f} precision={test_op.precision:.3f} "
      f"f1={test_op.f1:.3f} flagged={test_op.flagged_rate:.3f}")

# sanity: recall-first threshold should yield recall >= an F1-optimal threshold's
from sklearn.metrics import precision_recall_curve
p, r, t = precision_recall_curve(frames["validation"]["y_binary"].values, val_scores)
p, r = p[:-1], r[:-1]
f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)
f1_t = float(t[int(np.argmax(f1))])
test_f1 = model.metrics_at(frames["test"]["y_binary"].values, test_scores, f1_t)
print(f"TEST f1-first:     recall={test_f1.recall:.3f} precision={test_f1.precision:.3f} f1={test_f1.f1:.3f}")
print(f"\nPLAUSIBILITY CHECK -- recall-first recall ({test_op.recall:.3f}) "
      f">= f1-first recall ({test_f1.recall:.3f})? {test_op.recall >= test_f1.recall}")
