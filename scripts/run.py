"""Train ToSFlag, choose the operating point on the cost asymmetry, evaluate.

Run on a machine with HF access:
    python scripts/run.py
Or against local parquet:
    python scripts/run.py --local data/train.parquet data/validation.parquet data/test.parquet

Writes:
    reports/distribution.csv   -- class balance per split (the "why not accuracy" table)
    reports/operating_point.csv -- chosen threshold and the metrics it yields
    reports/comparison.csv      -- recall-first point vs an F1/accuracy-first point
    reports/error_analysis.md   -- worst false negatives (the clauses we missed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tosflag import data, model  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()

    if args.local:
        frames = data.load_local_frames(
            {"train": args.local[0], "validation": args.local[1], "test": args.local[2]}
        )
    else:
        frames = data.load_frames()

    REPORTS.mkdir(exist_ok=True)

    # 1. The table that motivates the whole metric choice.
    dist = data.describe(frames)
    dist.to_csv(REPORTS / "distribution.csv", index=False, encoding="utf-8")
    print("\nClass distribution (note majority_baseline_acc -- this is why accuracy lies):")
    print(dist.to_string(index=False))

    train, val, test = frames["train"], frames["validation"], frames["test"]

    # 2. Fit on train.
    pipe = model.build_pipeline()
    pipe.fit(train["text"], train["y_binary"])

    # 3. Choose the operating point on VALIDATION using the cost asymmetry.
    val_scores = pipe.predict_proba(val["text"])[:, 1]
    op = model.choose_threshold(val["y_binary"].values, val_scores)
    print(
        f"\nChosen threshold (FN:FP cost = {model.FN_TO_FP_COST_RATIO}:1): {op.threshold:.3f}"
        f"\n  val recall={op.recall:.3f} precision={op.precision:.3f} "
        f"f1={op.f1:.3f} flagged={op.flagged_rate:.3f}"
    )

    # 4. Evaluate that fixed threshold on TEST (no peeking).
    test_scores = pipe.predict_proba(test["text"])[:, 1]
    test_op = model.metrics_at(test["y_binary"].values, test_scores, op.threshold)

    # 4b. Evaluate again excluding mislabeled non-clause stubs (headers etc.).
    # This separates genuine model misses from dataset label noise. The gap
    # between the two recall numbers is itself a reported finding.
    clause_mask = ~test["is_nonclause"].values
    n_nonclause_unfair = int(
        (test["is_nonclause"].values & (test["y_binary"].values == 1)).sum()
    )
    test_op_clauses = model.metrics_at(
        test["y_binary"].values[clause_mask],
        test_scores[clause_mask],
        op.threshold,
    )
    print(
        f"\nLabel-noise check: {n_nonclause_unfair} 'unfair'-labelled items on the "
        f"test set are non-clause stubs (headers/fragments)."
        f"\n  recall on ALL items:      {test_op.recall:.3f}"
        f"\n  recall on real clauses:   {test_op_clauses.recall:.3f}  "
        f"(the honest number; stubs are label noise, not model failure)"
    )

    # 5. Contrast against an F1-first point, to make the trade explicit.
    f1_op = _best_f1_point(val["y_binary"].values, val_scores)
    test_f1 = model.metrics_at(test["y_binary"].values, test_scores, f1_op.threshold)

    comparison = pd.DataFrame(
        [
            {
                "operating_point": "recall_first (cost-asymmetry)",
                "threshold": round(op.threshold, 3),
                "test_recall": round(test_op.recall, 3),
                "test_precision": round(test_op.precision, 3),
                "test_f1": round(test_op.f1, 3),
                "test_flagged_rate": round(test_op.flagged_rate, 3),
            },
            {
                "operating_point": "f1_first (the field's default)",
                "threshold": round(f1_op.threshold, 3),
                "test_recall": round(test_f1.recall, 3),
                "test_precision": round(test_f1.precision, 3),
                "test_f1": round(test_f1.f1, 3),
                "test_flagged_rate": round(test_f1.flagged_rate, 3),
            },
        ]
    )
    comparison.to_csv(REPORTS / "comparison.csv", index=False, encoding="utf-8")
    pd.DataFrame([test_op.__dict__]).to_csv(REPORTS / "operating_point.csv", index=False, encoding="utf-8")
    print("\nRecall-first vs F1-first on TEST (the central result):")
    print(comparison.to_string(index=False))

    # 6. Error analysis: the clauses we MISSED at the chosen threshold.
    _write_error_analysis(test, test_scores, op.threshold)
    print(f"\nArtifacts written to {REPORTS}/")


def _best_f1_point(y_true: np.ndarray, scores: np.ndarray) -> model.Operating:
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(y_true, scores)
    p, r = p[:-1], r[:-1]
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)
    i = int(np.argmax(f1))
    return model.Operating(float(t[i]), float(p[i]), float(r[i]), float(f1[i]), 0.0)


def _write_error_analysis(test: pd.DataFrame, scores: np.ndarray, threshold: float) -> None:
    test = test.copy()
    test["score"] = scores
    test["pred"] = (scores >= threshold).astype(int)
    fn = test[(test["pred"] == 0) & (test["y_binary"] == 1)].sort_values("score")
    lines = [
        "# Error analysis: what ToSFlag misses\n",
        f"At the recall-first threshold ({threshold:.3f}), the model still misses "
        f"{len(fn)} unfair clauses on the test set. These are the false negatives - "
        "the failures that matter, since a missed unfair clause is the costly error.\n",
        "Lowest-scoring misses (the model was most confident these were fair):\n",
    ]
    for _, row in fn.head(15).iterrows():
        txt = row["text"][:300].replace("\n", " ")
        lines.append(f"- (score {row['score']:.3f}, `{row['unfairness_level']}`) {txt}")
    (REPORTS / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
