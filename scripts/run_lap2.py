"""Lap 2 comparison: does legal-bert close the SEMANTIC gap TF-IDF left open?

This reuses the EXACT operating-point logic from Lap 1 (model.choose_threshold,
model.metrics_at) so the two models are compared on identical terms: same data,
same cost ratio, same metric code. The only thing that changes is the scorer.

The decisive output is not "which has higher recall" -- it is `semantic_recovery`:
of the specific clauses TF-IDF missed at its recall-first threshold, how many
does legal-bert now catch? That is the test of the Lap 1 hypothesis that the
residual misses were semantic.

Run (GPU recommended for finetune; probe mode is CPU-ok):
    python scripts/run_lap2.py                 # probe mode (frozen encoder)
    python scripts/run_lap2.py --finetune      # full fine-tune
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tosflag import data, model  # noqa: E402
from tosflag.transformer import EncoderConfig, TransformerScorer  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", action="store_true", help="full fine-tune (GPU)")
    ap.add_argument("--local", nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()

    frames = (
        data.load_local_frames(
            {"train": args.local[0], "validation": args.local[1], "test": args.local[2]}
        )
        if args.local
        else data.load_frames()
    )
    train, val, test = frames["train"], frames["validation"], frames["test"]
    REPORTS.mkdir(exist_ok=True)

    cost = model.FN_TO_FP_COST_RATIO

    # --- Baseline (Lap 1): TF-IDF + structural features ---
    base = model.build_pipeline().fit(train["text"], train["y_binary"])
    base_val = base.predict_proba(val["text"])[:, 1]
    base_op = model.choose_threshold(val["y_binary"].values, base_val, cost)
    base_test = base.predict_proba(test["text"])[:, 1]
    base_m = model.metrics_at(test["y_binary"].values, base_test, base_op.threshold)

    # --- Lap 2: legal-bert scorer, SAME operating-point logic ---
    cfg = EncoderConfig(mode="finetune" if args.finetune else "probe")
    print(f"Fitting legal-bert ({cfg.mode} mode) -- this is the slow step...")
    tr = TransformerScorer(cfg).fit(train["text"], train["y_binary"].values)
    tr_val = tr.predict_proba_unfair(val["text"])
    tr_op = model.choose_threshold(val["y_binary"].values, tr_val, cost)
    tr_test = tr.predict_proba_unfair(test["text"])
    tr_m = model.metrics_at(test["y_binary"].values, tr_test, tr_op.threshold)

    comparison = pd.DataFrame(
        [
            _row("tfidf_baseline (Lap1)", base_op.threshold, base_m),
            _row(f"legal_bert_{cfg.mode} (Lap2)", tr_op.threshold, tr_m),
        ]
    )
    comparison.to_csv(REPORTS / "lap2_comparison.csv", index=False, encoding="utf-8")
    print("\nLap 1 vs Lap 2 on TEST (same cost ratio, same metric code):")
    print(comparison.to_string(index=False))

    # --- The decisive check: semantic-miss recovery ---
    y = test["y_binary"].values
    base_pred = (base_test >= base_op.threshold).astype(int)
    tr_pred = (tr_test >= tr_op.threshold).astype(int)

    # The recovery population must EXCLUDE non-clause stubs (NOTICES, Themes,
    # "You will not:", "ix.", "Apple Pay - privacy policy"). These are dataset
    # label noise -- a model "catching" them is not a semantic recovery, just an
    # accidental flag on a header. Counting them inflates the recovery rate.
    # We mirror run.py's discipline: report the honest clause-only number, and
    # the raw number alongside, so the gap is visible rather than hidden.
    is_clause = ~test["is_nonclause"].values

    base_missed_raw = (base_pred == 0) & (y == 1)        # all TF-IDF false negatives
    base_missed = base_missed_raw & is_clause            # ...that are real clauses
    recovered = base_missed & (tr_pred == 1)             # real-clause recoveries
    still_missed = base_missed & (tr_pred == 0)

    # raw (stub-inclusive) figures, kept only to show the honesty gap
    recovered_raw = base_missed_raw & (tr_pred == 1)
    n_missed_raw = int(base_missed_raw.sum())
    n_rec_raw = int(recovered_raw.sum())

    n_missed = int(base_missed.sum())
    n_rec = int(recovered.sum())
    n_stub_missed = n_missed_raw - n_missed
    print(
        f"\nSemantic-miss recovery (the Lap 1 hypothesis test):"
        f"\n  TF-IDF missed {n_missed} real unfair clauses at its recall-first point."
        f"\n  legal-bert recovers {n_rec} of {n_missed} "
        f"({100*n_rec/max(n_missed,1):.0f}%)."
    )
    # Only mention stubs if any are actually present (i.e. the frame was loaded
    # with drop_junk=False). With the default cleaned loader this is 0 and the
    # note is suppressed, since raw and clean numbers are then identical.
    if n_stub_missed > 0:
        print(
            f"  ({n_stub_missed} non-clause stubs were excluded as label noise; "
            f"raw stub-inclusive figure: {int(recovered_raw.sum())}/{n_missed_raw}.)"
        )

    # Write the recovered clauses so the qualitative claim is inspectable.
    rec_df = test[recovered][["text", "unfairness_level"]].copy()
    rec_df.to_csv(REPORTS / "lap2_recovered_clauses.csv", index=False, encoding="utf-8")
    _write_recovery_md(test, recovered, still_missed, n_missed, n_rec)
    print(f"\nArtifacts written to {REPORTS}/ (lap2_comparison, recovered_clauses, recovery.md)")


def _row(name, threshold, m):
    return {
        "model": name,
        "threshold": round(threshold, 3),
        "test_recall": round(m.recall, 3),
        "test_precision": round(m.precision, 3),
        "test_f1": round(m.f1, 3),
        "test_flagged_rate": round(m.flagged_rate, 3),
    }


def _write_recovery_md(test, recovered, still_missed, n_missed, n_rec):
    lines = [
        "# Lap 2: did legal-bert close the semantic gap?\n",
        f"Lap 1's TF-IDF model missed {n_missed} real unfair clauses at its "
        f"recall-first operating point (the dataset's non-clause header stubs are "
        f"already removed upstream by `data.py`'s cleaning, so every item here is "
        f"a real clause). The Lap 1 error analysis argued these residual misses "
        f"were SEMANTIC - waiver/responsibility clauses unfair by legal meaning, "
        f"not vocabulary. If that diagnosis was right, a legal-domain encoder "
        f"should recover them.\n",
        f"**Result: legal-bert recovers {n_rec} of {n_missed} real missed clauses "
        f"({100*n_rec/max(n_missed,1):.0f}%).**\n",
        "## Clauses recovered (TF-IDF missed, legal-bert caught)\n",
    ]
    for _, r in test[recovered].head(20).iterrows():
        lines.append(f"- (`{r['unfairness_level']}`) {r['text'][:280]}")
    lines.append("\n## Still missed by both (the residual hard cases)\n")
    for _, r in test[still_missed].head(15).iterrows():
        lines.append(f"- (`{r['unfairness_level']}`) {r['text'][:280]}")
    (REPORTS / "lap2_recovery.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
