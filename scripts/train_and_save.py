"""Train ToSFlag once and freeze it for inference.

Fits the EXACT Lap-1 pipeline (model.build_pipeline) on cleaned train data,
derives the committed recall-first threshold on validation the same way run.py
does, and writes the model + its operating point to models/ as one unit.

This exists so inference does NOT depend on HuggingFace, on a validation split,
or on reports/operating_point.csv being present. The model and the threshold it
was chosen at travel together: retrain here and the threshold is recomputed in
the same step, so they can never silently disagree.

Run (mirrors run.py's data path exactly):
    python scripts/train_and_save.py
    python scripts/train_and_save.py --local data/train.parquet data/validation.parquet data/test.parquet

Writes:
    models/tosflag_tfidf.joblib   -- the fitted Pipeline (commit this, ~1-5 MB)
    models/tosflag_meta.json      -- {threshold, fn_fp_cost, sklearn_version, ...}

CPU-only. No GPU, no torch. This is the TF-IDF path.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tosflag import data, model  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS / "tosflag_tfidf.joblib"
META_PATH = MODELS / "tosflag_meta.json"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--local", nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()

    frames = (
        data.load_local_frames(
            {"train": args.local[0], "validation": args.local[1], "test": args.local[2]}
        )
        if args.local
        else data.load_frames()
    )
    train, val = frames["train"], frames["validation"]

    # Fit on train -- identical to run.py / export_report.py.
    pipe = model.build_pipeline().fit(train["text"], train["y_binary"])

    # Derive the operating point on VALIDATION via the cost asymmetry -- the same
    # call run.py makes. We save the resulting threshold, not a hardcoded 0.456,
    # so its provenance is this exact training run.
    val_scores = pipe.predict_proba(val["text"])[:, 1]
    op = model.choose_threshold(val["y_binary"].values, val_scores)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    meta = {
        "threshold": round(float(op.threshold), 6),
        "fn_fp_cost": model.FN_TO_FP_COST_RATIO,
        "val_recall_at_threshold": round(float(op.recall), 4),
        "val_precision_at_threshold": round(float(op.precision), 4),
        "sklearn_version": sklearn.__version__,
        "trained_at": date.today().isoformat(),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "model_file": MODEL_PATH.name,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model  -> {MODEL_PATH}  ({MODEL_PATH.stat().st_size/1e6:.2f} MB)")
    print(f"Saved meta   -> {META_PATH}")
    print(f"  threshold (FN:FP={model.FN_TO_FP_COST_RATIO}:1) = {op.threshold:.3f}"
          f"  val recall={op.recall:.3f} precision={op.precision:.3f}")
    print("\nNote: unpickling this model requires `tosflag.model` on the import path "
          "(it contains the custom _CapsTfidf / StructuralFeatures classes). "
          f"Saved with scikit-learn {sklearn.__version__}; load with a matching minor version.")


if __name__ == "__main__":
    main()
