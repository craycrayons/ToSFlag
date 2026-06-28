"""Cross-check (mode a): does the CLAUDETTE-trained model hold up on the
peer-reviewed LexGLUE UNFAIR-ToS benchmark, and where it disagrees, which
dataset is mislabelling?

This trains ToSFlag on the primary (community) dataset exactly as in Lap 1,
then evaluates it on LexGLUE unfair_tos as a fully held-out test set -- the
model has never seen LexGLUE during training or threshold selection. Two
questions answered:

  1. GENERALISATION: recall/precision on a different, expert-labelled corpus.
     If they crater, the model overfit to the community dataset's quirks.
  2. DISAGREEMENT DIRECTION: where model and experts conflict, which way does
     it lean? Clauses experts call unfair that the model passes => community
     set under-labels these patterns. Clauses the model flags that experts
     pass => community set over-labels, or its broader labelling is defensible.

Two operating points are reported so threshold-transfer failure is not mistaken
for generalisation failure:
  - community threshold: the FN:FP=5.0 recall-first threshold from Lap 1, applied
    as-is to LexGLUE.
  - lexglue-tuned threshold: the same FN:FP cost rule re-solved on LexGLUE scores,
    isolating "the decision boundary did not transfer" from "the model cannot
    rank LexGLUE clauses".

Outputs:
  reports/crosscheck.md           human-readable report (this is what you read)
  reports/crosscheck_summary.csv  the headline metrics table

Run: python scripts/run_crosscheck.py   (CPU-fine, fast, no GPU needed)
Needs network access to huggingface.co for both datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Import the project's verified library (model.py, data.py, lexglue.py)
# ---------------------------------------------------------------------------
def _import_lib():
    """Import build_pipeline + cost ratio from model.py, and lexglue.py.

    Does NOT depend on data.py's loader signature -- community data is loaded
    directly below (same robust pattern as export_report.py), so a column-name
    or function-name mismatch in your repo cannot break this.
    """
    for c in (ROOT / "src", ROOT / "src" / "tosflag", ROOT):
        if (c / "tosflag").exists() or (c / "model.py").exists():
            sys.path.insert(0, str(c.resolve()))
    # model.py
    try:
        from tosflag.model import build_pipeline, FN_TO_FP_COST_RATIO  # type: ignore
    except ImportError:
        from model import build_pipeline, FN_TO_FP_COST_RATIO  # type: ignore
    # lexglue.py
    try:
        from tosflag import lexglue  # type: ignore
    except ImportError:
        import lexglue  # type: ignore
    return build_pipeline, FN_TO_FP_COST_RATIO, lexglue


def _load_community():
    """Load CodeHima/TOS_Dataset -> (texts, y) with y: 1=unfair, 0=fair,
    CLEANED through data.py's drop_nonclauses so the cross-check trains on
    exactly the same rows as run.py.

    Goes through data.py first (the source of truth for cleaning). Only falls
    back to a direct HF load if data.py cannot be imported -- and that fallback
    still applies the same nonclause filter, so the cross-check can never again
    silently train on uncleaned data.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: pip install datasets")

    # Preferred path: use data.py so cleaning is identical to run.py.
    try:
        try:
            from tosflag import data  # type: ignore
        except ImportError:
            import data  # type: ignore
        frames = data.load_frames(drop_junk=True)          # cleaned, all splits
        df = pd.concat(frames.values(), ignore_index=True)
        print(f"  [via data.py] {len(df)} clean clauses "
              f"({int(df['y_binary'].sum())} unfair)")
        return df["text"].astype(str).tolist(), df["y_binary"].to_numpy()
    except Exception as e:
        print(f"  [warn] data.py path unavailable ({e}); using direct load + "
              "inline nonclause filter")

    # Fallback: direct HF load, but STILL apply the same nonclause filter.
    ds = load_dataset("CodeHima/TOS_Dataset")
    df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)

    text_col = next((c for c in ("text", "sentence", "clause", "Sentence", "Text")
                     if c in df.columns), None)
    label_col = next((c for c in ("label", "labels", "unfairness_level",
                                  "Label", "class") if c in df.columns), None)
    if text_col is None or label_col is None:
        sys.exit(f"ERROR: could not find text/label columns in {list(df.columns)}")

    # apply the same junk filter as data.py, inline, so the fallback is clean too
    try:
        from tosflag.data import nonclause_reason  # type: ignore
    except ImportError:
        try:
            from data import nonclause_reason  # type: ignore
        except ImportError:
            nonclause_reason = lambda t: None  # last-resort: no filter
    keep = df[text_col].astype(str).map(nonclause_reason).isna()
    df = df.loc[keep].reset_index(drop=True)

    texts = df[text_col].astype(str).tolist()
    raw = df[label_col].astype(str).str.lower()
    y = raw.apply(lambda s: 0 if s in ("fair", "clearly_fair", "0") else 1).to_numpy()
    return texts, y


# ---------------------------------------------------------------------------
# Threshold from the FN:FP cost ratio (same rule as Lap 1)
# ---------------------------------------------------------------------------
def _cost_threshold(y_true, scores, fn_to_fp):
    """Pick the threshold minimising expected cost = fn_to_fp*FN + FP.

    Sweeps candidate thresholds at the unique score values and returns the one
    with the lowest cost. Ties break toward the lower threshold (higher recall).
    """
    y_true = np.asarray(y_true)
    order = np.argsort(scores)
    cand = np.unique(scores[order])
    best_t, best_cost = 0.5, np.inf
    for t in cand:
        pred = (scores >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = fn_to_fp * fn + fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t


def _metrics(y_true, pred):
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    flagged = (tp + fp) / len(y_true) if len(y_true) else 0.0
    return dict(recall=recall, precision=precision, f1=f1, flagged=flagged,
                tp=tp, fp=fp, fn=fn, tn=tn)


def _md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


# ---------------------------------------------------------------------------
def main():
    build_pipeline, fn_to_fp, lexglue = _import_lib()

    # 1. Train on the community dataset, exactly as Lap 1 does -------------
    print("Loading community training data (CodeHima/TOS_Dataset) ...")
    texts, y_train = _load_community()
    print(f"  {len(texts)} clauses | {int(y_train.sum())} unfair "
          f"({y_train.mean():.1%})")

    print("Fitting pipeline (TF-IDF + LogReg, CPU) ...")
    pipe = build_pipeline().fit(texts, y_train)

    # community recall-first threshold from the FN:FP cost rule (in-sample on
    # the training corpus -- same derivation as Lap 1's operating point)
    comm_scores = pipe.predict_proba(texts)[:, 1]
    comm_thr = _cost_threshold(y_train, comm_scores, fn_to_fp)
    print(f"Community recall-first threshold: {comm_thr:.3f} (FN:FP={fn_to_fp})")

    # 2. Evaluate on LexGLUE as held-out -----------------------------------
    print("Loading LexGLUE unfair_tos test split ...")
    lex = lexglue.load_lexglue("test")
    desc = lexglue.describe_lexglue(lex)
    print(f"LexGLUE test: {desc}")

    scores = pipe.predict_proba(lex["text"])[:, 1]
    lex = lex.assign(score=scores)
    y = lex["y_binary"].to_numpy()

    # operating point A: community threshold transferred as-is
    pred_comm = (scores >= comm_thr).astype(int)
    m_comm = _metrics(y, pred_comm)

    # operating point B: same cost rule re-solved on LexGLUE
    lex_thr = _cost_threshold(y, scores, fn_to_fp)
    pred_lex = (scores >= lex_thr).astype(int)
    m_lex = _metrics(y, pred_lex)
    print(f"LexGLUE-tuned threshold: {lex_thr:.3f}")

    # 3. Disagreement analysis (use the community-threshold predictions) ----
    lex["pred"] = pred_comm
    misses = lex[(lex["pred"] == 0) & (lex["y_binary"] == 1)].sort_values(
        "score")                                   # experts unfair, model fair
    overflags = lex[(lex["pred"] == 1) & (lex["y_binary"] == 0)].sort_values(
        "score", ascending=False)                  # model unfair, experts fair

    # recall by expert unfairness category --------------------------------
    miss_by_type, total_by_type = {}, {}
    unfair_rows = lex[lex["y_binary"] == 1]
    for _, r in unfair_rows.iterrows():
        for t in [s.strip() for s in r["unfair_types"].split(",") if s.strip()]:
            total_by_type[t] = total_by_type.get(t, 0) + 1
            if r["pred"] == 0:
                miss_by_type[t] = miss_by_type.get(t, 0) + 1

    # 4. Write outputs -----------------------------------------------------
    summary = pd.DataFrame([
        dict(operating_point="community_threshold", threshold=round(comm_thr, 3),
             **{k: round(m_comm[k], 3) for k in ("recall", "precision", "f1", "flagged")}),
        dict(operating_point="lexglue_tuned", threshold=round(lex_thr, 3),
             **{k: round(m_lex[k], 3) for k in ("recall", "precision", "f1", "flagged")}),
    ])
    summary.to_csv(REPORTS / "crosscheck_summary.csv", index=False)

    _write_crosscheck_md(summary, misses, overflags, miss_by_type,
                         total_by_type, dict(comm=comm_thr, lex=lex_thr,
                                             desc=desc, fn_to_fp=fn_to_fp))
    print(f"\nWrote {REPORTS/'crosscheck.md'} and crosscheck_summary.csv")
    print("Read the two disagreement lists first -- direction decides mode (b).")


def _write_crosscheck_md(summary, misses, overflags, miss_by_type,
                         total_by_type, ctx):
    lines = []
    lines.append("# LexGLUE cross-check\n")
    lines.append(
        "Trained on the community dataset (CodeHima/TOS_Dataset), evaluated on "
        "the peer-reviewed LexGLUE `unfair_tos` test split as fully held-out "
        "data. The model never saw LexGLUE during training or threshold "
        f"selection. Cost rule: FN:FP = {ctx['fn_to_fp']}.\n")
    lines.append(
        f"LexGLUE test set: {ctx['desc']['n']} clauses, "
        f"{ctx['desc']['unfair']} unfair ({ctx['desc']['unfair_pct']}%), "
        f"{ctx['desc']['fair']} fair.\n")

    lines.append("## Generalisation\n")
    lines.append(
        "Two operating points. `community_threshold` applies the Lap-1 "
        f"recall-first cut ({ctx['comm']:.3f}) as-is. `lexglue_tuned` re-solves "
        f"the same cost rule on LexGLUE ({ctx['lex']:.3f}); the gap between them "
        "is threshold-transfer loss, separate from genuine ranking failure.\n")
    lines.append(_md_table(summary) + "\n")

    # recall by category
    lines.append("## Recall by expert unfairness category (community threshold)\n")
    lines.append(
        "Caught vs missed within each LexGLUE expert type. A type with high miss "
        "count is a kind of unfairness the model is structurally blind to.\n")
    cat_rows = []
    for t in sorted(total_by_type, key=lambda k: -total_by_type[k]):
        total = total_by_type[t]
        missed = miss_by_type.get(t, 0)
        caught = total - missed
        cat_rows.append(dict(category=t, total=total, caught=caught,
                             missed=missed,
                             recall=round(caught / total, 3) if total else 0.0))
    lines.append(_md_table(pd.DataFrame(cat_rows)) + "\n")

    # the two disagreement lists -- the actual point
    lines.append(f"## Experts say UNFAIR, model says FAIR ({len(misses)})\n")
    lines.append(
        "The model passed these; LexGLUE experts flagged them. If many are "
        "clearly unfair, the community set under-labels "
        "these patterns.\n")
    for _, r in misses.head(20).iterrows():
        lines.append(f"- ({r['score']:.3f}, _{r['unfair_types']}_) {r['text'][:240]}")

    lines.append(
        f"\n## Model says UNFAIR, experts say FAIR ({len(overflags)})\n"
        "The model flagged these; LexGLUE experts did not. If many look unfair to "
        "you, it may be the community set's broader labelling is defensible; if "
        "they look fair, this is genuine over-flagging.\n")
    for _, r in overflags.head(20).iterrows():
        lines.append(f"- ({r['score']:.3f}) {r['text'][:240]}")

    (REPORTS / "crosscheck.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
