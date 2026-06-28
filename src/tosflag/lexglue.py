"""LexGLUE UNFAIR-ToS loader, collapsed to the project's binary scheme.

Source: coastalcph/lex_glue, config "unfair_tos" (CC-BY-4.0). Sentence-level
annotation by legal experts under EU consumer law, across 50 ToS documents.
Each row carries `text` and `labels` -- a (possibly empty) list of unfair-type
ids (0..7). The binary collapse mirrors data.to_binary on the primary dataset:

    unfair = 1  if labels is non-empty (any unfair type present)
    fair   = 0  if labels is empty

Why this is the cross-check instrument, not a second training set (in mode a):
  This dataset is the peer-reviewed standard, sentence-level and expert-labelled.
  It does NOT have the header/fragment label noise found in the community
  primary dataset. So running the CLAUDETTE-trained model against it as held-out
  data tests two things at once: does the model generalise, and -- where they
  disagree -- which dataset is the one mislabelling.

The 8 unfair types, kept for category-level disagreement analysis:
  0 Limitation of liability   1 Unilateral termination   2 Unilateral change
  3 Content removal           4 Contract by using        5 Choice of law
  6 Jurisdiction              7 Arbitration
"""
from __future__ import annotations

import pandas as pd

LEXGLUE = ("coastalcph/lex_glue", "unfair_tos")

UNFAIR_TYPES = {
    0: "Limitation of liability",
    1: "Unilateral termination",
    2: "Unilateral change",
    3: "Content removal",
    4: "Contract by using",
    5: "Choice of law",
    6: "Jurisdiction",
    7: "Arbitration",
}


def load_lexglue(split: str = "test") -> pd.DataFrame:
    """Load one LexGLUE unfair_tos split with the binary collapse attached.

    Requires network access to huggingface.co.
    """
    from datasets import load_dataset

    ds = load_dataset(*LEXGLUE, split=split)
    df = ds.to_pandas()
    return _enrich_lexglue(df)


def _enrich_lexglue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # the text column is "text"; labels is an array-like of type ids
    df["text"] = df["text"].astype(str)

    def any_unfair(labels) -> int:
        try:
            return 1 if len(labels) > 0 else 0
        except TypeError:
            return 0

    def type_names(labels) -> str:
        try:
            return ", ".join(UNFAIR_TYPES.get(int(i), str(i)) for i in labels)
        except TypeError:
            return ""

    df["y_binary"] = df["labels"].map(any_unfair)
    df["unfair_types"] = df["labels"].map(type_names)
    return df


def describe_lexglue(df: pd.DataFrame) -> dict:
    n = len(df)
    pos = int(df["y_binary"].sum())
    return {
        "n": n,
        "unfair": pos,
        "fair": n - pos,
        "unfair_pct": round(100 * pos / n, 1) if n else 0.0,
    }
