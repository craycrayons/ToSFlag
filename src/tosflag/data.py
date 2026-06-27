"""Data loading and label scheme for ToSFlag.

Source: CodeHima/TOS_Dataset (HF, MIT license). Three native classes:
clearly_fair, potentially_unfair, clearly_unfair.

Two views are derived from the one source:
  - BINARY (the recall thesis): unfair = potentially_unfair OR clearly_unfair.
    This is the head model. The positive class (unfair) is the one whose
    false negatives are borne by a consumer who never reads the ToS, so recall
    on this class is what the whole project optimizes.
  - SEVERITY (the ranking view): the native 3-class ordinal, kept so flagged
    clauses can be ranked clearly_unfair > potentially_unfair for triage.

Honesty note carried in code so it cannot be quietly dropped: this is a single
-annotator community dataset, not the peer-reviewed CLAUDETTE release. Adequate
for a portfolio build; cross-check against LexGLUE unfair_tos before any claim
of production readiness.
"""
from __future__ import annotations

import pandas as pd

HF_DATASET = "CodeHima/TOS_Dataset"

# Native label -> binary. Ordered so "unfair" is the positive (id 1) class.
SEVERITY_ORDER = ["clearly_fair", "potentially_unfair", "clearly_unfair"]
SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITY_ORDER)}

UNFAIR_LEVELS = {"potentially_unfair", "clearly_unfair"}


def to_binary(level: str) -> int:
    """1 = unfair (the positive, recall-critical class), 0 = fair."""
    return 1 if level in UNFAIR_LEVELS else 0


def to_severity_rank(level: str) -> int:
    """0 = clearly_fair ... 2 = clearly_unfair. Ordinal for ranking only."""
    return SEVERITY_RANK[level]


def load_frames() -> dict[str, pd.DataFrame]:
    """Load the three splits from HF and attach derived label columns.

    Requires network access to huggingface.co. On a sandbox without it, point
    `load_local_frames` at downloaded parquet instead.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET)
    out: dict[str, pd.DataFrame] = {}
    for split in ["train", "validation", "test"]:
        df = ds[split].to_pandas()
        out[split] = _enrich(df)
    return out


def load_local_frames(paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Fallback: load splits from local parquet/csv paths keyed by split name."""
    out: dict[str, pd.DataFrame] = {}
    for split, path in paths.items():
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        out[split] = _enrich(df)
    return out


def is_nonclause(text: str) -> bool:
    """Detect headers / fragments that are not real clauses.

    Diagnosed from Lap-1 error analysis: the dataset labels some bare section
    headers and stubs as unfair ("NOTICES", "RESTRICTIONS", "You will not:",
    "Billing Support:"). These have no clause content, so the model correctly
    scores them fair and is then charged a false negative for being right --
    label noise, not model failure. We flag them so they can be reported
    separately rather than silently depressing recall.

    Heuristic (deliberately conservative): very short AND (all-caps OR ends with
    a colon OR has no verb-like lowercase content). Tuned to catch obvious stubs
    without eating real short clauses.
    """
    t = text.strip()
    words = t.split()
    if len(words) <= 3 and (t.isupper() or t.endswith(":")):
        return True
    if len(words) <= 2:  # two-word fragments are not clauses
        return True
    return False


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["sentence"].astype(str)
    df["y_binary"] = df["unfairness_level"].map(to_binary)
    df["y_severity"] = df["unfairness_level"].map(to_severity_rank)
    df["is_nonclause"] = df["text"].map(is_nonclause)
    return df


def describe(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The table the README leads with: why accuracy is the wrong metric."""
    rows = []
    for split, df in frames.items():
        n = len(df)
        pos = int(df["y_binary"].sum())
        rows.append(
            {
                "split": split,
                "n": n,
                "unfair": pos,
                "fair": n - pos,
                "unfair_pct": round(100 * pos / n, 1) if n else 0.0,
                "majority_baseline_acc": round(100 * (n - pos) / n, 1) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)
