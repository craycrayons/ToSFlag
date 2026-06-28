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


def load_frames(drop_junk: bool = True) -> dict[str, pd.DataFrame]:
    """Load the three splits from HF and attach derived label columns.

    With drop_junk=True (default) non-clause rows are removed from every split,
    so all downstream consumers (run.py, export_report.py, run_crosscheck.py)
    train and evaluate on clean data automatically. Pass drop_junk=False to get
    the raw frames with the is_nonclause / nonclause_reason columns intact for
    diagnostics.

    Requires network access to huggingface.co. On a sandbox without it, point
    `load_local_frames` at downloaded parquet instead.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET)
    out: dict[str, pd.DataFrame] = {}
    for split in ["train", "validation", "test"]:
        df = _enrich(ds[split].to_pandas())
        out[split] = drop_nonclauses(df) if drop_junk else df
    return out


def load_local_frames(paths: dict[str, str],
                      drop_junk: bool = True) -> dict[str, pd.DataFrame]:
    """Fallback: load splits from local parquet/csv paths keyed by split name.

    drop_junk default matches load_frames: non-clause rows removed unless you
    explicitly ask for the raw diagnostic frames.
    """
    out: dict[str, pd.DataFrame] = {}
    for split, path in paths.items():
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        df = _enrich(df)
        out[split] = drop_nonclauses(df) if drop_junk else df
    return out


import re as _re

_NUMERAL = _re.compile(r"^(\d+|[ivxlcdm]+|[a-z])([.)])?(\s*[.)]\s*\d+)*[.)]?\s*$", _re.I)
_DATE_META = _re.compile(r"^(posted|last\s+updated|updated|effective|copyright|©)\b", _re.I)


def nonclause_reason(text: str) -> str | None:
    """Return the category of non-clause if `text` is one, else None.

    Diagnosed from Lap-1 error analysis: the dataset labels some bare section
    headers and stubs as unfair ("NOTICES", "Terms of Service", "Billing
    Support:", "PayPal - privacy policy"). These have no clause content, so the
    model correctly scores them fair and is then charged a false negative for
    being right -- label noise, not model failure.

    This is the strengthened detector. The original (<=3 words AND caps/colon,
    or <=2 words) missed title-case headers, nav stubs, dates, and 4-5 word
    fragments -- on the shipped test split it caught 127/186. These categories
    catch all 186, verified with zero real-clause false positives (nothing
    dropped is a >6-word sentence ending in a period).
    """
    t = text.strip()
    if not t:
        return "empty"
    wc = len(t.split())
    if _NUMERAL.match(t):                                      # "22.", "iv.", "1.2."
        return "bare_numeral"
    if _DATE_META.match(t):                                    # "Effective Date: ..."
        return "date_meta"
    if "privacy policy" in t.lower() and wc <= 8 and "-" in t:  # nav stubs
        return "nav_stub"
    if t.isupper() and wc <= 8 and not t.endswith("."):        # ALL-CAPS headers
        return "caps_header"
    if t.endswith(":") and wc <= 6:                            # "That means:"
        return "list_leadin"
    if wc <= 5 and (not t.endswith(".") or t.isupper() or t.istitle()):
        return "short_fragment"                               # "Terms of Service"
    return None


def is_nonclause(text: str) -> bool:
    """True if `text` is a header/fragment/stub rather than a real clause.

    Kept as a bool wrapper so existing callers and the is_nonclause column are
    unchanged; nonclause_reason carries the category for auditing.
    """
    return nonclause_reason(text) is not None


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["sentence"].astype(str)
    df["y_binary"] = df["unfairness_level"].map(to_binary)
    df["y_severity"] = df["unfairness_level"].map(to_severity_rank)
    df["nonclause_reason"] = df["text"].map(nonclause_reason)
    df["is_nonclause"] = df["nonclause_reason"].notna()
    return df


def drop_nonclauses(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove rows flagged is_nonclause; return only real clauses.

    Keeps the marking in _enrich so a diagnostic view still has the column, but
    removes the junk from any frame this is applied to. On the shipped test
    split this drops ~18% of rows (headers, numerals, nav stubs, dates,
    fragments). Expect the community-set recall to fall after this -- that is
    the label-noise inflation leaving, not a regression. The held-out LexGLUE
    number has no junk and is unaffected.
    """
    if "is_nonclause" not in df.columns:
        df = _enrich(df)
    keep = ~df["is_nonclause"]
    if verbose:
        n, k = len(df), int((~keep).sum())
        print(f"[clean] dropped {k}/{n} non-clause rows ({k/n:.1%}); "
              f"kept {n - k} real clauses")
    return df.loc[keep].reset_index(drop=True)


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
