import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from tosflag import lexglue


def test_binary_collapse_from_multilabel():
    df = pd.DataFrame([
        {"text": "a", "labels": [1]},
        {"text": "b", "labels": []},
        {"text": "c", "labels": [2, 5]},
        {"text": "d", "labels": []},
    ])
    out = lexglue._enrich_lexglue(df)
    assert out["y_binary"].tolist() == [1, 0, 1, 0]


def test_type_names_mapped():
    df = pd.DataFrame([{"text": "x", "labels": [0, 7]}])
    out = lexglue._enrich_lexglue(df)
    assert "Limitation of liability" in out.iloc[0]["unfair_types"]
    assert "Arbitration" in out.iloc[0]["unfair_types"]
