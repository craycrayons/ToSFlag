import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from tosflag import data, model


def test_binary_collapse():
    assert data.to_binary("clearly_fair") == 0
    assert data.to_binary("potentially_unfair") == 1
    assert data.to_binary("clearly_unfair") == 1


def test_severity_order():
    assert data.to_severity_rank("clearly_fair") < data.to_severity_rank("clearly_unfair")


def test_higher_cost_ratio_does_not_lower_recall():
    # The real invariant: raising the FN penalty should never REDUCE the
    # recall the cost-minimiser selects. Tests the thesis mechanism directly.
    rng = np.random.default_rng(1)
    y = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([rng.uniform(0, 0.6, 90), rng.uniform(0.3, 1.0, 10)])
    r_low = model.choose_threshold(y, scores, cost_ratio=1.0).recall
    r_high = model.choose_threshold(y, scores, cost_ratio=20.0).recall
    assert r_high >= r_low


def test_metrics_at_consistency():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    op = model.metrics_at(y, scores, threshold=0.5)
    assert op.recall == 1.0 and op.precision == 1.0
