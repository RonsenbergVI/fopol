"""Minutes features: causal by construction, and robust to the close-season gap."""

import pytest

from fopol.models.minutes import FEATURES, build_features


def test_features_are_strictly_causal(minutes_row):
    """Row *t*'s features use only fixtures before *t*."""
    f = build_features([minutes_row("p", g, 90) for g in range(1, 5)])
    first = dict(zip(FEATURES, f.x[0], strict=True))
    assert first["prev_played"] == 0.0 and first["roll3_60plus"] == 0.0 and first["is_new"] == 1.0


def test_rolling_features_match_hand_computation(minutes_row):
    """Rolling start and play rates agree with a hand count."""
    rows = [minutes_row("p", g, m) for g, m in enumerate([90, 0, 30, 90, 90], start=1)]
    f = dict(zip(range(5), build_features(rows).x, strict=True))
    col = {n: i for i, n in enumerate(FEATURES)}
    assert [f[i][col["prev_60plus"]] for i in range(5)] == [0.0, 1.0, 0.0, 0.0, 1.0]
    assert f[3][col["roll3_60plus"]] == pytest.approx(1 / 3)
    assert f[4][col["roll3_played"]] == pytest.approx(2 / 3)


def test_close_season_gap_falls_back_to_longer_run_form(minutes_row):
    """A striker rested on the final day in May must not look dropped in August."""
    rows = [minutes_row("p", g, 0 if g == 10 else 90) for g in range(1, 11)]
    rows.append(minutes_row("p", 1, 90, season="2025-26", day0="2025-08-09"))
    f = build_features(rows)
    col = {n: i for i, n in enumerate(FEATURES)}
    opener = f.x[-1]
    assert opener[col["is_stale"]] == 1.0
    assert opener[col["prev_played"]] == pytest.approx(opener[col["roll6_played"]])
    assert opener[col["prev_played"]] > 0.5
