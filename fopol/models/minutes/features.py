"""Causal features for appearance models, built from PlayerStat records.

Every feature on a row is computed from that player's *earlier* fixtures only,
so a frame built over a whole season can be scored by gameweek without the
future leaking into the past. Frames are used here because the rolling maths
is vectorised; nothing here escapes as a frame.

Two lessons from the first version are baked in. ``prev_played`` is the
heaviest feature in the selection stage, and after a close season it is one
meaningless zero from May -- so any gap over ``STALE_GAP_DAYS`` marks the row
stale and falls back to the longer-run rate. And players unseen in training
get an explicit ``unknown`` index rather than an arbitrary effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fopol.base.data import Position
from fopol.data.player import PlayerStat

__all__ = ["FEATURES", "STALE_GAP_DAYS", "Features", "build_features"]

STALE_GAP_DAYS = 40.0
FEATURES = (
    "prev_60plus",
    "prev_played",
    "roll3_60plus",
    "roll3_played",
    "roll6_60plus",
    "roll6_played",
    "price_z",
    "is_home",
    "is_new",
    "is_stale",
)
POSITIONS = tuple(p.value for p in Position)


@dataclass(frozen=True)
class Features:
    """Design matrix and index arrays for a set of rows, in the rows' order."""

    x: np.ndarray
    """``(n_rows, len(FEATURES))``."""
    player_idx: np.ndarray
    """Index into ``players``; ``len(players)`` means unseen in training."""
    position_idx: np.ndarray
    band: np.ndarray
    """Observed appearance band, or -1 where the fixture is unplayed."""
    players: tuple[str, ...]


def _frame(rows: list[PlayerStat], tag: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player": [str(r.player) for r in rows],
            "position": [r.position.value for r in rows],
            "season": [r.season for r in rows],
            "kickoff": [r.kickoff for r in rows],
            "is_home": [float(r.is_home) for r in rows],
            "price": [r.price if r.price is not None else np.nan for r in rows],
            "band": [r.band if r.band is not None else np.nan for r in rows],
            "tag": tag,
            "order": range(len(rows)),
        }
    )


def build_features(
    history: list[PlayerStat],
    targets: list[PlayerStat] | None = None,
    *,
    players: tuple[str, ...] | None = None,
) -> Features:
    """Features for ``targets`` given ``history`` (or for history itself).

    Args:
        history: Played fixtures the features may look back over.
        targets: Rows to featurise. ``None`` featurises ``history`` -- the
            training case. Targets need not be played.
        players: Fixed player index space from a fitted model. ``None``
            derives it from ``history``.
    """
    if any(r.kickoff is None for r in history + (targets or [])):
        raise ValueError("appearance features need `kickoff` on every PlayerStat")

    parts = [_frame(history, "history")]
    if targets is not None:
        parts.append(_frame(targets, "target"))
    df = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["player", "kickoff"])
        .reset_index(drop=True)
    )

    g = df.groupby("player", sort=False)
    played = (df.band > 0).astype(float).where(df.band.notna())
    sixty = (df.band == 2).astype(float).where(df.band.notna())

    df["prev_60plus"] = g["band"].shift(1).eq(2).astype(float)
    df["prev_played"] = g["band"].shift(1).gt(0).astype(float)
    for w in (3, 6):
        df[f"roll{w}_60plus"] = sixty.groupby(df.player, sort=False).transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"roll{w}_played"] = played.groupby(df.player, sort=False).transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
        )

    gap = g["kickoff"].diff().dt.total_seconds() / 86400.0
    df["is_stale"] = (gap.isna() | gap.gt(STALE_GAP_DAYS)).astype(float)
    stale = df.is_stale == 1.0
    df.loc[stale, "prev_60plus"] = df.loc[stale, "roll6_60plus"]
    df.loc[stale, "prev_played"] = df.loc[stale, "roll6_played"]

    df["is_new"] = (g.cumcount() == 0).astype(float)
    df["price_z"] = (
        df.price.groupby([df.season, df.position], sort=False)
        .transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
        .fillna(0.0)
    )
    df[list(FEATURES)] = df[list(FEATURES)].fillna(0.0)

    if players is None:
        players = tuple(sorted(df.loc[df.tag == "history", "player"].unique()))
    lookup = {p: i for i, p in enumerate(players)}
    unknown = len(players)

    want = df[df.tag == ("target" if targets is not None else "history")].sort_values("order")
    return Features(
        x=want[list(FEATURES)].to_numpy(dtype=np.float32),
        player_idx=want.player.map(lambda p: lookup.get(p, unknown)).to_numpy(dtype=np.int32),
        position_idx=want.position.map(POSITIONS.index).to_numpy(dtype=np.int32),
        band=want.band.fillna(-1).to_numpy(dtype=np.int32),
        players=players,
    )
