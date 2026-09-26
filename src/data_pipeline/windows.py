"""Window selection for the stage-04 datasets (train / val / test split by calendar year).

A window is ``seq_len`` consecutive hours ``[s, s + seq_len)`` whose first ``warmup_steps``
only warm the GRU state; the rest are scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data_pipeline.dataset_config import SPLIT_CODES, DatasetSettings

HOUR_S = 3600


@dataclass(frozen=True)
class WindowSelection:
    """Selected window start indices (into the full record) per split, and selection counts."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    n_candidates: int
    n_wet: int
    n_dry_kept: int
    n_straddling: int

    def starts(self, split: str) -> np.ndarray:
        """Window starts of ``split`` (``train`` / ``val`` / ``test``)."""
        if split not in SPLIT_CODES:
            raise KeyError(f"unknown split {split!r}; expected one of {list(SPLIT_CODES)}")
        return getattr(self, split)

    def counts(self) -> dict[str, int]:
        return {"n_candidates": self.n_candidates, "n_wet": self.n_wet, "n_dry_kept": self.n_dry_kept,
                "n_straddling": self.n_straddling, "n_train": int(self.train.size), "n_val": int(self.val.size),
                "n_test": int(self.test.size)}


def local_index(timestamps: Any, tz: str) -> pd.DatetimeIndex:
    """Strictly increasing local-time ``DatetimeIndex`` (tz-naive stamps are local time)."""
    index = pd.DatetimeIndex(timestamps)
    index = index.tz_localize(tz) if index.tz is None else index.tz_convert(tz)
    if len(index) > 1 and not (np.diff(index.as_unit("s").asi8) > 0).all():
        raise ValueError("timestamps must be strictly increasing")
    return index


def break_counts(epoch_s: np.ndarray) -> np.ndarray:
    """``out[i]`` = number of non-hourly steps between positions 0 and ``i`` (for contiguity tests)."""
    return np.concatenate([[0], np.cumsum(np.diff(epoch_s) != HOUR_S)])


def select_windows(timestamps: pd.DatetimeIndex, areal_mm: np.ndarray, settings: DatasetSettings) -> WindowSelection:
    """Choose train / val / test window starts ``s`` (window = hours ``[s, s + seq_len)``).

    A start is a candidate when its ``lookback_hours`` of history exist and all hours
    ``s - lookback .. s + seq_len - 1`` are hourly-contiguous, and ``month(s)`` is in season.
    Wet windows (areal total >= ``wet_window_min_mm``) are kept, dry ones with probability
    ``dry_window_keep_frac`` (seeded). The split is that of the calendar year of the LAST
    hour (test if in ``test_years``, val if in ``val_years``, else train); windows whose
    first and last hour fall in different splits are dropped. Train starts lie on a local
    wall-clock grid of ``train_stride_h`` hours, val and test starts on ``val_stride_h``.
    """
    ts = local_index(timestamps, settings.timezone)
    areal = np.nan_to_num(np.asarray(areal_mm, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0)
    if areal.size != len(ts):
        raise ValueError(f"areal_mm has {areal.size} values but timestamps has {len(ts)}")
    lookback, seq_len, n = settings.lookback_hours, settings.seq_len, len(ts)
    empty = np.zeros(0, dtype=np.int64)
    if n < lookback + seq_len:
        return WindowSelection(train=empty, val=empty, test=empty, n_candidates=0, n_wet=0, n_dry_kept=0,
                               n_straddling=0)
    starts = np.arange(lookback, n - seq_len + 1, dtype=np.int64)
    ends = starts + seq_len - 1
    breaks = break_counts(ts.as_unit("s").asi8)
    base = (breaks[ends] == breaks[starts - lookback]) & np.isin(ts.month.to_numpy()[starts], settings.season_months)
    cumulative = np.concatenate([[0.0], np.cumsum(np.maximum(areal, 0.0))])
    wet = cumulative[starts + seq_len] - cumulative[starts] >= settings.wet_window_min_mm - 1e-9
    keep_dry = np.random.default_rng(settings.seed).random(n)[starts] < settings.dry_window_keep_frac
    codes = settings.split_codes(ts.year.to_numpy())
    first, last = codes[starts], codes[ends]
    straddle = first != last
    wall_hour = ts.tz_localize(None).as_unit("s").asi8 // HOUR_S
    kept = base & (wet | keep_dry)
    selected = kept & ~straddle
    on_train_grid = wall_hour[starts] % settings.train_stride_h == 0
    on_eval_grid = wall_hour[starts] % settings.val_stride_h == 0
    return WindowSelection(
        train=starts[selected & (last == SPLIT_CODES["train"]) & on_train_grid],
        val=starts[selected & (last == SPLIT_CODES["val"]) & on_eval_grid],
        test=starts[selected & (last == SPLIT_CODES["test"]) & on_eval_grid],
        n_candidates=int(base.sum()), n_wet=int((base & wet).sum()), n_dry_kept=int((base & ~wet & keep_dry).sum()),
        n_straddling=int((kept & straddle).sum()),
    )


def window_positive_counts(labels: np.ndarray, starts: np.ndarray, seq_len: int, warmup: int) -> np.ndarray:
    """Flooded node-steps per window over its scored steps ``[s + warmup, s + seq_len)`` (int64 [W])."""
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(0, dtype=np.int64)
    per_hour = np.asarray(labels).sum(axis=1, dtype=np.int64)
    cumulative = np.concatenate([[0], np.cumsum(per_hour)])
    return cumulative[starts + seq_len] - cumulative[starts + warmup]
