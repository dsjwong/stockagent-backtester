"""Anon-vs-real gap statistics for the forward test."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_DAYS_FOR_CI = 60          # ~3 months of daily returns; below this the CI is withheld


def gap_returns(real_equity: pd.Series, anon_equity: pd.Series) -> pd.Series:
    """Daily return difference real - anon on the common dates."""
    r = pd.concat([real_equity.pct_change(), anon_equity.pct_change()], axis=1, join="inner").dropna()
    return r.iloc[:, 0] - r.iloc[:, 1]


def bootstrap_gap(diff: pd.Series, n_boot: int = 5000, block: int = 5, seed: int = 0, alpha: float = 0.05) -> dict:
    """Circular moving-block bootstrap of the mean daily real-anon return difference, annualised x252.
    Returns n, point estimate, and the (1-alpha) CI, or ci=None while n < MIN_DAYS_FOR_CI."""
    x = diff.to_numpy(dtype=float)
    n = len(x)
    out = {"n_days": n, "mean_annualised": float(x.mean() * 252) if n else float("nan"), "ci": None, "block": block}
    if n < MIN_DAYS_FOR_CI:
        return out
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = x[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1) * 252
    out["ci"] = (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))
    return out
