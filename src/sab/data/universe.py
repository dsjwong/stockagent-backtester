"""Fixed trading universe for v1: 15 US large-caps + SPY as a benchmark reference.

Choice
------
15 liquid US large-caps spanning tech, financials, energy, healthcare, consumer staples
and retail, plus SPY (S&P 500 ETF) kept ONLY as an external reference series (it is not
part of the tradeable universe; the project's official benchmark is buy-and-hold
equal-weight of the 15 names).

Survivorship-bias caveat (deliberate, documented)
------------------------------------------------
These names were chosen in 2026 with full hindsight: every one of them survived and most
grew a lot over 2018-2026. A universe built this way overstates what any strategy --
including buy-and-hold -- would have earned versus a point-in-time selection made in
2018. For v1 we accept this and state it in every results write-up, because the goal is
a *relative* comparison (LLM agents vs classical vs buy-and-hold on the same names), not
an absolute return claim. Fixing it properly needs point-in-time index constituents,
which is out of scope.

Dates
-----
START = 2018-01-01 gives ~8 years of daily bars and covers both target regimes
(2021 bull, 2022 drawdown) plus the 2020 crash. END is "today" at download time and is
recorded in data/snapshots/MANIFEST.json, so results are reproducible from the snapshot.
"""

from __future__ import annotations

import datetime as _dt

TICKERS: tuple[str, ...] = (
    # tech / communication
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    # financials
    "JPM", "BAC",
    # energy
    "XOM", "CVX",
    # healthcare
    "JNJ", "PFE", "UNH",
    # consumer staples / retail
    "PG", "KO", "WMT",
)
# 15 names: 5 tech, 2 financials, 2 energy, 3 healthcare, 3 staples/retail. META was
# left out to limit the mega-cap-tech tilt (still heavy -- see caveat above).

BENCHMARK: str = "SPY"

ALL_SYMBOLS: tuple[str, ...] = TICKERS + (BENCHMARK,)

START: str = "2018-01-01"


def end_date() -> str:
    """Today's date (ISO). Evaluated at download time and written to the manifest."""
    return _dt.date.today().isoformat()
