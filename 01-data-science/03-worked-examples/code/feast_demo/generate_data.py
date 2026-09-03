"""Generate the two synthetic parquet sources for the Feast local demo.

Companion code for:
  Data Science/Worked Examples/feature-store-feast.md

Simulates a delivery-driver ETA/risk-scoring system -- the same "driver" entity
Feast's own quickstart uses ([source: Feast Quickstart](https://docs.feast.dev/getting-started/quickstart)
(checked 2026-09-02), per NOTE-17). Two feature groups, deliberately computed on
different cadences to make the "slow vs fast" distinction concrete:

  driver_stats.parquet    -- SLOW / batch features: a nightly job aggregates each
                             driver's trip history into conv_rate (accepted-trip
                             rate) and avg_daily_trips over the trailing window.
                             One row per driver per day, 14 days of history.

  driver_activity.parquet -- FAST / near-real-time features: an app-side counter
                             updated every few minutes: trips_today (resets each
                             day) and minutes_since_last_trip. 8 updates per
                             driver on the final day, a few minutes apart --
                             standing in for what a streaming job would push
                             continuously in production (Pitfalls section
                             explains why this parquet file is a *stand-in*, not
                             how you'd wire a real streaming source).

Both are written as Parquet with an explicit event_timestamp column -- Feast's
FileSource / point-in-time join keys off that column
(NOTE-17: FileSource(path=..., timestamp_field="event_timestamp")).

Run:
    .venv-feast/Scripts/python.exe generate_data.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = Path(__file__).parent / "feature_repo" / "data"
ANCHOR_FILE = DATA_DIR / "anchor_time.txt"
DRIVER_IDS = [1001, 1002, 1003, 1004, 1005]

# Anchor "now" to the actual wall-clock time this is run, truncated to the
# hour. Feast's online store enforces each FeatureView's `ttl` against the
# REAL wall clock at query time -- if this demo hardcoded a fixed calendar
# date, the online retrieval in run_demo.py would return nulls once real time
# drifts past that date's ttl window. Writing the anchor to a file lets
# run_demo.py reuse the exact same timestamp, however long after generation
# it's actually run.
NOW = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def make_driver_stats() -> pd.DataFrame:
    """SLOW features: one row per driver per day, 14 days back from NOW."""
    rng = np.random.default_rng(SEED)
    rows = []
    for driver_id in DRIVER_IDS:
        base_conv = rng.uniform(0.55, 0.95)
        base_trips = rng.uniform(8, 25)
        for day_offset in range(14, -1, -1):  # oldest first
            ts = NOW - timedelta(days=day_offset)
            # Slow drift over the two weeks so point-in-time joins are visibly
            # different at different training timestamps.
            drift = (14 - day_offset) * rng.uniform(-0.01, 0.01)
            rows.append({
                "driver_id": driver_id,
                "event_timestamp": ts,
                "conv_rate": float(np.clip(base_conv + drift, 0.0, 1.0)),
                "avg_daily_trips": float(max(0.0, base_trips + drift * 10)),
            })
    df = pd.DataFrame(rows)
    df["created"] = NOW  # Feast convention: a second timestamp for write-time dedup
    return df


def make_driver_activity() -> pd.DataFrame:
    """FAST features: several updates per driver on the final day only."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for driver_id in DRIVER_IDS:
        trips_today = 0
        for update_idx in range(8):  # every ~40 min across the final day
            ts = NOW - timedelta(days=0, hours=(7 - update_idx) * 0.7)
            trips_today += rng.integers(0, 2)
            rows.append({
                "driver_id": driver_id,
                "event_timestamp": ts,
                "trips_today": int(trips_today),
                "minutes_since_last_trip": float(rng.uniform(1, 45)),
            })
    df = pd.DataFrame(rows)
    df["created"] = NOW
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stats_df = make_driver_stats()
    activity_df = make_driver_activity()

    stats_path = DATA_DIR / "driver_stats.parquet"
    activity_path = DATA_DIR / "driver_activity.parquet"
    stats_df.to_parquet(stats_path, index=False)
    activity_df.to_parquet(activity_path, index=False)
    ANCHOR_FILE.write_text(NOW.isoformat())

    print(f"anchor NOW = {NOW.isoformat()} -> {ANCHOR_FILE}")
    print(f"wrote {len(stats_df)} rows -> {stats_path}")
    print(f"wrote {len(activity_df)} rows -> {activity_path}")
    print(stats_df.tail(3).to_string(index=False))
    print(activity_df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
