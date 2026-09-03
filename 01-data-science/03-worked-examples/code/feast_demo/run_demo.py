"""Drive the local Feast demo end to end: apply, materialize, then retrieve
features BOTH ways -- get_historical_features() (training) and
get_online_features() (serving) -- through the same FeatureStore object.

Companion code for:
  Data Science/Worked Examples/feature-store-feast.md

Signatures confirmed against Feast 0.66.0 installed in .venv-feast
(NOTE-17-feast-api.md, checked 2026-09-02):
    FeatureStore.materialize(start_date, end_date, feature_views=None, ...)
    FeatureStore.get_historical_features(entity_df=None, features=[], ...)
        -> RetrievalJob (call .to_df())
    FeatureStore.get_online_features(features, entity_rows, ...)
        -> OnlineResponse (call .to_df())

Run (from the feast_demo/ directory, using the dedicated venv):
    .venv-feast/Scripts/python.exe generate_data.py     # writes the parquet sources
    ../../../../.venv-feast/Scripts/feast.exe -c feature_repo apply   # registers entities/views
    .venv-feast/Scripts/python.exe run_demo.py           # materialize + both retrievals

This script calls `feast apply` itself (via feast.repo_operations) so the
whole pipeline runs from one command.
"""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from feast import FeatureStore
from feast.repo_operations import apply_total

REPO_DIR = Path(__file__).parent / "feature_repo"
ANCHOR_FILE = REPO_DIR / "data" / "anchor_time.txt"
# Read back the exact "now" generate_data.py used, so entity_df timestamps
# below line up with the rows actually written to Parquet -- whatever real
# date this script happens to run on.
NOW = datetime.fromisoformat(ANCHOR_FILE.read_text().strip())


def run_apply(store: FeatureStore) -> None:
    """Equivalent of running `feast apply` from the CLI, invoked
    programmatically so this one script is the whole pipeline.

    apply_total() does os.chdir(REPO_DIR) internally and then imports every
    .py file in the repo by module name (e.g. "feature_definitions") to
    collect the Entity/FeatureView objects -- that import only resolves if
    REPO_DIR is on sys.path, which the `feast` CLI gets for free (its console
    launcher runs with cwd on sys.path) but a plain `python run_demo.py`
    invocation does not, so it's added explicitly here."""
    original_cwd = Path.cwd()
    sys.path.insert(0, str(REPO_DIR))
    try:
        repo_config = store.config
        apply_total(repo_config, REPO_DIR, skip_source_validation=False)
    finally:
        os.chdir(original_cwd)
    print("[apply] entities/feature views registered.\n")


def run_materialize(store: FeatureStore) -> None:
    """Copy feature values from the offline (Parquet) store into the online
    (SQLite) store, for the window covering all generated data."""
    start = NOW - timedelta(days=21)
    end = NOW
    store.materialize(start_date=start, end_date=end)
    print("[materialize] offline -> online copy complete.\n")


def run_historical_retrieval(store: FeatureStore) -> pd.DataFrame:
    """TRAINING path: point-in-time-correct join. For each (driver_id,
    event_timestamp) row in entity_df, Feast attaches the feature values that
    were the MOST RECENT AS OF that timestamp -- never a value from the future
    relative to that row. This is what prevents point-in-time leakage: a
    training example timestamped "3 days ago" cannot see a feature value
    computed "yesterday"."""
    entity_df = pd.DataFrame({
        "driver_id": [1001, 1001, 1002, 1003],
        # Deliberately different timestamps per row, including one well in
        # the past, to prove the join respects time per-row rather than just
        # grabbing "the latest" for every row.
        "event_timestamp": [
            NOW - timedelta(days=10),   # driver 1001, 10 days ago
            NOW,                         # driver 1001, right now
            NOW - timedelta(days=5),    # driver 1002, 5 days ago
            NOW,                         # driver 1003, right now
        ],
    })
    job = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_stats:conv_rate",
            "driver_stats:avg_daily_trips",
        ],
    )
    df = job.to_df()
    print("[get_historical_features] point-in-time-correct training rows:")
    print(df.sort_values(["driver_id", "event_timestamp"]).to_string(index=False))
    print()
    return df


def run_online_retrieval(store: FeatureStore) -> pd.DataFrame:
    """SERVING path: latest known value per entity, right now -- the fields a
    live prediction request would fetch in single-digit milliseconds from
    SQLite (Redis/DynamoDB in production), no Parquet scan involved."""
    response = store.get_online_features(
        features=[
            "driver_stats:conv_rate",
            "driver_stats:avg_daily_trips",
            "driver_activity:trips_today",
            "driver_activity:minutes_since_last_trip",
        ],
        entity_rows=[
            {"driver_id": 1001},
            {"driver_id": 1002},
            {"driver_id": 1003},
        ],
    )
    df = response.to_df()
    print("[get_online_features] latest values for a live request:")
    print(df.to_string(index=False))
    print()
    return df


def main() -> None:
    store = FeatureStore(repo_path=str(REPO_DIR))
    run_apply(store)
    # Re-open after apply so the store object sees the freshly written registry.
    store = FeatureStore(repo_path=str(REPO_DIR))
    run_materialize(store)
    hist_df = run_historical_retrieval(store)
    online_df = run_online_retrieval(store)

    # The train/serve-skew punchline: same feature (driver 1001's conv_rate),
    # two different retrieval paths, two different -- both CORRECT -- answers.
    hist_now = hist_df[
        (hist_df["driver_id"] == 1001) & (hist_df["event_timestamp"] == NOW)
    ]["conv_rate"].iloc[0]
    online_now = online_df[online_df["driver_id"] == 1001]["conv_rate"].iloc[0]
    print(f"driver 1001 conv_rate via get_historical_features (row timestamped NOW): {hist_now!r}")
    print(f"driver 1001 conv_rate via get_online_features       (latest, right now): {online_now!r}")
    # Same FeatureView, same underlying value -- but the online store round-trips
    # through the FeatureView's declared Float32 protobuf representation
    # (feature_definitions.py: Field(name="conv_rate", dtype=Float32)) while the
    # offline Parquet path stays float64, so expect agreement to ~1e-7, not
    # bit-for-bit equality. That precision gap is itself worth knowing about
    # before trusting a raw `==` between offline- and online-retrieved features.
    print(f"agree to within 1e-6 (both retrieval paths, same FeatureView): "
          f"{math.isclose(hist_now, online_now, abs_tol=1e-6)}")


if __name__ == "__main__":
    main()
