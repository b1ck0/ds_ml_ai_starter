"""Regression: predicting NYC taxi fares -- companion code for
Data Science/Worked Examples/regression-nyc-taxi.md (SPEC-DS-5).

What it does:
  1. Synthesizes a realistic NYC taxi trip dataset (haversine-based distance, traffic-aware
     duration, a Manhattan-core congestion surcharge, plus a handful of deliberately dirty
     rows) -- see research/NOTE-7-nyc-taxi-dataset.md for why we synthesize instead of
     downloading the real (50GB) NYC TLC data.
  2. Cleans it (drops rows with missing GPS, non-positive fares, impossible passenger
     counts, and billing-glitch outliers) and reports what was removed.
  3. Engineers features from raw coordinates: haversine distance, and a lat/long grid
     "sector" bucket that captures the congestion-zone signal raw coordinates alone can't.
  4. Trains LinearRegression, RandomForestRegressor, HistGradientBoostingRegressor across
     three feature stages (raw coords -> +distance -> +sector) and reports MSE/RMSE/MAE/R2
     for every (stage, model) combination on ONE shared holdout split.
  5. Demonstrates empirically: (a) plain OLS predictions are scale-invariant but its
     coefficients are only meaningfully comparable once scaled; (b) KNeighborsRegressor
     (a distance-based model) degrades badly without scaling; (c) trees are unaffected
     either way.
  6. Demonstrates one-hot vs ordinal encoding: ordinal is fine (and cheaper) for a truly
     ordered category (traffic_level); using it on a nominal category (payment_type)
     silently invents a false ordering and hurts the linear model.
  7. Produces fairness diagnostics: residual histogram, residual-vs-fitted, y-vs-yhat
     parity plot, and a feature-importance bar chart (RandomForest's native importances
     next to HistGradientBoosting's permutation importances, since HGB has no native
     `feature_importances_` -- verified in research/NOTE-5-sklearn-core-apis.md).
  8. Runs three pitfall demos: location target-encoding leakage, extrapolation beyond the
     training coordinate range, and RMSE's outsized sensitivity to a single bad outlier.
  9. A light RandomizedSearchCV mention on HistGradientBoosting (small grid, few iterations).

Environment (verified in research/NOTE-2-package-versions.md and
research/NOTE-5-sklearn-core-apis.md, checked 2026-09-02):
    pandas==3.0.5, numpy==2.5.2, matplotlib==3.11.1, scipy==1.18.1, scikit-learn==1.9.0
    Python 3.12+ (this script was run and gated on Python 3.13.7 -- see chapter for exact
    version installed in this project's .venv).

Run:
    python regression_taxi.py
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

RNG_SEED = 42
CODE_DIR = Path(__file__).resolve().parent
ARTEFACTS_DIR = CODE_DIR.parent / "artefacts"
DATASETS_DIR = CODE_DIR.parent / "datasets"

# NYC bounding box used for synthesis (research/NOTE-7-nyc-taxi-dataset.md).
NYC_LAT_MIN, NYC_LAT_MAX = 40.58, 40.92
NYC_LON_MIN, NYC_LON_MAX = -74.26, -73.75

# A Manhattan-core "congestion zone" bounding box. Trips picked up inside it carry a flat
# surcharge that is NOT a function of distance or duration -- a signal raw lat/lon values
# can't express directly, but a grid-sector bucket feature can. This is what section 6 of
# the chapter (coordinate feature engineering) is built to surface.
ZONE_LAT_MIN, ZONE_LAT_MAX = 40.70, 40.79
ZONE_LON_MIN, ZONE_LON_MAX = -74.02, -73.96
CONGESTION_SURCHARGE = 2.75

GRID_N = 6  # 6x6 pickup-location grid -> up to 36 sector buckets


# ---------------------------------------------------------------------------
# 1. Haversine distance
# ---------------------------------------------------------------------------
def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray, radius_km: float = 6371.0
) -> np.ndarray:
    """Great-circle distance between two lat/lon points, in kilometres.

    Formula verified against Baeldung and Underground Mathematics (Cambridge) --
    research/NOTE-7-nyc-taxi-dataset.md. Inputs are degrees; Earth radius 6371 km (mean).
    """
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    h = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    central_angle = 2 * np.arcsin(np.sqrt(h))
    return radius_km * central_angle


# ---------------------------------------------------------------------------
# 2. Synthesize the dataset
# ---------------------------------------------------------------------------
def synthesize_trips(n_rows: int = 6000, seed: int = RNG_SEED) -> pd.DataFrame:
    """Synthesize a realistic NYC taxi trip dataset with a handful of dirty rows.

    Fare model: fare = $2.50 base + $2.50/km + $0.35/min + $2.75 congestion surcharge
    (pickup inside the Manhattan-core zone) + noise. Duration depends on distance and a
    traffic_level draw (low/medium/high), which sets an average speed. This mirrors the
    dataset decision and generator sketch in research/NOTE-7-nyc-taxi-dataset.md.
    """
    rng = np.random.default_rng(seed)

    pickup_lat = rng.uniform(NYC_LAT_MIN, NYC_LAT_MAX, n_rows)
    pickup_lon = rng.uniform(NYC_LON_MIN, NYC_LON_MAX, n_rows)
    # Dropoff = pickup + a small random offset (most trips are short hops), clamped to NYC.
    dropoff_lat = np.clip(pickup_lat + rng.normal(0, 0.045, n_rows), NYC_LAT_MIN, NYC_LAT_MAX)
    dropoff_lon = np.clip(pickup_lon + rng.normal(0, 0.045, n_rows), NYC_LON_MIN, NYC_LON_MAX)

    distance_km = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    passenger_count = rng.choice([1, 2, 3, 4, 5, 6], size=n_rows, p=[0.70, 0.15, 0.08, 0.04, 0.02, 0.01])
    payment_type = rng.choice(["card", "cash", "mobile_wallet"], size=n_rows, p=[0.60, 0.30, 0.10])
    traffic_level = rng.choice(["low", "medium", "high"], size=n_rows, p=[0.30, 0.45, 0.25])

    speed_by_level = {"low": 38.0, "medium": 24.0, "high": 11.0}
    base_speed_kmh = np.array([speed_by_level[t] for t in traffic_level])
    speed_kmh = np.clip(base_speed_kmh + rng.normal(0, 3.0, n_rows), 4.0, None)
    duration_min = np.clip((distance_km / speed_kmh) * 60.0 + rng.normal(0, 1.5, n_rows), 1.0, None)

    in_zone = (
        (pickup_lat >= ZONE_LAT_MIN)
        & (pickup_lat <= ZONE_LAT_MAX)
        & (pickup_lon >= ZONE_LON_MIN)
        & (pickup_lon <= ZONE_LON_MAX)
    )
    congestion_surcharge = np.where(in_zone, CONGESTION_SURCHARGE, 0.0)

    # Per-km rate is NOT a single global slope: short hops (<=3km) carry a higher rate (like
    # a minimum-fare psychology), and traffic level applies a surge MULTIPLIER on top of that
    # rate rather than just stretching duration. That distance<->traffic interaction plus the
    # rate kink is where a single-slope LinearRegression term structurally can't match what a
    # tree can pick up by splitting -- it's the reason boosting/bagging get a real edge in
    # section 5 of the chapter, instead of an artificially linear toy problem.
    base_fare, per_min = 2.50, 0.35
    short_trip_rate, long_trip_rate = 2.90, 2.30
    per_km = np.where(distance_km <= 3.0, short_trip_rate, long_trip_rate)
    surge_multiplier_by_level = {"low": 1.00, "medium": 1.10, "high": 1.25}
    surge_multiplier = np.array([surge_multiplier_by_level[t] for t in traffic_level])
    effective_per_km = per_km * surge_multiplier

    # Payment type carries a real, NON-monotonic adjustment (cash is the $0 baseline; card
    # carries a processing surcharge; mobile-wallet has a small promo discount) -- there is
    # no natural order to "card < cash < mobile_wallet", which is exactly what section 7
    # (encoding) needs to show why ordinal-encoding a nominal category is a mistake.
    payment_adjustment_by_type = {"cash": 0.00, "card": 1.50, "mobile_wallet": -3.00}
    payment_adjustment = np.array([payment_adjustment_by_type[p] for p in payment_type])

    noise = rng.normal(0, 1.4, n_rows)
    fare_amount = (
        base_fare
        + effective_per_km * distance_km
        + per_min * duration_min
        + congestion_surcharge
        + payment_adjustment
        + noise
    )
    fare_amount = np.maximum(fare_amount, 2.50)

    pickup_datetime = [datetime(2024, 1, 1) + timedelta(minutes=int(i * 8.7)) for i in range(n_rows)]

    df = pd.DataFrame(
        {
            "pickup_datetime": pickup_datetime,
            "pickup_lat": pickup_lat,
            "pickup_lon": pickup_lon,
            "dropoff_lat": dropoff_lat,
            "dropoff_lon": dropoff_lon,
            "passenger_count": passenger_count,
            "payment_type": payment_type,
            "traffic_level": traffic_level,
            "distance_km": distance_km,
            "duration_min": duration_min,
            "fare_amount": fare_amount,
        }
    )

    _inject_dirty_rows(df, rng)
    return df


def _inject_dirty_rows(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Mutate df in place: inject realistic data-quality problems, in disjoint row blocks."""
    n = len(df)
    all_idx = rng.permutation(n)

    n_neg_fare = int(0.007 * n)
    n_null_gps = int(0.008 * n)
    n_bad_passengers = int(0.004 * n)
    n_outlier_fare = int(0.003 * n)

    cursor = 0
    neg_fare_idx = all_idx[cursor : cursor + n_neg_fare]
    cursor += n_neg_fare
    null_gps_idx = all_idx[cursor : cursor + n_null_gps]
    cursor += n_null_gps
    bad_passenger_idx = all_idx[cursor : cursor + n_bad_passengers]
    cursor += n_bad_passengers
    outlier_fare_idx = all_idx[cursor : cursor + n_outlier_fare]

    # Sign-flip error at the payment terminal: a refund or chargeback logged as a raw negative.
    df.loc[neg_fare_idx, "fare_amount"] = -df.loc[neg_fare_idx, "fare_amount"].abs()

    # GPS dropout: half lose pickup coordinates, half lose dropoff -- fare/distance/duration
    # were already recorded by the meter before the signal was lost, so those stay populated.
    half = len(null_gps_idx) // 2
    df.loc[null_gps_idx[:half], ["pickup_lat", "pickup_lon"]] = np.nan
    df.loc[null_gps_idx[half:], ["dropoff_lat", "dropoff_lon"]] = np.nan

    # Impossible passenger counts from a faulty sensor (0, or absurdly high).
    df.loc[bad_passenger_idx[: len(bad_passenger_idx) // 2], "passenger_count"] = 0
    df.loc[bad_passenger_idx[len(bad_passenger_idx) // 2 :], "passenger_count"] = 9

    # Billing glitch: a large flat amount gets erroneously added on top of the real fare
    # (e.g. a duplicated toll/surcharge line). Additive rather than multiplicative so this
    # produces a clear outlier regardless of how cheap the original trip was.
    df.loc[outlier_fare_idx, "fare_amount"] = df.loc[outlier_fare_idx, "fare_amount"] + rng.uniform(
        400.0, 900.0, size=len(outlier_fare_idx)
    )


# ---------------------------------------------------------------------------
# 3. Clean
# ---------------------------------------------------------------------------
def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that fail basic sanity checks, reporting what was removed and why."""
    start_n = len(df)
    reasons: list[tuple[str, int]] = []

    missing_gps = df[["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"]].isna().any(axis=1)
    reasons.append(("missing GPS coordinates", int(missing_gps.sum())))
    df = df.loc[~missing_gps]

    non_positive_fare = df["fare_amount"] <= 0
    reasons.append(("non-positive fare_amount", int(non_positive_fare.sum())))
    df = df.loc[~non_positive_fare]

    invalid_passengers = ~df["passenger_count"].between(1, 6)
    reasons.append(("passenger_count outside 1-6", int(invalid_passengers.sum())))
    df = df.loc[~invalid_passengers]

    # Domain sanity cap, not a statistical one: even a worst-case legitimate trip in this
    # dataset (~20 km at crawling 4 km/h traffic) tops out well under $150; the injected
    # billing-glitch rows add $400-$900 on top of the real fare, so $250 cleanly separates
    # the two without touching any genuine long/slow trip.
    unrealistic_fare = df["fare_amount"] > 250
    reasons.append(("fare_amount > $250 (billing glitch)", int(unrealistic_fare.sum())))
    df = df.loc[~unrealistic_fare]

    print("\n=== cleaning report ===")
    print(f"rows before cleaning: {start_n}")
    for reason, count in reasons:
        print(f"  removed for {reason}: {count}")
    print(f"rows after cleaning: {len(df)} ({start_n - len(df)} removed total)")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Feature engineering: grid sector bucketing
# ---------------------------------------------------------------------------
def add_pickup_sector(df: pd.DataFrame, grid_n: int = GRID_N) -> pd.DataFrame:
    """Bucket pickup_lat/pickup_lon into a grid_n x grid_n grid over the NYC bounding box.

    This is the "map-sector" feature: a coarse neighbourhood id that lets a model learn
    location effects (like the congestion surcharge) that raw continuous lat/lon can't
    express without a very deep tree or a lot of data.
    """
    df = df.copy()
    lat_bins = np.linspace(NYC_LAT_MIN, NYC_LAT_MAX, grid_n + 1)
    lon_bins = np.linspace(NYC_LON_MIN, NYC_LON_MAX, grid_n + 1)

    row = np.clip(np.digitize(df["pickup_lat"], lat_bins) - 1, 0, grid_n - 1)
    col = np.clip(np.digitize(df["pickup_lon"], lon_bins) - 1, 0, grid_n - 1)
    df["pickup_sector"] = [f"R{r}C{c}" for r, c in zip(row, col)]
    return df


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MSE, RMSE, MAE, R2 -- verified sklearn 1.9.0 APIs, research/NOTE-5-sklearn-core-apis.md.

    NOTE: sklearn 1.9.0 removed the `squared=` kwarg from mean_squared_error(); RMSE must be
    computed via the dedicated root_mean_squared_error() function (added in 1.4, canonical
    since). `mean_squared_error(y_true, y_pred, squared=False)` raises TypeError on this
    install -- do not use that old pattern.
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# 6. Pipelines
# ---------------------------------------------------------------------------
TRAFFIC_ORDER = ["low", "medium", "high"]


def build_pipeline(
    model,
    numeric_cols: list[str],
    onehot_cols: list[str],
    ordinal_cols: list[str] | None = None,
    ordinal_categories: list[list[str]] | None = None,
    scale: str | None = None,
) -> Pipeline:
    """A ColumnTransformer + estimator Pipeline -- one fit()/predict() contract, like a
    builder that assembles preprocessing + model into a single typed unit.

    scale: None (trees -- passthrough numeric), "standard" (StandardScaler), or "minmax"
    (MinMaxScaler). One-hot for nominal categoricals, ordinal (explicit category order) for
    ordered ones. All transformer classes verified in research/NOTE-5-sklearn-core-apis.md.
    """
    if scale == "standard":
        numeric_transformer = StandardScaler()
    elif scale == "minmax":
        numeric_transformer = MinMaxScaler()
    else:
        numeric_transformer = "passthrough"

    transformers = [("numeric", numeric_transformer, numeric_cols)]
    if onehot_cols:
        # sparse_output=False: this dataset is small enough that a dense array is fine, and
        # HistGradientBoostingRegressor rejects sparse input outright.
        transformers.append(
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_cols)
        )
    if ordinal_cols:
        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1),
                ordinal_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def fit_and_score(pipeline: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame, target: str = "fare_amount"):
    feature_cols = [c for step in pipeline.named_steps["preprocess"].transformers for c in step[2]]
    pipeline.fit(train_df[feature_cols], train_df[target])
    preds = pipeline.predict(test_df[feature_cols])
    metrics = regression_metrics(test_df[target].to_numpy(), preds)
    return pipeline, preds, metrics


# ---------------------------------------------------------------------------
# 7. Main model comparison across the three feature stages
# ---------------------------------------------------------------------------
def run_stage_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit LinearRegression, RandomForest, HistGradientBoosting across three feature
    stages (raw coordinates -> +haversine distance -> +grid sector) on ONE shared holdout
    split, so every "lift" number is a fair like-for-like comparison.
    """
    base_numeric = ["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon", "duration_min", "passenger_count"]
    onehot_cols = ["payment_type"]
    ordinal_cols = ["traffic_level"]
    ordinal_categories = [TRAFFIC_ORDER]

    stages = {
        "raw_coords": {"numeric": base_numeric, "onehot": onehot_cols},
        "plus_distance": {"numeric": base_numeric + ["distance_km"], "onehot": onehot_cols},
        "plus_sector": {"numeric": base_numeric + ["distance_km"], "onehot": onehot_cols + ["pickup_sector"]},
    }

    model_specs = {
        "LinearRegression": (LinearRegression(), "standard"),
        "RandomForest": (RandomForestRegressor(n_estimators=200, max_depth=None, random_state=RNG_SEED, n_jobs=-1), None),
        # A lower learning_rate + more boosting rounds (max_iter) + shallower trees
        # (max_leaf_nodes) is the standard boosting recipe for squeezing out the sequential
        # residual-fitting advantage without overfitting -- see section 5's "default vs
        # tuned" demo for what happens with sklearn's out-of-the-box defaults instead.
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, random_state=RNG_SEED
            ),
            None,
        ),
    }

    rows = []
    fitted = {}
    for stage_name, cols in stages.items():
        for model_name, (model, scale) in model_specs.items():
            pipeline = build_pipeline(
                model=model,
                numeric_cols=cols["numeric"],
                onehot_cols=cols["onehot"],
                ordinal_cols=ordinal_cols,
                ordinal_categories=ordinal_categories,
                scale=scale,
            )
            fitted_pipeline, preds, metrics = fit_and_score(pipeline, train_df, test_df)
            rows.append({"stage": stage_name, "model": model_name, **metrics})
            fitted[(stage_name, model_name)] = (fitted_pipeline, preds)
            print(
                f"[{stage_name:>13}] {model_name:<21} "
                f"RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  R2={metrics['r2']:.4f}"
            )

    return pd.DataFrame(rows), fitted


# ---------------------------------------------------------------------------
# 8. Scaling demo: LinearRegression / KNN / RandomForest, unscaled vs scaled
# ---------------------------------------------------------------------------
def scaling_demo(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["distance_km", "duration_min", "passenger_count"]
    scalings = [None, "standard", "minmax"]
    scaling_labels = {None: "unscaled", "standard": "StandardScaler", "minmax": "MinMaxScaler"}

    model_specs = {
        "LinearRegression": lambda: LinearRegression(),
        "KNeighborsRegressor(k=15)": lambda: KNeighborsRegressor(n_neighbors=15),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=200, random_state=RNG_SEED, n_jobs=-1),
    }

    rows = []
    for model_name, make_model in model_specs.items():
        for scale in scalings:
            pipeline = build_pipeline(make_model(), numeric_cols=numeric_cols, onehot_cols=[], scale=scale)
            _, _, metrics = fit_and_score(pipeline, train_df, test_df)
            rows.append({"model": model_name, "scaling": scaling_labels[scale], **metrics})

    result = pd.DataFrame(rows)
    print("\n=== scaling demo: numeric-only features, no encoding ===")
    print(result.to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# 9. Encoding demo: ordinal (correct) vs ordinal-misused-on-nominal
# ---------------------------------------------------------------------------
def encoding_demo(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["distance_km", "duration_min", "passenger_count"]

    # Correct: payment_type (nominal, no order) one-hot; traffic_level (ordered) ordinal.
    correct_pipeline = build_pipeline(
        LinearRegression(),
        numeric_cols=numeric_cols,
        onehot_cols=["payment_type"],
        ordinal_cols=["traffic_level"],
        ordinal_categories=[TRAFFIC_ORDER],
        scale="standard",
    )
    _, _, correct_metrics = fit_and_score(correct_pipeline, train_df, test_df)

    # Misused: ordinal-encode payment_type too, in plain alphabetical order (card=0, cash=1,
    # mobile_wallet=2) -- there IS no real order between payment methods, so this invents one.
    misused_pipeline = build_pipeline(
        LinearRegression(),
        numeric_cols=numeric_cols,
        onehot_cols=[],
        ordinal_cols=["payment_type", "traffic_level"],
        ordinal_categories=[sorted(train_df["payment_type"].unique()), TRAFFIC_ORDER],
        scale="standard",
    )
    _, _, misused_metrics = fit_and_score(misused_pipeline, train_df, test_df)

    result = pd.DataFrame(
        [
            {"encoding": "one-hot(payment_type) + ordinal(traffic_level)  [correct]", **correct_metrics},
            {"encoding": "ordinal(payment_type, alphabetical) + ordinal(traffic_level)  [misused]", **misused_metrics},
        ]
    )
    print("\n=== encoding demo: one-hot vs ordinal on LinearRegression ===")
    print(result.to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# 10. Plots (artefacts)
# ---------------------------------------------------------------------------
def plot_residual_histogram(y_true: np.ndarray, y_pred: np.ndarray, model_label: str) -> Path:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residuals, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual fare - predicted fare, $)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual distribution -- {model_label} (holdout)")
    fig.tight_layout()
    out_path = ARTEFACTS_DIR / "residual_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_residual_vs_fitted(y_true: np.ndarray, y_pred: np.ndarray, model_label: str) -> Path:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_pred, residuals, s=10, alpha=0.4, color="#4C72B0")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted value (predicted fare, $)")
    ax.set_ylabel("Residual ($)")
    ax.set_title(f"Residual vs fitted -- {model_label} (holdout)")
    fig.tight_layout()
    out_path = ARTEFACTS_DIR / "residual_vs_fitted.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_parity(predictions_by_model: dict[str, np.ndarray], y_true: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, len(predictions_by_model), figsize=(5 * len(predictions_by_model), 4.5), sharey=True)
    lims = (0, max(y_true.max(), *(p.max() for p in predictions_by_model.values())) * 1.05)
    for ax, (model_name, preds) in zip(axes, predictions_by_model.items()):
        ax.scatter(y_true, preds, s=8, alpha=0.35, color="#4C72B0")
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="perfect prediction (45 deg)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual fare ($)")
        ax.set_title(model_name)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Predicted fare ($)")
    fig.suptitle("y vs y-hat parity -- holdout, plus_sector feature stage")
    fig.tight_layout()
    out_path = ARTEFACTS_DIR / "parity_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sector_scatter(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    sectors = sorted(df["pickup_sector"].unique())
    cmap = plt.get_cmap("tab20", len(sectors))
    sector_to_idx = {s: i for i, s in enumerate(sectors)}
    colors = df["pickup_sector"].map(sector_to_idx)
    scatter = ax.scatter(df["pickup_lon"], df["pickup_lat"], c=colors, cmap=cmap, s=6, alpha=0.6)

    zone_rect = plt.Rectangle(
        (ZONE_LON_MIN, ZONE_LAT_MIN),
        ZONE_LON_MAX - ZONE_LON_MIN,
        ZONE_LAT_MAX - ZONE_LAT_MIN,
        fill=False,
        edgecolor="red",
        linewidth=2,
        label="congestion zone (surcharge)",
    )
    ax.add_patch(zone_rect)

    lat_bins = np.linspace(NYC_LAT_MIN, NYC_LAT_MAX, GRID_N + 1)
    lon_bins = np.linspace(NYC_LON_MIN, NYC_LON_MAX, GRID_N + 1)
    for lb in lat_bins:
        ax.axhline(lb, color="gray", linewidth=0.4)
    for lb in lon_bins:
        ax.axvline(lb, color="gray", linewidth=0.4)

    ax.set_xlabel("Pickup longitude")
    ax.set_ylabel("Pickup latitude")
    ax.set_title(f"Pickup locations by grid sector ({GRID_N}x{GRID_N}), congestion zone outlined")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path = ARTEFACTS_DIR / "map_sector_scatter.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_feature_importance(rf_pipeline: Pipeline, hgb_pipeline: Pipeline, test_df: pd.DataFrame) -> Path:
    """RandomForest's native feature_importances_ next to HistGradientBoosting's
    permutation_importance -- HGB has no native feature_importances_ (verified against
    the installed sklearn==1.9.0, research/NOTE-5-sklearn-core-apis.md).
    """
    feature_names = rf_pipeline.named_steps["preprocess"].get_feature_names_out()
    rf_importances = rf_pipeline.named_steps["model"].feature_importances_

    hgb_feature_cols = [c for step in hgb_pipeline.named_steps["preprocess"].transformers for c in step[2]]
    perm = permutation_importance(
        hgb_pipeline, test_df[hgb_feature_cols], test_df["fare_amount"], n_repeats=10, random_state=RNG_SEED, n_jobs=-1
    )
    hgb_feature_names = hgb_pipeline.named_steps["preprocess"].get_feature_names_out()

    def top_n(names, values, n=8):
        order = np.argsort(values)[::-1][:n]
        return [names[i] for i in order], values[order]

    rf_names, rf_vals = top_n(feature_names, rf_importances)
    hgb_names, hgb_vals = top_n(hgb_feature_names, perm.importances_mean)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(rf_names[::-1], rf_vals[::-1], color="#4C72B0")
    axes[0].set_title("RandomForest: native feature_importances_", fontsize=11)
    axes[0].set_xlabel("Importance (mean impurity decrease)")

    axes[1].barh(hgb_names[::-1], hgb_vals[::-1], color="#DD8452")
    axes[1].set_title("HistGradientBoosting: permutation_importance", fontsize=11)
    axes[1].set_xlabel("Importance (mean RMSE increase when shuffled)")

    fig.suptitle("Feature importance -- top 8, plus_sector feature stage")
    fig.tight_layout()
    out_path = ARTEFACTS_DIR / "feature_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 11. Pitfall demos
# ---------------------------------------------------------------------------
def leakage_demo(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Location target-encoding leakage: encode pickup_sector as the mean fare per sector.

    Proper: fit the sector->mean-fare map on TRAIN ONLY, apply to both splits.
    Leaky: fit the map on TRAIN+TEST COMBINED (i.e. the target values of the very rows
    you're about to "predict" leak into their own feature), apply to both splits.
    The leaky version's holdout RMSE looks better than it has any right to.
    """
    full = pd.concat([train_df, test_df], axis=0)

    proper_map = train_df.groupby("pickup_sector")["fare_amount"].mean()
    leaky_map = full.groupby("pickup_sector")["fare_amount"].mean()

    global_mean = train_df["fare_amount"].mean()
    train_proper = train_df["pickup_sector"].map(proper_map).fillna(global_mean).to_numpy().reshape(-1, 1)
    test_proper = test_df["pickup_sector"].map(proper_map).fillna(global_mean).to_numpy().reshape(-1, 1)
    test_leaky = test_df["pickup_sector"].map(leaky_map).fillna(global_mean).to_numpy().reshape(-1, 1)

    model_proper = LinearRegression().fit(train_proper, train_df["fare_amount"])
    rmse_proper = root_mean_squared_error(test_df["fare_amount"], model_proper.predict(test_proper))

    model_leaky = LinearRegression().fit(train_proper, train_df["fare_amount"])
    rmse_leaky = root_mean_squared_error(test_df["fare_amount"], model_leaky.predict(test_leaky))

    print("\n=== pitfall demo: location target-encoding leakage ===")
    print(f"proper (encoding map fit on train only):  holdout RMSE = {rmse_proper:.3f}")
    print(f"leaky  (encoding map fit on train+test):   holdout RMSE = {rmse_leaky:.3f}")


def extrapolation_demo(fitted: dict) -> None:
    """Query each model on one absurd out-of-range trip (150 km -- well beyond the
    ~57 km max possible inside this dataset's NYC bounding box) using the plus_sector
    feature stage's fitted pipelines, and print what each model predicts.
    """
    sample = pd.DataFrame(
        {
            "pickup_lat": [NYC_LAT_MIN],
            "pickup_lon": [NYC_LON_MIN],
            "dropoff_lat": [NYC_LAT_MAX],
            "dropoff_lon": [NYC_LON_MAX],
            "duration_min": [240.0],
            "passenger_count": [1],
            "distance_km": [150.0],  # deliberately far beyond any training-set distance
            "payment_type": ["card"],
            "traffic_level": ["medium"],
            "pickup_sector": ["R0C0"],
        }
    )
    print("\n=== pitfall demo: extrapolation beyond the training coordinate range ===")
    print("query: a hypothetical 150 km trip (max in training data is ~57 km straight-line)")
    for model_name in ["LinearRegression", "RandomForest", "HistGradientBoosting"]:
        pipeline, _ = fitted[("plus_sector", model_name)]
        feature_cols = [c for step in pipeline.named_steps["preprocess"].transformers for c in step[2]]
        pred = pipeline.predict(sample[feature_cols])[0]
        print(f"  {model_name:<21} predicts ${pred:,.2f}")


def rmse_outlier_sensitivity_demo(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """RMSE punishes the single worst residual quadratically; MAE does not. The model's
    real holdout residuals are all fairly small (this model fits well), so to make the
    effect visible we simulate the realistic failure mode directly: ONE row's recorded
    fare_amount has a data-entry error (a decimal-point slip: $8.40 logged as $84.00) --
    everything else is untouched. Recompute both metrics with vs without that one bad label.
    """
    # Pick a row with a roughly typical fare (closest to the holdout median), not an
    # already-extreme one -- the point is that ANY ordinary row can wreck RMSE if its label
    # is wrong, not that we went looking for the biggest possible fare to corrupt.
    bad_idx = int(np.argmin(np.abs(y_true - np.median(y_true))))
    y_true_corrupted = y_true.copy()
    y_true_corrupted[bad_idx] = y_true_corrupted[bad_idx] * 10  # decimal-point data-entry slip

    clean_metrics = regression_metrics(y_true, y_pred)
    corrupted_metrics = regression_metrics(y_true_corrupted, y_pred)

    print("\n=== pitfall demo: RMSE's sensitivity to a single bad label ===")
    print(
        f"row {bad_idx}: true fare ${y_true[bad_idx]:.2f} mis-recorded as "
        f"${y_true_corrupted[bad_idx]:.2f} (a decimal-point slip) -- 1 row out of {len(y_true)}"
    )
    print(f"clean data:     RMSE={clean_metrics['rmse']:.3f}  MAE={clean_metrics['mae']:.3f}")
    print(f"1 bad label:    RMSE={corrupted_metrics['rmse']:.3f}  MAE={corrupted_metrics['mae']:.3f}")
    rmse_jump_pct = 100 * (corrupted_metrics["rmse"] - clean_metrics["rmse"]) / clean_metrics["rmse"]
    mae_jump_pct = 100 * (corrupted_metrics["mae"] - clean_metrics["mae"]) / clean_metrics["mae"]
    print(f"RMSE jumped {rmse_jump_pct:.1f}% from ONE bad label; MAE jumped only {mae_jump_pct:.1f}%")


def in_congestion_zone(df: pd.DataFrame) -> pd.Series:
    return (
        (df["pickup_lat"] >= ZONE_LAT_MIN)
        & (df["pickup_lat"] <= ZONE_LAT_MAX)
        & (df["pickup_lon"] >= ZONE_LON_MIN)
        & (df["pickup_lon"] <= ZONE_LON_MAX)
    )


def zone_lift_demo(test_df: pd.DataFrame, fitted: dict, y_true: np.ndarray) -> None:
    """The aggregate RMSE lift from adding pickup_sector (plus_distance -> plus_sector) is
    small -- only ~3% of trips start inside the congestion zone, so the fix to those rows'
    predictions is diluted across the other 97% when you look at whole-holdout RMSE. Zoom
    into just the in-zone rows and the lift is obvious: those trips carry a flat $2.75
    surcharge that plain lat/lon coordinates (or distance/duration alone) can't express,
    but a coarse grid-sector bucket can.
    """
    mask = in_congestion_zone(test_df).to_numpy()
    _, preds_no_sector = fitted[("plus_distance", "LinearRegression")]
    _, preds_with_sector = fitted[("plus_sector", "LinearRegression")]

    metrics_no_sector = regression_metrics(y_true[mask], preds_no_sector[mask])
    metrics_with_sector = regression_metrics(y_true[mask], preds_with_sector[mask])

    print(f"\n=== coordinate FE lift, zoomed into the congestion zone ({mask.sum()} of {len(mask)} holdout rows) ===")
    print(f"LinearRegression, no sector feature:   RMSE={metrics_no_sector['rmse']:.3f}  MAE={metrics_no_sector['mae']:.3f}")
    print(f"LinearRegression, + pickup_sector:      RMSE={metrics_with_sector['rmse']:.3f}  MAE={metrics_with_sector['mae']:.3f}")


def bagging_vs_boosting_overfitting_demo(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Boosting's sequential residual-fitting can win, but not automatically: sklearn's
    out-of-the-box HistGradientBoosting defaults (max_iter=100, learning_rate=0.1) are
    a much cruder search than 300 rounds at a gentler 0.05 with shallower leaves. This is
    boosting's "overfitting risk" from the spec made concrete: unlike RandomForest (which
    barely moves with n_estimators once it's past ~100 trees, since bagging only reduces
    variance), a boosting model's error surface is sensitive to learning_rate/max_iter/leaf
    size, and a naive default configuration is not guaranteed to beat a well-tuned bagging
    baseline -- let alone a well-specified linear model.
    """
    numeric_cols = ["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon", "duration_min", "passenger_count", "distance_km"]
    onehot_cols = ["payment_type", "pickup_sector"]
    ordinal_cols = ["traffic_level"]

    configs = {
        "HGB (sklearn defaults: max_iter=100, lr=0.1)": HistGradientBoostingRegressor(random_state=RNG_SEED),
        "HGB (tuned: max_iter=300, lr=0.05, max_leaf_nodes=15)": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, random_state=RNG_SEED
        ),
        "RandomForest (n_estimators=200)": RandomForestRegressor(n_estimators=200, random_state=RNG_SEED, n_jobs=-1),
    }

    print("\n=== bagging vs boosting: default HGB is not automatically the winner ===")
    for label, model in configs.items():
        pipeline = build_pipeline(
            model, numeric_cols=numeric_cols, onehot_cols=onehot_cols, ordinal_cols=ordinal_cols,
            ordinal_categories=[TRAFFIC_ORDER], scale=None,
        )
        _, _, metrics = fit_and_score(pipeline, train_df, test_df)
        print(f"  {label:<55} RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.4f}")


def randomized_search_mention(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """A light RandomizedSearchCV pass on HistGradientBoosting -- a small grid, few
    iterations, 3-fold CV. Not a deep tuning exercise (out of scope per SPEC-DS-5); just
    enough to show the API and that it's a drop-in wrapper around any estimator.
    Signature verified: research/NOTE-5-sklearn-core-apis.md /
    sklearn.model_selection.RandomizedSearchCV (installed sklearn==1.9.0).
    """
    numeric_cols = ["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon", "duration_min", "passenger_count", "distance_km"]
    pipeline = build_pipeline(
        HistGradientBoostingRegressor(random_state=RNG_SEED),
        numeric_cols=numeric_cols,
        onehot_cols=["payment_type", "pickup_sector"],
        ordinal_cols=["traffic_level"],
        ordinal_categories=[TRAFFIC_ORDER],
        scale=None,
    )
    param_distributions = {
        "model__max_iter": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 30],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=6,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=RNG_SEED,
        n_jobs=-1,
    )
    feature_cols = numeric_cols + ["payment_type", "pickup_sector", "traffic_level"]
    search.fit(train_df[feature_cols], train_df["fare_amount"])
    test_rmse = root_mean_squared_error(test_df["fare_amount"], search.predict(test_df[feature_cols]))

    print("\n=== RandomizedSearchCV mention (light: n_iter=6, cv=3) ===")
    print(f"best params: {search.best_params_}")
    print(f"best CV RMSE: {-search.best_score_:.3f}")
    print(f"holdout RMSE with best params: {test_rmse:.3f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    raw = synthesize_trips(n_rows=6000, seed=RNG_SEED)
    raw_path = DATASETS_DIR / "nyc_taxi_synthetic_raw.csv"
    raw.to_csv(raw_path, index=False)
    print(f"Wrote: {raw_path} ({len(raw)} rows, including dirty rows)")

    clean = clean_trips(raw)
    clean = add_pickup_sector(clean)

    train_df, test_df = train_test_split(clean, test_size=0.2, random_state=RNG_SEED)
    print(f"\ntrain rows: {len(train_df)}  test rows: {len(test_df)}")

    print("\n=== main comparison: 3 feature stages x 3 models, shared holdout ===")
    metrics_table, fitted = run_stage_comparison(train_df, test_df)
    metrics_path = ARTEFACTS_DIR / "metrics_comparison.csv"
    metrics_table.to_csv(metrics_path, index=False)
    print(f"\nWrote: {metrics_path}")

    y_true = test_df["fare_amount"].to_numpy()
    zone_lift_demo(test_df, fitted, y_true)

    bagging_vs_boosting_overfitting_demo(train_df, test_df)
    scaling_demo(train_df, test_df)
    encoding_demo(train_df, test_df)

    # Diagnostics on the plus_sector stage (the best-engineered feature set) for all 3 models.
    best_model = metrics_table.loc[metrics_table["stage"] == "plus_sector"].sort_values("rmse").iloc[0]
    best_model_name = best_model["model"]
    best_pipeline, best_preds = fitted[("plus_sector", best_model_name)]
    print(f"\nBest (stage=plus_sector) model by holdout RMSE: {best_model_name}")

    hist_path = plot_residual_histogram(y_true, best_preds, best_model_name)
    rvf_path = plot_residual_vs_fitted(y_true, best_preds, best_model_name)

    predictions_by_model = {name: fitted[("plus_sector", name)][1] for name in ["LinearRegression", "RandomForest", "HistGradientBoosting"]}
    parity_path = plot_parity(predictions_by_model, y_true)

    sector_scatter_path = plot_sector_scatter(clean)

    rf_pipeline, _ = fitted[("plus_sector", "RandomForest")]
    hgb_pipeline, _ = fitted[("plus_sector", "HistGradientBoosting")]
    importance_path = plot_feature_importance(rf_pipeline, hgb_pipeline, test_df)

    # Linear coefficients as importances (LO4) -- only meaningful because numeric features
    # were StandardScaled inside the pipeline.
    linear_pipeline, linear_preds = fitted[("plus_sector", "LinearRegression")]
    coef_names = linear_pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = linear_pipeline.named_steps["model"].coef_
    coef_table = pd.DataFrame({"feature": coef_names, "coefficient": coefs}).sort_values(
        "coefficient", key=np.abs, ascending=False
    )
    print("\n=== LinearRegression coefficients as importances (scaled features, plus_sector stage) ===")
    print(coef_table.head(10).to_string(index=False))

    # Pitfall demos.
    leakage_demo(train_df, test_df)
    extrapolation_demo(fitted)
    rmse_outlier_sensitivity_demo(y_true, best_preds)
    randomized_search_mention(train_df, test_df)

    print(f"\nWrote: {hist_path}")
    print(f"Wrote: {rvf_path}")
    print(f"Wrote: {parity_path}")
    print(f"Wrote: {sector_scatter_path}")
    print(f"Wrote: {importance_path}")


if __name__ == "__main__":
    main()
