"""Train a tiny delivery-ETA regressor and persist it for the online service.

This mirrors what a Java engineer would call a "build step that produces an artifact" —
except the artifact isn't a .jar, it's a fitted scikit-learn estimator serialized with
joblib. `app.py` (the FastAPI service) loads exactly the file this script produces.

Run:  .venv-serving/Scripts/python train_model.py
Produces: model.joblib, model_metadata.json (both written next to this script)

Versions pinned per research/NOTE-19-inference-serving.md and research/NOTE-5-sklearn-core-apis.md:
scikit-learn==1.9.0, joblib (installed 1.6.0), numpy==2.5.2.
"""
from __future__ import annotations

import json
import pathlib
import sklearn
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- 1. A tiny synthetic "delivery ETA" dataset -----------------------------------------
# Feature order matters for both training and serving: it's the API contract between
# train_model.py and app.py, made explicit below (and re-checked at serve time — see
# app.py's startup validation, which is the versioning-skew guard this chapter's pitfalls
# section talks about).
FEATURE_ORDER = ["distance_km", "num_items", "is_peak_hour"]
MODEL_VERSION = "eta-regressor-v1"

rng = np.random.default_rng(seed=42)
n_samples = 2000

distance_km = rng.uniform(0.5, 15.0, size=n_samples)
num_items = rng.integers(1, 12, size=n_samples)
is_peak_hour = rng.integers(0, 2, size=n_samples)

noise = rng.normal(0, 2.5, size=n_samples)
eta_minutes = (
    8.0
    + 3.1 * distance_km
    + 1.4 * num_items
    + 6.0 * is_peak_hour
    + noise
)
eta_minutes = np.clip(eta_minutes, 5.0, None)  # a delivery never takes < 5 minutes

X = np.column_stack([distance_km, num_items, is_peak_hour])
y = eta_minutes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 2. Fit a pipeline: scale, then a small random forest -------------------------------
# A Pipeline bundles preprocessing + model behind one fit/predict contract (NOTE-5) — the
# same object that trains is the object that serves, so there's no separate "reimplement
# the scaling in the API layer" step to drift out of sync.
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=40, max_depth=5, random_state=42)),
    ]
)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"scikit-learn={sklearn.__version__}")
print(f"test RMSE: {rmse:.3f} minutes")

# --- 3. Persist with joblib, not pickle ---------------------------------------------------
# joblib is the scikit-learn-recommended path for persisting fitted estimators — it
# handles the numpy arrays inside a fitted model more efficiently than raw pickle, though
# the file it writes is still a pickle-based format under the hood (research/NOTE-19).
out_dir = pathlib.Path(__file__).parent
model_path = out_dir / "model.joblib"
joblib.dump(pipeline, model_path)

metadata = {
    "model_version": MODEL_VERSION,
    "feature_order": FEATURE_ORDER,
    "sklearn_version": sklearn.__version__,
    "test_rmse_minutes": round(float(rmse), 3),
}
metadata_path = out_dir / "model_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2))

print(f"wrote {model_path}")
print(f"wrote {metadata_path}")
print(json.dumps(metadata, indent=2))
