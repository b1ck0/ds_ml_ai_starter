"""MLflow Tracking + Model Registry on a synthetic subscription-churn dataset.

Companion code for:
  Data Science/Worked Examples/model-registry-mlflow.md

What it does:
  1. Builds a synthetic subscription-churn dataset (deterministic, seeded) with a fixed
     train/test split shared by every run, so runs are comparable apples-to-apples.
  2. Logs one run "by hand" -- mlflow.start_run, log_param, log_metric, log_artifact,
     mlflow.sklearn.log_model -- to show the primitives before any convenience wrapper.
  3. Logs four more runs with mlflow.sklearn.autolog() enabled, varying
     RandomForestClassifier hyperparameters, to show what autolog captures for free
     (params, training-set metrics, model, and classifier diagnostic plots) and why you
     still log your own held-out metric on top of it (autolog's metrics are computed on
     the TRAINING data passed to .fit(), not the held-out test set).
  4. Queries every run back out with mlflow.search_runs(), builds a runs-comparison
     table, and exports it as CSV + a bar chart (real artefacts, not screenshots).
  5. Registers the best run's model in the Model Registry and promotes it with a
     mutable ALIAS ("champion") -- not the deprecated Staging/Production STAGES
     (NOTE-16: MLflow 3.15.2 deprecates stages in favour of aliases).
  6. Trains one more (tuned) model, registers it as version 2, and re-points the
     "champion" alias at it -- the promotion / rollback workflow.
  7. Reloads the aliased model with mlflow.sklearn.load_model("models:/<name>@champion")
     and runs inference on the held-out test set, to show reload-for-inference and
     reproducibility (matches the accuracy logged at training time).
  8. Captures a text listing of the registry structure (registered model, versions,
     aliases, source runs) and the on-disk mlruns/ artifact tree -- the "captured
     listing" artefact called for by the chapter spec in place of a UI screenshot.

Environment (installed in the project's DEDICATED .venv-mlflow -- see the chapter's
"Local Environment Setup" note on why this chapter gets its own venv; NOTE-16 checked
2026-09-02, NOTE-5 checked 2026-09-02):
    mlflow==3.15.2, scikit-learn==1.9.0, pandas==2.3.3, numpy==2.5.2, matplotlib==3.11.1
    Python 3.12+ (this script was run and gated on Python 3.13.7 in .venv-mlflow).
    NOTE: mlflow==3.15.2 declares "pandas<3", which is why this chapter's pandas
    (2.3.3) differs from the 3.0.5 pinned in other Data Science chapters that don't
    depend on mlflow -- exactly the kind of transitive version conflict a dedicated
    virtual environment (an isolated classpath, in Java terms) exists to contain.

Run (from this directory, so the local sqlite store and mlruns/ artifact tree land in
a predictable place -- see the "Local Environment Setup" note on why the working
directory matters here):
    cd "Data Science/Worked Examples/code"
    ../../../.venv-mlflow/Scripts/python.exe mlflow_tracking.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

RNG_SEED = 42
HERE = Path(__file__).resolve().parent
ARTEFACTS_DIR = HERE.parent / "artefacts"
MLRUNS_DIR = HERE / "mlruns"          # local artifact store (gitignored)
TRACKING_DB = HERE / "mlflow.db"      # local sqlite tracking store (gitignored)

EXPERIMENT_NAME = "ds12-churn-model-registry-demo"
REGISTRY_MODEL_NAME = "churn_classifier"


def make_churn_data(n: int = 800, seed: int = RNG_SEED) -> pd.DataFrame:
    """Synthetic subscription-churn dataset: one row per customer.

    Five features drive churn through a logistic (sigmoid) relationship, plus
    Gaussian noise on the linear score -- the same "known ground truth, seeded"
    approach used throughout this course so a reader can trust the numbers are
    reproducible, not cherry-picked.
    """
    rng = np.random.default_rng(seed)
    tenure_months = rng.uniform(0, 60, n)
    monthly_spend = rng.uniform(10, 200, n)
    support_tickets = rng.poisson(1.5, n).astype(float)
    discount_pct = rng.uniform(0, 30, n)
    age = rng.integers(18, 75, n).astype(float)

    linear_score = (
        -0.055 * tenure_months
        + 0.35 * support_tickets
        - 0.03 * discount_pct
        + 0.01 * monthly_spend
        - 0.01 * age
        - 1.0
        + rng.normal(0, 0.6, n)
    )
    churn_probability = 1 / (1 + np.exp(-linear_score))
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend,
            "support_tickets": support_tickets,
            "discount_pct": discount_pct,
            "age": age,
            "churn": churn,
        }
    )


def eval_holdout(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float]:
    preds = model.predict(X_test)
    return float(accuracy_score(y_test, preds)), float(f1_score(y_test, preds))


def log_manual_run(X_train, y_train, X_test, y_test) -> str:
    """Run 1: explicit logging, no autolog -- the primitives.

    mlflow.start_run() opens a run (think: one CI build record); log_param /
    log_metric / log_artifact / log_model attach data to it, one call per fact.
    """
    params = {"model": "LogisticRegression", "solver": "lbfgs", "C": 1.0, "max_iter": 500}
    with mlflow.start_run(run_name="manual-logreg-baseline") as run:
        for key, value in params.items():
            mlflow.log_param(key, value)

        model = LogisticRegression(
            solver=params["solver"], C=params["C"], max_iter=params["max_iter"]
        )
        model.fit(X_train, y_train)

        test_accuracy, test_f1 = eval_holdout(model, X_test, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)

        # log_artifact: attach an arbitrary file to the run, not just a number.
        report_path = HERE / "_scratch_feature_notes.txt"
        report_path.write_text(
            "Features: tenure_months, monthly_spend, support_tickets, discount_pct, age\n"
            f"Train rows: {len(X_train)}  Test rows: {len(X_test)}\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(report_path), artifact_path="notes")
        report_path.unlink()

        mlflow.sklearn.log_model(model, name="model")
        print(f"[manual run] run_id={run.info.run_id} "
              f"test_accuracy={test_accuracy:.4f} test_f1={test_f1:.4f}")
        return run.info.run_id


def log_autolog_runs(X_train, y_train, X_test, y_test) -> list[str]:
    """Runs 2-5: mlflow.sklearn.autolog() -- one line replaces the manual log_param/
    log_metric/log_model calls above for every sklearn estimator trained afterwards.

    autolog's own metrics (training_accuracy_score, training_f1_score, ...) are
    computed on the TRAINING split passed to .fit() -- that's why this function still
    logs test_accuracy/test_f1 by hand: autolog convenience does not replace evaluating
    on held-out data.
    """
    mlflow.sklearn.autolog(log_models=True)
    configs = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 50, "max_depth": None},
        {"n_estimators": 150, "max_depth": 5},
        {"n_estimators": 300, "max_depth": 8},
    ]
    run_ids = []
    for i, cfg in enumerate(configs, start=1):
        with mlflow.start_run(run_name=f"autolog-rf-{i}") as run:
            model = RandomForestClassifier(random_state=RNG_SEED, **cfg)
            model.fit(X_train, y_train)
            test_accuracy, test_f1 = eval_holdout(model, X_test, y_test)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_f1", test_f1)
            print(f"[autolog run {i}] run_id={run.info.run_id} cfg={cfg} "
                  f"test_accuracy={test_accuracy:.4f} test_f1={test_f1:.4f}")
            run_ids.append(run.info.run_id)
    mlflow.sklearn.autolog(disable=True)
    return run_ids


def build_comparison_table(experiment_id: str) -> pd.DataFrame:
    """Pull every run back out with mlflow.search_runs() and shape it into a table
    a human (or a CI dashboard) would actually want to read.
    """
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time ASC"])
    cols = {
        "run_id": runs["run_id"],
        "run_name": runs["tags.mlflow.runName"],
        "test_accuracy": runs["metrics.test_accuracy"],
        "test_f1": runs["metrics.test_f1"],
        "n_estimators": runs.get("params.n_estimators"),
        "max_depth": runs.get("params.max_depth"),
        "solver": runs.get("params.solver"),
    }
    table = pd.DataFrame(cols).sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    return table


def plot_comparison(table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4c72b0" if name.startswith("autolog") else "#dd8452" for name in table["run_name"]]
    ax.bar(table["run_name"], table["test_accuracy"], color=colors)
    ax.set_ylabel("Held-out test accuracy")
    ax.set_title("Runs comparison: churn classifier (MLflow Tracking)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(table["test_accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "mlflow_runs_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def register_and_alias(run_id: str, alias: str) -> int:
    """Register the model logged under a run and point `alias` at the new version.

    Aliases (not the deprecated Staging/Production stages -- NOTE-16) are named,
    MUTABLE pointers to a specific model version: closer to a movable `latest` /
    `stable` git tag than to an immutable `v1.0.3` release tag. Re-running this
    function with a different run_id creates a NEW version and moves the alias --
    that's the promotion (and rollback) mechanism.
    """
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=REGISTRY_MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTRY_MODEL_NAME, alias=alias, version=model_version.version
    )
    print(f"[registry] {REGISTRY_MODEL_NAME} version {model_version.version} "
          f"<- run {run_id}; alias '{alias}' now points at version {model_version.version}")
    return int(model_version.version)


def capture_registry_listing(experiment_id: str) -> Path:
    """Text snapshot of the registry structure + the on-disk mlruns/ artifact tree --
    the "captured listing" artefact in place of a UI screenshot.
    """
    client = MlflowClient()
    lines: list[str] = []

    lines.append(f"Experiment: {EXPERIMENT_NAME} (id={experiment_id})")
    lines.append(f"Tracking URI: {mlflow.get_tracking_uri()}")
    lines.append("")
    lines.append(f"Registered model: {REGISTRY_MODEL_NAME}")
    rm = client.get_registered_model(REGISTRY_MODEL_NAME)
    for alias, version in sorted(rm.aliases.items()):
        lines.append(f"  alias '@{alias}' -> version {version}")
    lines.append("  versions:")
    for mv in client.search_model_versions(f"name='{REGISTRY_MODEL_NAME}'"):
        aliases_on_version = [a for a, v in rm.aliases.items() if v == mv.version]
        alias_note = f" (aliases: {', '.join('@' + a for a in aliases_on_version)})" if aliases_on_version else ""
        lines.append(f"    version={mv.version}  source_run_id={mv.run_id}  status={mv.status}{alias_note}")

    lines.append("")
    lines.append(f"On-disk artifact tree ({MLRUNS_DIR.relative_to(HERE)}/), directories only:")
    if MLRUNS_DIR.exists():
        for path in sorted(MLRUNS_DIR.rglob("*")):
            if path.is_dir():
                depth = len(path.relative_to(MLRUNS_DIR).parts)
                lines.append("  " + "  " * depth + path.name + "/")

    out_path = ARTEFACTS_DIR / "mlflow_registry_listing.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def capture_logged_model_requirements(run_id: str) -> Path | None:
    """Copy the requirements.txt that mlflow.sklearn.log_model auto-generated next to
    the model artifact -- proof that MLflow snapshots the exact library versions a
    model was trained with (the "environment drift" pitfall's antidote).
    """
    client = MlflowClient()
    local_dir = client.download_artifacts(run_id, "model")
    req_path = Path(local_dir) / "requirements.txt"
    if not req_path.exists():
        return None
    out_path = ARTEFACTS_DIR / "mlflow_logged_model_requirements.txt"
    out_path.write_text(req_path.read_text(encoding="utf-8"), encoding="utf-8")
    return out_path


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME, artifact_location=MLRUNS_DIR.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    df = make_churn_data()
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets", "discount_pct", "age"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df["churn"], test_size=0.25, random_state=RNG_SEED, stratify=df["churn"]
    )

    print("=== dataset ===")
    print(f"rows={len(df)}  churn_rate={df['churn'].mean():.3f}")

    manual_run_id = log_manual_run(X_train, y_train, X_test, y_test)
    autolog_run_ids = log_autolog_runs(X_train, y_train, X_test, y_test)

    table = build_comparison_table(experiment_id)
    print("\n=== runs comparison (sorted by held-out test_accuracy) ===")
    print(table.to_string(index=False))

    comparison_csv = ARTEFACTS_DIR / "mlflow_runs_comparison.csv"
    table.to_csv(comparison_csv, index=False)
    comparison_png = plot_comparison(table)

    best_run_id = table.iloc[0]["run_id"]
    best_run_name = table.iloc[0]["run_name"]
    print(f"\nBest run by held-out test_accuracy: {best_run_name} ({best_run_id})")

    v1 = register_and_alias(best_run_id, alias="champion")

    # One more, better-tuned iteration -- registered as v2, promoted over v1.
    with mlflow.start_run(run_name="autolog-rf-tuned") as run:
        mlflow.sklearn.autolog(log_models=True)
        tuned_model = RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=2, random_state=RNG_SEED
        )
        tuned_model.fit(X_train, y_train)
        tuned_accuracy, tuned_f1 = eval_holdout(tuned_model, X_test, y_test)
        mlflow.log_metric("test_accuracy", tuned_accuracy)
        mlflow.log_metric("test_f1", tuned_f1)
        mlflow.sklearn.autolog(disable=True)
        tuned_run_id = run.info.run_id
        print(f"\n[tuned run] run_id={tuned_run_id} test_accuracy={tuned_accuracy:.4f} "
              f"test_f1={tuned_f1:.4f}")

    v2 = register_and_alias(tuned_run_id, alias="champion")
    print(f"\n'champion' alias moved: version {v1} -> version {v2}")

    # Reload the CURRENT champion for inference -- this is what a serving process does.
    champion_model = mlflow.sklearn.load_model(f"models:/{REGISTRY_MODEL_NAME}@champion")
    reload_preds = champion_model.predict(X_test)
    reload_accuracy = float(accuracy_score(y_test, reload_preds))
    print(f"\n[reload for inference] models:/{REGISTRY_MODEL_NAME}@champion "
          f"test_accuracy={reload_accuracy:.4f} (should match tuned run: {tuned_accuracy:.4f})")
    assert abs(reload_accuracy - tuned_accuracy) < 1e-9, "reloaded model must reproduce training-time metric"

    listing_path = capture_registry_listing(experiment_id)
    requirements_path = capture_logged_model_requirements(tuned_run_id)

    print(f"\nWrote: {comparison_csv}")
    print(f"Wrote: {comparison_png}")
    print(f"Wrote: {listing_path}")
    if requirements_path:
        print(f"Wrote: {requirements_path}")


if __name__ == "__main__":
    main()
