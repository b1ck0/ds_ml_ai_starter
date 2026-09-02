"""Drift detection: PSI, KS-test, and metric decay on synthetic drifting data.

Companion code for:
  Data Science/Production Considerations/monitoring-and-drift.md

What it does:
  1. Trains a "champion" LogisticRegression on a synthetic REFERENCE distribution --
     the training-time data a model would have shipped with. The ground-truth label
     rule is known exactly (this is simulation, not a real dataset), which is what
     lets this script measure drift's effect on accuracy instead of just asserting it.
  2. Simulates two 20-week PRODUCTION streams drawn from that same generator:
       - DATA DRIFT stream: the input distribution P(X1) walks away from the
         reference from week 10 onward (rising mean), while the true X->y
         relationship is held fixed the whole time.
       - CONCEPT DRIFT stream: P(X1) never moves -- every week looks like the
         reference on the inputs alone -- but the true X->y relationship rotates
         starting week 10 (feature 1's coefficient walks from +1.4 to -1.0), so the
         SAME inputs increasingly map to a DIFFERENT label.
  3. Computes, per week, per stream: PSI(X1) and a KS-test(X1) against the reference
     window (both are input-DISTRIBUTION tests -- no labels required), the
     champion's live accuracy (needs that week's true labels), and its predicted-
     positive rate (a label-free proxy for prediction/model drift -- since it is
     computed from predict(X) alone, it moves with data_drift (0.437 -> 0.977 by
     week 19, tracking PSI) but stays flat and noisy for concept_drift (0.46-0.57,
     no trend), because a fixed model's predictions depend only on X, never on y).
  4. Shows PSI/KS climbing hard for the data-drift stream (PSI ~5.3 by week 19,
     "significant drift") while staying flat for the concept-drift stream (PSI
     ~0.05, "stable") -- and the OPPOSITE pattern on accuracy: the data-drift
     stream's accuracy holds up fine (it even ticks up here, because pushing X1
     further from zero makes the coef1*X1 term dominate the noise, which makes the
     label easier, not harder, for a model that already has the right sign on
     coef1), while the concept-drift stream's accuracy collapses from ~0.91 to
     ~0.49 -- barely better than a coin flip -- with PSI never once crossing 0.1.
     That crossed pairing is the concrete point: a screaming PSI does not always
     mean the model is hurt, and a quiet PSI does not always mean it's safe --
     distribution tests and live-performance tests catch DIFFERENT failure modes,
     and neither alone is enough (research/NOTE-20-drift-detection.md's caveat:
     "Drift != model degradation").
  5. Saves four artefacts: distribution_shift.png, psi_ks_over_time.png,
     metric_decay.png, promotion_decision.png (a champion/challenger flowchart,
     drawn schematically -- not learned from data, like cv_fold_diagram.png in
     Data Science/Worked Examples/code/splitting_and_leakage.py).

Grounded:
  - PSI formula (Actual% - Expected%) * ln(Actual% / Expected%) and thresholds
    (<0.1 stable / 0.1-0.25 moderate / >=0.25 significant):
    research/NOTE-20-drift-detection.md.
  - scipy.stats.ks_2samp(data1, data2, alternative='two-sided', method='auto'):
    research/NOTE-20-drift-detection.md and research/NOTE-3-scipy-test-apis.md
    (scipy 1.18.1).
  - Data vs concept vs model/prediction drift definitions: research/NOTE-20-drift-
    detection.md.
  - sklearn.datasets.make_classification is NOT used here (the generator needs an
    explicit, inspectable true label rule so drift's effect on accuracy can be
    measured directly) -- LogisticRegression and accuracy_score signatures are
    per research/NOTE-5-sklearn-core-apis.md.

Environment (research/NOTE-2-package-versions.md, checked 2026-09-02):
    numpy==2.5.2, matplotlib==3.11.1, scipy==1.18.1, scikit-learn==1.9.0,
    Python 3.11+ (this script was run and gated against exactly those installed
    versions in this project's .venv).

Run:
    python drift_detection.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

RNG_SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

N_REFERENCE = 4000      # rows in the training-time / reference window
N_PER_WEEK = 300        # rows arriving per production week
N_WEEKS = 20             # weeks 0..19 of simulated production traffic
DRIFT_START_WEEK = 10    # both streams are drift-free before this week

# PSI thresholds, industry heuristics (research/NOTE-20-drift-detection.md):
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25


# ---------------------------------------------------------------------------
# 1. Data generation: a known ground-truth label rule, so drift's effect on
#    accuracy can be measured directly instead of asserted.
# ---------------------------------------------------------------------------

def true_label(x1: np.ndarray, x2: np.ndarray, coef1: float, rng: np.random.Generator) -> np.ndarray:
    """y = 1{coef1*X1 - 1.1*X2 + noise > 0}. coef1 is the "concept" -- change it
    and the SAME (X1, X2) maps to a different y."""
    logit = coef1 * x1 - 1.1 * x2
    noise = rng.normal(scale=0.5, size=x1.shape)
    return (logit + noise > 0).astype(int)


def make_reference(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The training-time distribution: X1, X2 ~ independent standard normals,
    labelled with coef1=1.4 (the relationship the champion model is trained on)."""
    x1 = rng.normal(loc=0.0, scale=1.0, size=N_REFERENCE)
    x2 = rng.normal(loc=0.0, scale=1.0, size=N_REFERENCE)
    y = true_label(x1, x2, coef1=1.4, rng=rng)
    X = np.column_stack([x1, x2])
    return X, y


def make_week(week: int, stream: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One week of production data for one of the two drift scenarios.

    stream='data_drift':    X1's mean ramps up from week DRIFT_START_WEEK onward
                             (P(X) shifts); the label rule stays coef1=1.4 always.
    stream='concept_drift': X1, X2 are drawn EXACTLY like the reference every week
                             (P(X) never moves); coef1 ramps from 1.4 down to -1.0
                             from week DRIFT_START_WEEK onward (X->y shifts).
    """
    weeks_into_drift = max(0, week - DRIFT_START_WEEK)
    if stream == "data_drift":
        mean_shift = weeks_into_drift * 0.30  # 0 .. 2.7 std devs by week 19
        x1 = rng.normal(loc=mean_shift, scale=1.0, size=N_PER_WEEK)
        x2 = rng.normal(loc=0.0, scale=1.0, size=N_PER_WEEK)
        coef1 = 1.4
    elif stream == "concept_drift":
        x1 = rng.normal(loc=0.0, scale=1.0, size=N_PER_WEEK)
        x2 = rng.normal(loc=0.0, scale=1.0, size=N_PER_WEEK)
        drift_fraction = weeks_into_drift / (N_WEEKS - 1 - DRIFT_START_WEEK)
        coef1 = 1.4 + drift_fraction * (-1.0 - 1.4)
    else:
        raise ValueError(f"unknown stream: {stream!r}")

    y = true_label(x1, x2, coef1=coef1, rng=rng)
    X = np.column_stack([x1, x2])
    return X, y


# ---------------------------------------------------------------------------
# 2. Drift detectors: PSI and KS-test on a single feature's distribution.
# ---------------------------------------------------------------------------

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10,
                 eps: float = 1e-4) -> float:
    """Population Stability Index between two 1-D samples.

    PSI = sum_i (actual_pct_i - expected_pct_i) * ln(actual_pct_i / expected_pct_i)
    (research/NOTE-20-drift-detection.md). Bin edges are the REFERENCE sample's
    deciles (n_bins equal-FREQUENCY bins over the reference window) -- a standard
    PSI binning choice, since it guarantees every reference bin starts with ~10% of
    the reference mass, so PSI==0 exactly when `current` matches `reference`. The
    outer edges are widened to +-inf so any `current` value outside the reference's
    observed range still lands in a bin instead of being silently dropped. `eps`
    floors every bin proportion so an empty bin in either sample never produces
    log(0) or a division by zero -- a standard PSI smoothing step, not part of the
    formula itself.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / ref_counts.sum(), eps, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_label(psi: float) -> str:
    """research/NOTE-20-drift-detection.md thresholds (industry heuristics, not
    universal cutoffs -- see the chapter's pitfalls section)."""
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate shift"
    return "significant drift"


def run_ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test; returns (statistic, pvalue).

    scipy.stats.ks_2samp(data1, data2, alternative='two-sided', method='auto') --
    signature for scipy 1.18.1 per research/NOTE-20-drift-detection.md and
    research/NOTE-3-scipy-test-apis.md. p < 0.05 rejects the null hypothesis that
    both samples come from the same distribution, i.e. drift is detected.
    """
    result = ks_2samp(reference, current, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# 3. Run both streams week by week, collecting PSI / KS / accuracy / pred-rate.
# ---------------------------------------------------------------------------

def run_stream(stream: str, champion: LogisticRegression, reference_x1: np.ndarray,
                rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (per-week metrics, week-19's raw X1 sample) -- the latter is only
    needed for the distribution-shift histogram, so it travels as a plain second
    return value rather than df.attrs (pandas' concat rejects DataFrames whose
    .attrs holds an ndarray, since attrs equality can't be truth-tested)."""
    rows = []
    last_x1 = None
    for week in range(N_WEEKS):
        X_week, y_week = make_week(week, stream, rng)
        x1_week = X_week[:, 0]

        psi = compute_psi(reference_x1, x1_week)
        ks_stat, ks_pvalue = run_ks_test(reference_x1, x1_week)

        y_pred = champion.predict(X_week)
        acc = accuracy_score(y_week, y_pred)
        pred_positive_rate = float(y_pred.mean())

        rows.append({
            "stream": stream, "week": week, "psi": psi, "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue, "accuracy": acc,
            "pred_positive_rate": pred_positive_rate,
        })
        last_x1 = x1_week
    return pd.DataFrame(rows), last_x1


# ---------------------------------------------------------------------------
# 4. Artefacts.
# ---------------------------------------------------------------------------

def plot_distribution_shift(reference_x1: np.ndarray, data_drift_late_x1: np.ndarray,
                             concept_drift_late_x1: np.ndarray) -> Path:
    """Reference X1 vs week-19 X1, for both scenarios side by side. Data drift is
    visible on sight; concept drift's inputs look identical to the reference --
    the point being that eyeballing (or PSI/KS-testing) the inputs alone cannot
    catch concept drift."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bins = np.linspace(-3, 6, 40)

    axes[0].hist(reference_x1, bins=bins, alpha=0.6, label="reference (training)",
                 color="#4C72B0", density=True)
    axes[0].hist(data_drift_late_x1, bins=bins, alpha=0.6, label="week 19 (data drift)",
                 color="#C44E52", density=True)
    axes[0].set_title("Data drift: P(X1) has visibly shifted")
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)

    axes[1].hist(reference_x1, bins=bins, alpha=0.6, label="reference (training)",
                 color="#4C72B0", density=True)
    axes[1].hist(concept_drift_late_x1, bins=bins, alpha=0.6,
                 label="week 19 (concept drift)", color="#55A868", density=True)
    axes[1].set_title("Concept drift: P(X1) is unchanged\n(the X->y rule moved instead)")
    axes[1].set_xlabel("X1")
    axes[1].legend(fontsize=8)

    fig.suptitle("Reference vs late-window input distribution, per scenario")
    fig.tight_layout()
    out_path = ARTEFACTS_DIR / "distribution_shift.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_psi_ks_over_time(results: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
    colors = {"data_drift": "#C44E52", "concept_drift": "#55A868"}
    labels = {"data_drift": "data-drift stream", "concept_drift": "concept-drift stream"}

    ax_psi = axes[0]
    for stream, color in colors.items():
        sub = results[results["stream"] == stream]
        ax_psi.plot(sub["week"], sub["psi"], marker="o", color=color, label=labels[stream])
    ax_psi.axhline(PSI_STABLE, color="grey", linestyle="--", linewidth=1,
                    label=f"PSI={PSI_STABLE:.2f} (stable / moderate)")
    ax_psi.axhline(PSI_SIGNIFICANT, color="black", linestyle="--", linewidth=1,
                    label=f"PSI={PSI_SIGNIFICANT:.2f} (moderate / significant)")
    ax_psi.axvline(DRIFT_START_WEEK, color="grey", linestyle=":", linewidth=1)
    ax_psi.set_ylabel("PSI(X1) vs reference")
    ax_psi.set_title("PSI over time")
    ax_psi.legend(fontsize=8, loc="upper left")

    ax_ks = axes[1]
    for stream, color in colors.items():
        sub = results[results["stream"] == stream]
        ax_ks.plot(sub["week"], sub["ks_pvalue"], marker="o", color=color, label=labels[stream])
    ax_ks.axhline(0.05, color="black", linestyle="--", linewidth=1, label="p=0.05")
    ax_ks.axvline(DRIFT_START_WEEK, color="grey", linestyle=":", linewidth=1,
                  label=f"drift starts (week {DRIFT_START_WEEK})")
    ax_ks.set_ylabel("KS-test p-value vs reference")
    ax_ks.set_xlabel("production week")
    ax_ks.set_title("KS-test p-value over time (p < 0.05 = drift detected)")
    ax_ks.legend(fontsize=8, loc="center left")

    fig.tight_layout()
    out_path = ARTEFACTS_DIR / "psi_ks_over_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_metric_decay(results: pd.DataFrame, reference_accuracy: float) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {"data_drift": "#C44E52", "concept_drift": "#55A868"}
    labels = {"data_drift": "data-drift stream", "concept_drift": "concept-drift stream"}

    for stream, color in colors.items():
        sub = results[results["stream"] == stream]
        ax.plot(sub["week"], sub["accuracy"], marker="o", color=color, label=labels[stream])
    ax.axhline(reference_accuracy, color="#4C72B0", linestyle="--", linewidth=1,
               label=f"reference-window accuracy ({reference_accuracy:.3f})")
    ax.axvline(DRIFT_START_WEEK, color="grey", linestyle=":", linewidth=1,
               label=f"drift starts (week {DRIFT_START_WEEK})")
    ax.set_xlabel("production week")
    ax.set_ylabel("champion model accuracy (that week's labels)")
    ax.set_title("Live accuracy over time: same champion model, two drift scenarios")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "metric_decay.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_promotion_decision() -> Path:
    """Schematic (not data-driven) champion/challenger promotion flowchart --
    same 'draw the process, don't fit it' approach as plot_cv_fold_diagram in
    Data Science/Worked Examples/code/splitting_and_leakage.py."""
    fig, ax = plt.subplots(figsize=(9, 7.5))

    def box(xy, w, h, text, facecolor="#4C72B0", fontsize=9, textcolor="white"):
        x, y = xy
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=facecolor, edgecolor="white", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                color=textcolor, wrap=True)
        return x, y, w, h

    def arrow(start, end, text=None, color="#333333"):
        patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=1.4)
        ax.add_patch(patch)
        if text:
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            ax.text(mx + 0.15, my, text, ha="left", va="center", fontsize=8, color=color)

    # Row 1: champion + challenger
    champ = box((0.3, 6.2), 2.6, 0.9, "Champion\n(current production model)",
                facecolor="#8C8C8C")
    chall = box((4.6, 6.2), 2.6, 0.9, "Challenger\n(retrained candidate)",
                facecolor="#DD8452")

    # Row 2: two evaluation sets, shared by both models
    recent = box((0.3, 4.2), 2.6, 0.9, "Recent labelled window\n(catches what's true now)",
                 facecolor="#4C72B0")
    golden = box((4.6, 4.2), 2.6, 0.9, "Frozen golden set\n(catches regressions on\nknown-important cases)",
                 facecolor="#4C72B0")

    arrow((1.6, 6.2), (1.6, 5.1))
    arrow((5.9, 6.2), (5.9, 5.1))
    arrow((2.9, 4.65), (4.6, 4.65))
    arrow((2.9, 6.65), (4.6, 6.65))

    # Row 3: decision diamond
    diamond_center = (3.75, 2.55)
    dx, dy = 2.05, 0.85
    diamond = Polygon([
        (diamond_center[0], diamond_center[1] + dy),
        (diamond_center[0] + dx, diamond_center[1]),
        (diamond_center[0], diamond_center[1] - dy),
        (diamond_center[0] - dx, diamond_center[1]),
    ], closed=True, facecolor="#55A868", edgecolor="white", linewidth=1.2)
    ax.add_patch(diamond)
    ax.text(*diamond_center, "Challenger >= champion\non BOTH sets?",
            ha="center", va="center", fontsize=8.5, color="white")

    arrow((1.6, 4.2), (3.3, 3.15))
    arrow((5.9, 4.2), (4.2, 3.15))

    # Row 4: outcomes
    promote = box((0.1, 0.5), 3.0, 0.9, "PROMOTE\nchallenger becomes champion",
                  facecolor="#55A868")
    keep = box((4.9, 0.5), 3.0, 0.9,
               "KEEP champion\ninvestigate / retrain differently",
               facecolor="#C44E52")

    arrow((3.0, 1.9), (1.6, 1.4), text="yes")
    arrow((4.5, 1.9), (6.4, 1.4), text="no")

    ax.set_xlim(-0.2, 7.5)
    ax.set_ylim(0.2, 7.3)
    ax.axis("off")
    ax.set_title("Champion/challenger promotion decision")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "promotion_decision.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5. Orchestration.
# ---------------------------------------------------------------------------

def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # Train the champion once, on the reference window only.
    X_ref, y_ref = make_reference(rng)
    champion = LogisticRegression(random_state=RNG_SEED)
    champion.fit(X_ref, y_ref)
    reference_accuracy = accuracy_score(y_ref, champion.predict(X_ref))
    print("=== champion trained on reference window ===")
    print(f"reference rows: {len(y_ref)}, positive rate: {y_ref.mean():.3f}")
    print(f"reference-window accuracy: {reference_accuracy:.4f}")

    reference_x1 = X_ref[:, 0]

    print("\n=== simulating 20 weeks of production traffic, two scenarios ===")
    data_drift_results, last_week_data_drift = run_stream("data_drift", champion, reference_x1, rng)
    concept_drift_results, last_week_concept_drift = run_stream(
        "concept_drift", champion, reference_x1, rng
    )
    results = pd.concat([data_drift_results, concept_drift_results], ignore_index=True)

    pd.set_option("display.width", 120)
    print("\n--- data_drift stream ---")
    print(data_drift_results[["week", "psi", "ks_statistic", "ks_pvalue", "accuracy",
                               "pred_positive_rate"]].round(4).to_string(index=False))
    print("\n--- concept_drift stream ---")
    print(concept_drift_results[["week", "psi", "ks_statistic", "ks_pvalue", "accuracy",
                                  "pred_positive_rate"]].round(4).to_string(index=False))

    final_data_drift = data_drift_results.iloc[-1]
    final_concept_drift = concept_drift_results.iloc[-1]
    print("\n=== week 19 summary ===")
    print(f"data_drift:    PSI={final_data_drift['psi']:.3f} ({psi_label(final_data_drift['psi'])}), "
          f"KS p={final_data_drift['ks_pvalue']:.2e}, accuracy={final_data_drift['accuracy']:.4f} "
          f"(reference {reference_accuracy:.4f})")
    print(f"concept_drift: PSI={final_concept_drift['psi']:.3f} ({psi_label(final_concept_drift['psi'])}), "
          f"KS p={final_concept_drift['ks_pvalue']:.2e}, accuracy={final_concept_drift['accuracy']:.4f} "
          f"(reference {reference_accuracy:.4f})")

    assert final_data_drift["psi"] >= PSI_SIGNIFICANT, "data-drift PSI should reach 'significant'"
    assert final_concept_drift["psi"] < PSI_STABLE, "concept-drift PSI should stay 'stable'"
    assert final_concept_drift["accuracy"] < final_data_drift["accuracy"], (
        "concept drift should hurt accuracy more than data drift here"
    )
    assert (reference_accuracy - final_concept_drift["accuracy"]) > 0.15, (
        "concept drift should visibly crater accuracy"
    )

    table_path = ARTEFACTS_DIR / "drift_metrics_by_week.csv"
    results.to_csv(table_path, index=False)

    dist_path = plot_distribution_shift(reference_x1, last_week_data_drift, last_week_concept_drift)
    psi_ks_path = plot_psi_ks_over_time(results)
    decay_path = plot_metric_decay(results, reference_accuracy)
    promotion_path = plot_promotion_decision()

    print(f"\nWrote: {table_path}")
    print(f"Wrote: {dist_path}")
    print(f"Wrote: {psi_ks_path}")
    print(f"Wrote: {decay_path}")
    print(f"Wrote: {promotion_path}")


if __name__ == "__main__":
    main()
