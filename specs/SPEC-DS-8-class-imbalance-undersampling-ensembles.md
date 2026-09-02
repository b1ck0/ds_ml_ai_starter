# SPEC-DS-8: Class imbalance — undersampling & ensembles for the minority class

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-6

## Intent
When the positive class is rare (fraud, defects, churn), a naive classifier scores 99% by predicting
"no" every time. Teach how to actually catch the minority: resampling, class weights, threshold
tuning, and an ensemble-of-undersampled-models approach.

## Learning objectives
- LO1 — Explain why imbalance breaks accuracy and why PR-AUC/recall are the metrics that matter here.
- LO2 — Apply class weighting (`class_weight`) and random undersampling / oversampling (SMOTE); know the trade-offs.
- LO3 — Build the "undersample the majority + train an ensemble (voting/bagging) over several balanced subsets" approach and show its lift over a single model.
- LO4 — Tune the decision threshold to a business cost, not a default 0.5.

## Scope
In: class weights, random under/oversampling, SMOTE, EasyEnsemble/BalancedBagging-style voting, threshold tuning, PR-AUC/recall focus.
Out: cost-sensitive learning theory depth, anomaly-detection framing (mention + link).

## Outline
1. What & why — the "always predict majority" trap, with the metric that exposes it.
2. Baseline on an imbalanced dataset — confusion matrix showing missed positives.
3. Remedies — class_weight; random undersample/oversample; SMOTE. Compare PR-AUC/recall.
4. Ensemble of undersampled learners — train k models each on a balanced subset, vote; show the gain.
5. Threshold tuning to a cost — pick the operating point from the PR curve.
6. Pitfalls — resampling BEFORE the split (leakage!), optimising accuracy, SMOTE on categoricals.

## Assets to produce
- Prose: "Data Science/Worked Examples/class-imbalance.md"
- Code: "Data Science/Worked Examples/code/class_imbalance.py"
- Artefacts: PR curves (baseline vs remedies); confusion matrices; a recall/PR-AUC comparison table.

## Claims to ground (Haiku, before writing)
- [ ] Verify the `imbalanced-learn` (imblearn) current version on PyPI + APIs: RandomUnderSampler, RandomOverSampler, SMOTE, BalancedBaggingClassifier / EasyEnsembleClassifier, and its Pipeline. Confirm it installs on the target Python and its sklearn-version compatibility.
- [ ] Pick an imbalanced dataset with a verified loader (sklearn `make_classification(weights=...)` for a runnable synthetic, or a real one like credit-card fraud if small/licensed). Recommend the runnable path.
- [ ] Reuse NOTE-5/NOTE-6-metrics for PR-AUC, recall, precision_recall_curve.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — code runs, the ensemble beats the baseline on recall/PR-AUC, artefacts produced, snippet-check passes. AC3 — imblearn version+APIs + dataset grounded. AC4 — resample-after-split discipline shown; the "99% accuracy is worthless here" point made concrete.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.
