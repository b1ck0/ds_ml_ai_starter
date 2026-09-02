# SPEC-DS-6: Classification — who survived the Titanic

**Status:** approved
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-4, SPEC-DS-5 (metrics/scaling/encoding transfer)

## Intent
The flagship binary-classification chapter. Predict Titanic survival to teach the classification
metric zoo, three model families, the probability/threshold idea, and light feature engineering —
reusing the scaling/encoding lessons from regression.

## Learning objectives
- LO1 — Interpret Accuracy, Precision, Recall, F1, and the confusion matrix; know why accuracy lies on imbalanced data.
- LO2 — Read ROC-AUC and Precision-Recall AUC, and choose a decision threshold deliberately.
- LO3 — Train and compare Logistic Regression, Random Forest, Gradient Boosting; interpret coefficients/importances.
- LO4 — Do basic feature engineering (title from name, family size, fare bins) and one-hot vs ordinal encoding of categoricals.
- LO5 — Explain probability calibration at an intuition level and when it matters.

## Scope
In: the metric zoo, confusion matrix, ROC & PR curves, threshold choice, the three models, feature engineering, encoding/scaling recap.
Out: deep imbalance handling (→ SPEC-DS-8), multi-class/multi-label (→ SPEC-DS-7), SHAP (mention+link).

## Outline
1. What & why — classification vs regression; a yes/no decision with a probability behind it.
2. The data + feature engineering — Titanic; derive title/family-size; encode categoricals.
3. Metrics — confusion matrix, precision vs recall trade-off, F1; why accuracy misleads here.
4. Probabilities & thresholds — ROC-AUC, PR-AUC, moving the threshold; show both curves.
5. Three models compared on the holdout; importances/coefficients.
6. Pitfalls — accuracy on imbalance, leaking the target via engineered features, threshold=0.5 by default.

## Assets to produce
- Prose: "Data Science/Worked Examples/classification-titanic.md"
- Code: "Data Science/Worked Examples/code/classification_titanic.py"
- Artefacts: confusion matrix; ROC curve; PR curve; metric comparison table; importance bar chart.

## Claims to ground (Haiku, before writing)
- [ ] Confirm a reachable, licensed Titanic dataset + loader (seaborn.load_dataset('titanic') — verify columns/NaNs), or an alternative.
- [ ] Reuse NOTE-5 for sklearn model/metric/encoder APIs; additionally confirm roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix, classification_report.
- [ ] Verify the definitions of ROC-AUC vs PR-AUC against an authoritative source (when PR-AUC is preferred).

## Acceptance criteria
- [ ] AC1 — LO1–LO5 delivered. AC2 — code runs + all artefacts + snippet-check. AC3 — dataset + metric APIs + AUC definitions grounded. AC4 — accuracy-lies-on-imbalance shown empirically; Java framing for "a scored yes/no decision".

## Gates
Entry: approved; notes landed. Exit: DoD checklist.
