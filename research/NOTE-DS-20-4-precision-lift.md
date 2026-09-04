# NOTE-DS-20-4: Precision@top-N (precision@k) and lift

**Answer:** Precision@top-N (a.k.a. precision@k) = (true positives in top N) / N, where items are ranked by predicted score descending. It measures the fraction of the capacity-limited action (top N reviews, calls, fraud investigations) that are correct. Lift = precision@N / base_rate, expressing precision relative to a random/base-rate baseline. A lift of 1.0 means random performance; >1.0 means the ranking concentrates true positives at the top.

**Evidence:**

### 1. Precision@top-N (Precision@k)

From https://docs.galileo.ai/concepts/metrics/rag/retrieval-quality/precision-at-k and https://www.shaped.ai/blog/precision-k-measuring-what-matters-at-the-top-of-your-rankings:

**Definition & formula:**
```
Precision@N = (# of true positives in top N ranked items) / N
```

**Ranking procedure:**
1. Score all samples with the classifier's predict_proba output (probability of positive class).
2. Sort descending by score.
3. Take the top N items.
4. Count how many of those N are actually positive (true positives).
5. Precision@N = TP / N.

**Range:** 0 to 1. Perfect score (all top N are positive) = 1.0. Random ranking (only baseline% are positive) = base_rate.

**When to use:** When a team has a fixed capacity (e.g., N fraud analysts can review N cases per week, or a call center can make N outbound calls). Precision@N directly answers: "Of the N highest-scoring cases I act on, how many will I get right?"

**Example:**
- Base rate (prior): 1% of all cases are fraudulent.
- Model trained and ranked: top 100 cases.
- 15 of those 100 are actually fraudulent.
- Precision@100 = 15/100 = 0.15 = 15%.

### 2. Lift

From https://en.wikipedia.org/wiki/Lift_(data_mining) and https://www.kdnuggets.com/2016/03/lift-analysis-data-scientist-secret-weapon.html:

**Definition & formula:**
```
Lift@N = Precision@N / base_rate
```

or equivalently,
```
Lift@N = (fraction positive in top N) / (overall fraction positive)
```

**Interpretation:**
- **Lift = 1.0:** Top N has same positive rate as the overall population; the model is not ranking (random).
- **Lift > 1.0:** Top N has *higher* positive rate than the population; the model is working, concentrating positives at the top.
- **Lift < 1.0:** Top N has *lower* positive rate than the population (bad; model is anti-ranking).

**Example (continued from above):**
- Precision@100 = 0.15 = 15%.
- Base rate = 0.01 = 1%.
- Lift@100 = 0.15 / 0.01 = 15.

This means: "By using the model to rank, I'm 15 times more likely to find fraud in my top 100 than if I picked randomly."

### 3. Why Precision@N Matters for Rare Events

From the spec's use case (fraud, credit default, churn):
- Rare events (1–2% base rate) make raw accuracy and AUC misleading; a model could have AUC=0.95 and still be useless if the business can only act on 100 cases.
- Precision@N + lift directly measure what the business gets: "Of my N actions, how many succeed?"
- On imbalanced data, a model with good AUC but poor ranking (e.g., logistic regression on undersampled 50/50 data) will have low precision@N at the true prevalence, exposing miscalibration.

**Date verified:** 2026-09-04

**Caveats / limits:**
- Precision@N is **not rank-aware** beyond the threshold: it treats all positions equally within the top N (position 1 and position N are equally weighted). If user behavior or cost varies by rank, use rank-aware metrics like reciprocal rank, DCG, or NDCG instead.
- Lift is a relative metric and depends on base rate; lift values are only comparable across datasets with similar prevalence. On highly imbalanced data, small changes in precision can yield large changes in lift.
- Precision@N requires choosing N upfront; if the action budget is unknown or variable, plot precision@k for a range of k values (precision-recall curve alternative).
- On synthetic balanced data (50/50 undersampled), precision@N will appear inflated if evaluated on the resampled distribution; must evaluate on true-prevalence out-of-time data to get honest metrics.

**Recommendation:**
- **For the chapter:** Show precision@N for a few fixed N values (e.g., top 50, 100, 200) to illustrate the action-budget framing.
- **Also show the curve:** Plot precision@k for k=1 to 500 to show how precision decays as N grows (and lift shrinks toward 1.0).
- **Always measure on true prevalence:** Evaluate OOT hold-out that reflects the true ~1–2% base rate, not the resampled training data.
- **Lift as a sanity check:** Ensure lift@k > 1.0 for top k; if lift approaches 1.0, the model is not ranking, even if AUC is high (sign of calibration failure under resampling).
