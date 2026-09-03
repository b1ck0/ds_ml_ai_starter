# Hypothesis Testing & Exploratory Data Analysis

*Data Science · Worked Examples · SPEC-DS-1*

## The lady who could taste the milk

One afternoon in 1920s Cambridge, a colleague of the statistician Ronald Fisher poured herself a
cup of tea and refused to drink it: she said she could tell, just by taste, whether the milk had
gone into the cup before the tea or after. The men at the table were sure this was nonsense — tea
is tea. Fisher, being Fisher, didn't argue about it. He designed an experiment.

He gave her eight cups — four milk-first, four tea-first, in random order — and asked her to sort
them. If she was just guessing, she'd get some right and some wrong by luck alone, the same way a
flaky test sometimes passes even when the code underneath it is broken. The question Fisher
actually needed to answer wasn't "did she get most of them right" — it was **"if she genuinely
couldn't tell the difference, how surprising would this exact result be?"** That question, and the
machinery Fisher built to answer it, is the entire subject of this chapter. The lady — phycologist
Muriel Bristol — sorted all eight cups correctly
([source: Wikipedia, "Lady tasting tea"](https://en.wikipedia.org/wiki/Lady_tasting_tea), checked
2026-09-03), a result Fisher calculated has only a 1-in-70 chance of happening by pure guessing.

Walk through what Fisher actually did, step by step — it's the whole chapter in miniature:

**Step 1 — assume the boring explanation is true.** Start by assuming she can't actually tell the
difference. That assumption is the **null hypothesis**: under it, every cup she sorts correctly is
just luck.

**Step 2 — design a test that produces a number, not a vibe.** Eight cups, four-and-four, randomly
ordered. However many she gets right is a count you can compute odds for — "seems pretty good at
tea" isn't something you can reason about precisely.

**Step 3 — work out what luck alone would produce.** If she's purely guessing, there are
$\binom{8}{4}=70$ ways to pick which four cups she calls "milk-first," and exactly 1 of those 70
ways matches the truth on all eight cups. A perfect score by pure chance has probability
$1/70 \approx 1.4\%$.

**Step 4 — run the real experiment and compare.** Muriel Bristol sorted all eight cups correctly —
the single most extreme outcome possible, and one that only had a 1.4% chance of happening if she
were just guessing.

**Step 5 — decide what "surprising enough" means, then judge.** 1.4% is small enough that Fisher
rejected the null hypothesis — "no ability to tell the difference" — in favor of the alternative:
she really could taste it.

```mermaid
flowchart LR
    A["Step 1<br/>assume the boring explanation<br/>(null hypothesis)"] --> B["Step 2<br/>design a test that<br/>produces a number"]
    B --> C["Step 3<br/>work out what luck<br/>alone would produce"]
    C --> D["Step 4<br/>run the real experiment,<br/>compare to luck"]
    D --> E["Step 5<br/>decide: surprising enough<br/>to reject the null?"]
    E -.->|"this chapter runs the same loop<br/>on penguin flipper lengths"| A
```

That 1.4% is, in modern terms, a **p-value** — and "the lady tasting tea" is the origin story
statisticians still reach for when they explain what a p-value actually means. Fisher's 1935 book,
*The Design of Experiments*, is where the null hypothesis first appears as a formal, testable idea
([source: Wikipedia, "Lady tasting tea"](https://en.wikipedia.org/wiki/Lady_tasting_tea), checked
2026-09-03; book title and year confirmed via
[Wikipedia, "The Design of Experiments"](https://en.wikipedia.org/wiki/The_Design_of_Experiments),
checked 2026-09-03).

This chapter runs the exact same loop — assume the boring explanation, design a test, work out
what luck alone would produce, compare, decide — on a question you can't settle by eye: do two
penguin species really have different flipper lengths, or does it just look that way in a
smallish sample?

## 1. What & why

The tea-tasting loop maps directly onto vocabulary you already use every day. In Java, "this build
is flaky" is a hypothesis you test by running the suite many times and counting failures. Data
science formalizes that same move:

- **Null hypothesis (H0)** — the boring, default explanation: "there is no real difference; what
  I'm seeing is just sampling noise." Think of it as the *presumption of innocence* in a
  courtroom, or the default `assertTrue` that a flaky test is "probably just flaky, not
  actually broken."
- **Alternative hypothesis (H1)** — the claim you're actually interested in: "there *is* a real
  difference."
- **p-value** — given that H0 is true, the probability of seeing a difference at least this
  extreme by chance alone. A small p-value is evidence *against* H0 — it does **not** tell you
  the probability that H0 is true, and it does **not** tell you how *big* the difference is.
  That second gap is why this chapter also covers **effect size**.

A courtroom never *proves* innocence — it only says the evidence wasn't strong enough to convict.
Hypothesis testing works the same way: you never "prove" H0 true, you just fail to reject it.

```mermaid
flowchart TD
    START["collect evidence<br/>(the data)"] --> TEST{"compute a p-value:<br/>how surprising is this evidence<br/>if H0 were true?"}
    TEST -->|"p small (e.g. below 0.05)"| REJECT["reject H0<br/>('convict' -- evidence against<br/>'no real difference' is strong)"]
    TEST -->|"p not small"| FAIL["fail to reject H0<br/>('not guilty' -- NOT the same<br/>as 'innocent': just not enough<br/>evidence to convict)"]
```

And just like a flaky-test investigation, the answer depends entirely on how much evidence (how
much data) you collected — which is exactly the trap in the Pitfalls section below.

Here's where this chapter sits if you picture the data science process end to end — it lives in
the "make sense of the data before you build anything" stage, right after you've loaded it and
right before you'd start engineering features or training a model:

```mermaid
flowchart LR
    BU["Business<br/>Understanding"] --> DC["Data<br/>Collection"]
    DC --> CLEAN["Data<br/>Cleaning"]
    CLEAN --> EDA["EDA &amp; Hypothesis<br/>Testing<br/>&#9664; this chapter"]
    EDA --> FE["Feature<br/>Engineering"]
    FE --> MT["Model<br/>Training"]
    MT --> ME["Model<br/>Evaluation"]
```

## 2. The dataset: Palmer Penguins

This chapter uses the **Palmer Penguins** dataset — 344 penguins across three species (Adelie,
Chinstrap, Gentoo) measured on three islands in the Palmer Archipelago, Antarctica, by Dr. Kristen
Gorman for the Palmer Station Long Term Ecological Research Program (2007–2009). It's the modern,
actively-maintained successor to the 1936 Iris dataset for teaching EDA: it has real missing
values (Iris doesn't), a clean two-group and three-group comparison, and both numeric and
categorical columns.

- **License:** CC0 ("No Rights Reserved"), per the Palmer Station Data Policy.
- **Source / documentation:** https://allisonhorst.github.io/palmerpenguins/articles/intro.html
  (checked 2026-09-02)
- **Load method:** bundled with seaborn — `seaborn.load_dataset("penguins")` downloads from
  https://github.com/mwaskom/seaborn-data on first call and caches it locally (~36 KB); every
  later call is offline.

[source: NOTE-1-eda-dataset](../../research/NOTE-1-eda-dataset.md) (checked 2026-09-02)

### Environment

```text
pandas==3.0.5
numpy==2.5.2
matplotlib==3.11.1
scipy==1.18.1
seaborn==0.13.2
Python 3.12+
```

Pinned and verified against PyPI on 2026-09-02
([source: NOTE-2-package-versions](../../research/NOTE-2-package-versions.md)). This chapter's
code and artefacts were generated and gated on **Python 3.13.7**, and every package above
installed at exactly the pinned version with no substitutions — see the *Environment note* at the
end of this chapter for the one wrinkle NOTE-2 flagged (Python-version compatibility) that turned
out not to matter here.

## 3. EDA pass

Before testing anything, look at the data the way you'd read a stack trace before touching code —
shape first, then types, then what's missing, then the summary stats.

```python
import seaborn as sns

penguins = sns.load_dataset("penguins")

print(penguins.shape)
print(penguins.dtypes)
print(penguins.isna().sum())
print(penguins.describe())
```

Running this prints:

```text
=== shape ===
(344, 7)

=== dtypes ===
species                  str
island                   str
bill_length_mm       float64
bill_depth_mm        float64
flipper_length_mm    float64
body_mass_g          float64
sex                      str
dtype: object

=== missingness (NaN count per column) ===
species               0
island                0
bill_length_mm        2
bill_depth_mm         2
flipper_length_mm     2
body_mass_g           2
sex                  11
dtype: int64

=== describe() (numeric columns) ===
       bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
count      342.000000     342.000000         342.000000   342.000000
mean        43.921930      17.151170         200.915205  4201.754386
std          5.459584       1.974793          14.061714   801.954536
min         32.100000      13.100000         172.000000  2700.000000
25%         39.225000      15.600000         190.000000  3550.000000
50%         44.450000      17.300000         197.000000  4050.000000
75%         48.500000      18.700000         213.000000  4750.000000
max         59.600000      21.500000         231.000000  6300.000000
```

A few things worth noticing, the way you'd notice something off in a log dump:

- **`dtypes` shows `str`, not `object`.** In older pandas you'd see `object` for text columns —
  pandas 3.0 (pinned here per NOTE-2) infers a native string dtype by default. If you've read
  older pandas tutorials, expect `object` there instead; the behaviour is otherwise the same.
- **Missingness is small but nonzero (2–11 rows per column)** — that's `DataFrame.isna().sum()`,
  pandas' equivalent of grepping a log for `null`. This chapter treats missing rows by dropping
  them per-test (`dropna(subset=[...])`, only on the columns each test actually uses); a full
  chapter on smarter imputation strategies follows in DS-2.
- **The four numeric columns describe() covers are exactly `bill_length_mm`, `bill_depth_mm`,
  `flipper_length_mm`, `body_mass_g`.** `species`, `island`, and `sex` are categorical and don't
  show up in `describe()`'s default numeric summary — same reason `Stream<String>.summaryStatistics()`
  doesn't apply to a `Stream<Foo>`.

Two plots make the shape of the numeric data visible. First, the raw distribution of
`flipper_length_mm` across all 344 penguins:

```python
import matplotlib

matplotlib.use("Agg")  # headless: save figures, never call plt.show()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(penguins["flipper_length_mm"].dropna(), bins=20, color="#4C72B0", edgecolor="white")
ax.set_xlabel("Flipper length (mm)")
ax.set_ylabel("Count")
ax.set_title("Distribution of flipper length -- all species")
fig.tight_layout()
fig.savefig("flipper_length_histogram.png", dpi=150)
```

![Histogram of flipper length across all penguins](artefacts/flipper_length_histogram.png)

That histogram is clearly **bimodal** — two humps, not one bell curve. That's your first hint that
"all penguins" is hiding a mixture of different groups. A boxplot split by species confirms it:

```python
import seaborn as sns

fig, ax = plt.subplots(figsize=(7, 4.5))
order = ["Adelie", "Chinstrap", "Gentoo"]
sns.boxplot(data=penguins, x="species", y="flipper_length_mm", order=order, ax=ax)
ax.set_xlabel("Species")
ax.set_ylabel("Flipper length (mm)")
ax.set_title("Flipper length by species")
fig.tight_layout()
fig.savefig("flipper_length_boxplot_by_species.png", dpi=150)
```

![Boxplot of flipper length by species](artefacts/flipper_length_boxplot_by_species.png)

Gentoo penguins are obviously bigger-flippered than the other two — their box doesn't even overlap
the other species' boxes, so no test is needed to believe that gap is real. Adelie and Chinstrap
are a different story: their boxes overlap substantially. Stare at that overlap for a second and
ask yourself the honest question — **would you bet money those two species are really
different-flippered, or could that gap just be noise from measuring 151 particular Adelie birds
and 68 particular Chinstrap birds, out of every Adelie and Chinstrap that has ever existed?**
Eyeballing a boxplot can't settle that. That's exactly the ambiguity a hypothesis test is built to
resolve — and exactly the pair worth actually testing, rather than eyeballing.

## 4. A concrete question → a test

Which test you reach for depends entirely on the *shape* of the question — how many variables, and
whether they're numbers or categories:

```mermaid
flowchart TD
    Q{"what shape is the question?"}
    Q -->|"one numeric variable,<br/>two independent groups<br/>e.g. flipper length: Adelie vs Chinstrap"| TT["two-sample t-test<br/>scipy.stats.ttest_ind"]
    Q -->|"two categorical variables<br/>e.g. species vs island"| CS["chi-square test of independence<br/>scipy.stats.chi2_contingency"]
    Q -->|"one numeric variable,<br/>three or more groups"| OTHER["ANOVA -- generalizes<br/>the t-test to 3+ groups,<br/>not covered in this chapter"]
```

This chapter's two questions land on the first two branches.

### 4.1 "Does flipper length differ between Adelie and Chinstrap?" → t-test

The tempting shortcut is to just subtract the two group averages and call it a day. But a raw
"Adelie averages about 190mm, Chinstrap about 196mm" tells you nothing about whether that ~6mm gap
is a real species difference or just the particular 151-and-68 birds that got measured. You need a
number that answers the harder question: **"if the two species truly had identical flipper
lengths, how surprising would a gap this size be, given how much natural bird-to-bird variation
there already is inside each species?"**

That's exactly what a **two-sample t-test** computes. In plain language: it compares the size of
the gap between the two group means to the amount of "noise" inside each group, and expresses the
result as a single number, $t$ — a big $|t|$ means the gap is large relative to the noise; a small
$|t|$ means the gap is unremarkable next to how much groups naturally wobble.

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}}}$$

Reading every symbol in plain language: $\bar x_1, \bar x_2$ are the two group averages (Adelie's
mean flipper length, Chinstrap's mean flipper length); $s_1^2, s_2^2$ are each group's *own*
variance — how spread out that one species' measurements are (using each group's own variance,
instead of pooling them into one shared number, is exactly what makes this **Welch's** t-test
rather than the older Student's version — more on that below); $n_1, n_2$ are the two sample sizes.
The whole formula is "the gap between the means" divided by "how much wobble you'd expect the gap
to have, just from sampling" — the same shape as a z-score
([source: Wikipedia, "Welch's t-test"](https://en.wikipedia.org/wiki/Welch%27s_t-test), checked
2026-09-03).

This is a **two independent samples, one numeric variable** question — exactly what a two-sample
t-test answers. In `scipy.stats`, that's `ttest_ind`:

```python
from scipy.stats import ttest_ind

clean = penguins.dropna(subset=["flipper_length_mm", "species"])
adelie = clean.loc[clean["species"] == "Adelie", "flipper_length_mm"]
chinstrap = clean.loc[clean["species"] == "Chinstrap", "flipper_length_mm"]

result = ttest_ind(adelie, chinstrap, equal_var=False)  # Welch's t-test
print(result)
```

```text
TtestResult(statistic=-5.780384584564813, pvalue=6.049266635901903e-08, df=119.67695503084703)
```

`equal_var=False` matters: it switches from the *Student's* t-test (assumes both groups have the
same variance) to *Welch's* t-test (doesn't assume that). Verified against the scipy 1.18.1 docs:
`ttest_ind`'s default is `equal_var=True`, and you opt into Welch's explicitly
([source: NOTE-3-scipy-test-apis](../../research/NOTE-3-scipy-test-apis.md), checked 2026-09-02).
Since nothing here told us the two species have equal flipper-length variance, Welch's is the
safer default — the same instinct as not assuming two systems have identical error-rate
distributions just because they're both APIs.

`result` is a `TtestResult` namedtuple: `.statistic` is the t-statistic, `.pvalue` is what you
came for, `.df` is the degrees of freedom. Reading it: with 151 Adelie and 68 Chinstrap
penguins, mean flipper length is 189.95mm vs 195.82mm, and `p = 6.0e-08` — far below any
conventional threshold (0.05, 0.01).

```mermaid
flowchart LR
    P["p = 6.05e-08"] --> ASK{"what does this<br/>number actually mean?"}
    ASK -->|"correct reading"| RIGHT["IF the two species truly had<br/>identical flipper lengths, a gap<br/>this large would occur about<br/>6 times in 100 million samples"]
    ASK -->|"common misreading"| WRONG1["WRONG: 'there's a 6e-08<br/>chance H0 is true'"]
    ASK -->|"common misreading"| WRONG2["WRONG: 'the species differ<br/>by a huge amount'<br/>(that's effect size -- section 5)"]
```

**If there were truly no difference between the species, seeing a gap this large by chance alone
would happen about 6 times in 100 million samples.** That's strong evidence against H0 — Fisher's
same move from the tea cups, just with a t-statistic standing in for "how many cups sorted
correctly."

### 4.2 "Is species associated with island?" → chi-square test

A different shape of question, a different felt ambiguity: if species and island genuinely had
nothing to do with each other, would you expect a contingency table this lopsided — or does the
pattern below only look suspicious because you're staring at one particular sample of 342 birds?
This is a **two categorical variables** question — the right tool is a chi-square test of
independence on a contingency table (a cross-tab, same shape as a pivot table in a spreadsheet):

```python
import pandas as pd
from scipy.stats import chi2_contingency

clean = penguins.dropna(subset=["species", "island"])
contingency = pd.crosstab(clean["species"], clean["island"])
print(contingency)

chi2, p, dof, expected = chi2_contingency(contingency)
print(f"chi2={chi2:.3f}, p={p:.3e}, dof={dof}")
```

```text
island     Biscoe  Dream  Torgersen
species
Adelie         44     56         52
Chinstrap       0     68          0
Gentoo        124      0          0

chi2=299.550, p=1.355e-63, dof=4
```

`chi2_contingency` returns a `Chi2ContingencyResult` with `.statistic`, `.pvalue`, `.dof`, and
`.expected_freq` — the counts you'd expect under H0 ("species and island are independent")
([source: NOTE-3-scipy-test-apis](../../research/NOTE-3-scipy-test-apis.md)). Looking at the raw
table explains the astronomically small p-value before you even need the math: **Chinstrap only
appears on Dream, Gentoo only on Biscoe.** Species and island are almost perfectly entangled in
this dataset — likely because each species mostly breeds on one island. That's not a subtle
statistical effect; it's visible by eye, and the test just quantifies how implausible that pattern
would be if species and island were truly unrelated.

## 5. p-value & effect size — the "so what"

Both tests above rejected their null hypothesis emphatically. But a p-value only answers
*"is there probably a real difference?"* — it says nothing about *how big* that difference is in
terms you'd actually care about. That's what **effect size** measures, and — like a lot of the
most-used stats in applied work — **neither Cohen's d nor Cramér's V ships in `scipy.stats`**;
you compute both by hand
([source: NOTE-4-effect-sizes](../../research/NOTE-4-effect-sizes.md), checked 2026-09-02).

**Cohen's d** (for two-group numeric comparisons) is the mean difference, standardized by the
pooled standard deviation — "how many standard deviations apart are these two group means?":

$$d = \frac{\bar x_1 - \bar x_2}{s_{\text{pooled}}}, \qquad s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$$

In plain language: $\bar x_1 - \bar x_2$ is the same raw gap between means the t-test used; the
pooled standard deviation $s_{\text{pooled}}$ blends both groups' spread into a single "typical
wobble" ruler, and dividing by it converts "6mm" into "how many rulers wide is that gap" — a number
you can compare across completely different measurements
([source: NOTE-4-effect-sizes](../../research/NOTE-4-effect-sizes.md), checked 2026-09-02):

```python
import numpy as np


def cohens_d(x, y):
    """Cohen's d for two independent samples (pooled standard deviation)."""
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std


d = cohens_d(adelie, chinstrap)
print(f"Cohen's d = {d:.3f}")
```

```text
Cohen's d = -0.872
```

Conventional thresholds from Cohen (1988): **0.2 small, 0.5 medium, 0.8 large**
([source: NOTE-4-effect-sizes](../../research/NOTE-4-effect-sizes.md)). `|d| = 0.872` is a
**large** effect — this isn't just "detectable with enough data," the species really do have
substantially different flipper lengths.

**Cramér's V** (for two categorical variables) rescales the chi-square statistic into a 0–1
range so it's comparable across differently-sized tables:

$$V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1,\, c-1)}}$$

Plain language: $\chi^2$ is the chi-square statistic you already computed; $n$ is the total number
of penguins in the table; $r$ and $c$ are the number of rows and columns (species and islands);
$\min(r-1, c-1)$ is just a scaling factor so the result always lands between 0 and 1, no matter how
big the table is
([source: NOTE-4-effect-sizes](../../research/NOTE-4-effect-sizes.md), checked 2026-09-02):

```python
def cramers_v(chi2_stat, n, rows, cols):
    """Cramer's V for an r x c contingency table."""
    min_dim = min(rows - 1, cols - 1)
    return np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else np.nan


n = int(contingency.to_numpy().sum())
v = cramers_v(chi2, n, *contingency.shape)
print(f"Cramer's V = {v:.3f}")
```

```text
Cramer's V = 0.660
```

Thresholds: **<0.10 negligible, 0.10–0.30 weak, 0.30–0.50 moderate, >0.50 strong**
([source: NOTE-4-effect-sizes](../../research/NOTE-4-effect-sizes.md)). `V = 0.660` is a
**strong** association — consistent with what the contingency table already showed by eye.

Both results, plus the raw test statistics, are written to a small CSV artefact:

| test | statistic | p_value | effect_size_name | effect_size |
|---|---|---|---|---|
| Welch t-test: flipper_length_mm, Adelie vs Chinstrap | -5.780 | 6.05e-08 | Cohen's d | -0.872 |
| Chi-square: species vs island | 299.550 | 1.35e-63 | Cramer's V | 0.660 |

(full precision in [`artefacts/hypothesis_test_results.csv`](artefacts/hypothesis_test_results.csv))

The rule of thumb going forward: **report both numbers, always.** The p-value tells you whether to
trust that a difference is real; the effect size tells you whether that difference is *worth
caring about*. Section 6 shows exactly how far apart those two questions can drift.

## 6. Pitfalls

### 6.1 Huge N makes anything "significant"

This is the pitfall most Java engineers haven't internalized, because it doesn't have a direct
testing analogy: **with enough samples, a p-value can be tiny even when the real difference is
trivial.** p-values are sensitive to sample size in a way effect sizes are not.

```mermaid
flowchart LR
    N["increase sample size N"] --> P["p-value keeps shrinking --<br/>any nonzero gap eventually<br/>becomes 'significant'"]
    N -.->|"effect size does NOT<br/>depend on N"| D["Cohen's d stays flat --<br/>the true gap size<br/>never changed"]
```

Take `bill_depth_mm` between Adelie and Chinstrap — on the real sample sizes (151 vs 68), the test
finds *nothing* interesting:

```python
clean_bd = penguins.dropna(subset=["bill_depth_mm", "species"])
adelie_bill_depth = clean_bd.loc[clean_bd["species"] == "Adelie", "bill_depth_mm"].to_numpy()
chinstrap_bill_depth = clean_bd.loc[clean_bd["species"] == "Chinstrap", "bill_depth_mm"].to_numpy()

small_result = ttest_ind(adelie_bill_depth, chinstrap_bill_depth, equal_var=False)
print(f"t={small_result.statistic:.3f}, p={small_result.pvalue:.3f}, "
      f"d={cohens_d(adelie_bill_depth, chinstrap_bill_depth):.3f}")
```

```text
t=-0.438, p=0.662, d=-0.062
```

`p = 0.662` — not remotely significant, and `d = -0.062` says the effect is negligible even if it
were real. Now simulate what happens if we had 50,000 penguins of each species instead, by
resampling (with replacement) from the *same* real distributions:

```python
rng = np.random.default_rng(42)
adelie_big = rng.choice(adelie_bill_depth, size=50_000, replace=True)
chinstrap_big = rng.choice(chinstrap_bill_depth, size=50_000, replace=True)

big_result = ttest_ind(adelie_big, chinstrap_big, equal_var=False)
print(f"p={big_result.pvalue:.3e}, d={cohens_d(adelie_big, chinstrap_big):.3f}")
```

```text
p=4.424e-21, d=-0.060
```

The p-value collapsed from 0.662 to `4.4e-21` — you'd report that as "wildly significant." **But
Cohen's d barely moved (-0.062 → -0.060).** The underlying difference between the species never
changed; we just gathered enough evidence to detect a tiny, practically meaningless gap with
extreme confidence. This is exactly why Section 5's rule matters: a p-value alone, on a big
enough dataset, will eventually flag *any* nonzero difference as "significant." Report the effect
size, every time, or you will ship a model change chasing noise a stakeholder will call "real."

### 6.2 p-hacking

If you test enough hypotheses on the same dataset — different column pairs, different subgroups,
different thresholds — roughly 1 in 20 will look "significant" at `p < 0.05` by pure chance, even
if nothing is really going on. That's the same failure mode as re-running a flaky test until it
passes and calling the code correct. Decide your question *before* you look at the data, and if
you must run many comparisons, correct for it (Bonferroni is the simplest: divide your significance
threshold by the number of tests). A full treatment of multiple-testing correction is out of scope
here — this is the one-line warning to have in your head.

### 6.3 Assuming normality / equal variance without checking

The t-test's exact p-value depends on distributional assumptions. Section 4.1 used
`equal_var=False` (Welch's t-test) specifically to avoid *assuming* the two species have equal
variance — that assumption is often wrong in real data and Welch's correction is the safer
default, per the scipy 1.18.1 docs
([source: NOTE-3-scipy-test-apis](../../research/NOTE-3-scipy-test-apis.md)). For the chi-square
test, the standard caveat is that every cell's *expected* frequency should be at least 5 for the
test to be reliable — check `expected_freq` from `chi2_contingency`'s return value before trusting
a small or sparse table.

## 7. Recap & what's next

- A **null hypothesis** is the boring default ("no real difference") that a test tries to find
  evidence against — same shape as a courtroom's presumption of innocence, and the same move
  Fisher made when he assumed the lady couldn't tell tea-first from milk-first before he let the
  data argue otherwise.
- **`scipy.stats.ttest_ind`** compares means of two numeric groups; **`scipy.stats.chi2_contingency`**
  tests association between two categorical variables. Both return namedtuple-like results
  (`.statistic`, `.pvalue`, plus test-specific fields) — verified against scipy 1.18.1
  ([NOTE-3](../../research/NOTE-3-scipy-test-apis.md)).
- A small **p-value** is evidence the difference is real, not evidence it's *big*.
  **Effect size** (Cohen's d for means, Cramér's V for categorical association) tells you how big
  — and neither ships in scipy, so you compute them by hand
  ([NOTE-4](../../research/NOTE-4-effect-sizes.md)).
- On a big enough sample, almost any real-world difference becomes "significant." Always report
  effect size alongside the p-value.

This chapter dropped rows with missing values per-test — good enough for a clean teaching example,
but throwing away data is rarely the right production move. **DS-2 (imputation)** picks up exactly
here: smarter strategies for filling in `bill_depth_mm`, `sex`, and friends, and how each strategy
changes both your test results *and* your effect sizes. **DS-3 (collinearity)** then asks the
natural follow-up once you have several numeric columns: which of `bill_length_mm`, `bill_depth_mm`,
`flipper_length_mm`, and `body_mass_g` are actually telling you *different* things about a penguin,
and which are redundant.

---

### Environment note (for the architect)

NOTE-2 flagged that `numpy>=2.5.2` and `scipy>=1.18.1` require Python `>=3.12`, while `pandas` and
`matplotlib` support `3.11+`, and recommended pinning `numpy==1.26.x` / `scipy==1.17.x` if
targeting Python 3.11 specifically. This chapter's gate ran on **Python 3.13.7**, where all five
pinned versions from NOTE-2 (`pandas==3.0.5`, `numpy==2.5.2`, `matplotlib==3.11.1`,
`scipy==1.18.1`, `seaborn==0.13.2`) installed and ran with no substitution — so there is no
discrepancy to report for this environment. If a future chapter's gate runs on Python 3.11, revisit
NOTE-2's fallback recommendation for `numpy`/`scipy` before pinning `requirements.txt`.
