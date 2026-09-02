# NOTE-1: Best teaching dataset for EDA and distribution comparison

**Answer:** Use seaborn.load_dataset('penguins') — modern alternative to Iris with real-world missing values and clear two-group/categorical comparisons, ideal for teaching EDA to someone from a testing/QA background.

**Evidence:**
- **Source repository:** https://github.com/mwaskom/seaborn-data (Seaborn official sample datasets)
- **Original data:** Palmer Station Long Term Ecological Research Program, collected 2007-2009 by Dr. Kristen Gorman
- **License:** CC0 ("No Rights Reserved") in accordance with Palmer Station Data Policy and LTER Data Access Policy
- **Load method:** `seaborn.load_dataset('penguins')` — confirmed available in seaborn 0.13.2 (current stable)
- **Loading behavior:** Downloads from https://github.com/mwaskom/seaborn-data on first call, caches locally (~36 KB), requires internet on first run only (default cache_home=~/.seaborn/data)
- **Dataset link:** https://allisonhorst.github.io/palmerpenguins/articles/intro.html (official R package documentation, same data)
- **Date verified:** 2026-09-02

**Caveats / limits:**
- Seaborn 0.13.2 was released 2024-01-25; future versions may add/change datasets but penguin dataset is stable.
- Penguins dataset requires internet on first load (via `cache=True` default parameter).
- Iris, Titanic, and Tips are also available but Iris lacks missing values (unrealistic for teaching), Tips is smaller (244 rows), Titanic has moderate missingness but less educational structure.
- The Palmer Penguins paper (The R Journal 2022) explicitly positions this as a modern, representative alternative to Anderson's Iris for teaching, citing current ecological relevance.

**Recommendation:**
- Use `seaborn.load_dataset('penguins')` as the chapter's dataset. This gives students a real-world EDA challenge (missing values, multiple numeric columns, categorical groupings) that resembles actual data work.
- Code snippet: `import seaborn as sns; penguins = sns.load_dataset('penguins')` — requires pandas and seaborn installed, will cache dataset to disk after first download.
- Document that the data originates from Palmer Station LTER (CC0 license) to encourage reuse ethics in the reader.
- Example comparisons: numeric EDA on `bill_length_mm`, `bill_depth_mm` by `species` (3-level categorical); or `sex` vs `survived`-style inference in next chapter (DS-2).
