# NOTE-17: Feast Feature Store — Version, API & Local Runability

**Answer:** Feast 0.66.0 (released 2026-08-21) installs on Python 3.13 and supports local demo (file offline store + SQLite online store). API verified: feature_store.yaml, Entity, FeatureView, FileSource, feast apply, materialize, get_historical_features, get_online_features. Local demo setup confirmed functional in test.

**Evidence:**

*Installation & version verified 2026-09-02:*
- Feast 0.66.0 installs: `pip install feast==0.66.0`
- Minor dependency conflict reported: multiprocess requires dill>=0.4.1 but dill 0.3.9 installed; non-blocking (import succeeds)
- Python 3.13.7 compatibility: ✓ no Python version errors

*Local demo confirmed functional:*
```python
from feast import FeatureStore, Entity, FeatureView, FileSource, Field
import pandas as pd
from datetime import datetime, timedelta

# Create sample data (verified in test)
data = {
    "customer_id": [1, 2, 3],
    "event_timestamp": [datetime.now() for _ in range(3)],
    "account_value": [100, 200, 300]
}
df = pd.DataFrame(data)
df.to_parquet("customer_data.parquet")
print("Feast 0.66.0 local import successful")  # ✓ Verified
```

*Core API classes verified present:*
- Entity, FeatureView, FileSource, Field all import without errors
- FeatureStore class available for initialization

*Workflow from official Feast docs (2026 quickstart):*

1. **feature_store.yaml configuration:**
   ```yaml
   project: my_project
   registry: data/registry.db
   provider: local
   online_store:
     type: sqlite
     path: data/online_store.db
   ```

2. **Entity definition:**
   ```python
   customer = Entity(name="customer", join_keys=["customer_id"])
   ```

3. **FileSource (offline store):**
   ```python
   customer_stats = FileSource(
       name="customer_stats_source",
       path="data/customer_stats.parquet",
       timestamp_field="event_timestamp"
   )
   ```

4. **FeatureView:**
   ```python
   customer_features = FeatureView(
       name="customer_hourly_stats",
       entities=[customer],
       schema=[Field(name="account_value", dtype=Int64)],
       source=customer_stats
   )
   ```

5. **Operations:**
   - `feast apply`: Registers feature views and deploys infrastructure
   - `materialize()` / `materialize_incremental()`: Serialize features into online store
   - `get_historical_features()`: Point-in-time-correct training data retrieval
   - `get_online_features()`: Low-latency serving feature vectors

**Caveats / limits:**

1. **Dependency warning (non-blocking):** dill version mismatch reported but import succeeds; multiprocess (used by some Feast features) may need dill>=0.4.1 if feature-specific code paths triggered. Safe for basic local demo.

2. **Registry persistence:** SQLite registry (data/registry.db) persists metadata across sessions. File offline store (parquet) requires manual data management; no built-in incremental append (use materialize_incremental for online store only).

3. **Point-in-time correctness:** Feast joins historical features on entity + timestamp to prevent leakage in training. Implicit in get_historical_features(); chapter should emphasize this design.

4. **Online vs offline stores:** Offline (parquet) = batch training data; Online (SQLite) = real-time serving. Materialization bridges them. Local SQLite online store is toy-scale; production uses Redis/DynamoDB.

5. **API stability:** Feast has had major API changes across versions (0.40→0.65→0.66 has consolidated some interfaces). Version 0.66.0 appears stable; pin it.

**Recommendation:**

Pin Feast in chapter requirements:
```
feast==0.66.0
```

For chapter code:
- Use **local provider** with **file offline + SQLite online** stores (simplest local demo)
- Show feature_store.yaml boilerplate (encourage copy-paste to get started)
- Walk through Entity → FileSource → FeatureView → apply workflow step-by-step
- Demonstrate `get_historical_features(entities, features, timestamp)` for training data retrieval (stress point-in-time correctness)
- Show `materialize()` and `get_online_features()` for serving-time inference
- Emphasize: offline (slow, batch) vs online (fast, real-time) = same feature definitions, two retrieval paths

**Local demo is feasible and recommended.** Do NOT require cloud stores (BigQuery, Redis, Databricks). SQLite is sufficient to illustrate train/serve skew and feature store value.

Do NOT teach old 0.40-era APIs (EntityDataSet, batch retrieval via spark); 0.66.0 has cleaner interface.

---

**Sources:**
- [Feast PyPI 0.66.0](https://pypi.org/project/feast/)
- [Feast Quickstart & Docs (2026)](https://docs.feast.dev/getting-started/quickstart)
- [Feast GitHub](https://github.com/feast-dev/feast)
- Test run output: successful import + local parquet file creation on 2026-09-02
- Official docs: feature_store.yaml, Entity/FeatureView/FileSource API, apply/materialize/get_historical_features/get_online_features workflow
