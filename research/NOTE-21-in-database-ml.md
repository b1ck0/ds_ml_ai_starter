# NOTE-21: BigQuery ML and Redshift ML SQL Syntax and Supported Model Types (2026)

**Answer:**
BigQuery ML: CREATE MODEL / CREATE OR REPLACE MODEL with OPTIONS(model_type=...) supporting 10+ internal types (linear/logistic regression, K-means, matrix factorization, PCA, ARIMA_PLUS, contribution analysis), plus external (DNN, XGBoost, Random Forest, AutoML) and imported (ONNX, TensorFlow, TensorFlow Lite). Training runs in BigQuery. Redshift ML: CREATE MODEL with TARGET and FUNCTION, supporting XGBOOST, MLP, LINEAR_LEARNER, KMEANS, FORECAST; problem types REGRESSION/BINARY_CLASSIFICATION/MULTICLASS_CLASSIFICATION; training runs in Amazon SageMaker.

**Evidence:**

## BigQuery ML

**Official Documentation:** https://docs.cloud.google.com/bigquery/docs/bqml-introduction (Google Cloud, 2026)

**CREATE MODEL Syntax:**
```sql
CREATE OR REPLACE MODEL `project.dataset.model_name`
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['target_column']
) AS
SELECT col1, col2, col3, target_column
FROM `project.dataset.table`
WHERE condition;
```

**Supported Model Types:**

### Internally Trained (built-in):
- Linear regression (`LINEAR_REG`) — numeric prediction
- Logistic regression (`LOGISTIC_REG`) — binary/multiclass classification (up to 50 classes)
- K-means clustering (`KMEANS`) — unsupervised segmentation
- Matrix factorization (`MATRIX_FACTORIZATION`) — recommendations
- Principal Component Analysis (`PCA`) — dimensionality reduction
- Time series forecasting (`ARIMA_PLUS`, `ARIMA_PLUS_XREG`) — univariate/multivariate forecasting
- Contribution analysis (`CONTRIBUTION_ANALYSIS`) — dimensional metrics analysis

### Externally Trained (via Agent Platform):
- Deep neural networks (DNN)
- Wide & Deep learning
- Autoencoder
- Boosted trees (XGBoost-based)
- Random forest
- AutoML (automated algorithm selection)

### Imported Models:
- ONNX format (Open Neural Network Exchange)
- TensorFlow (including TensorFlow Lite)
- XGBoost (custom-trained)

### Remote Models:
- Models deployed to Agent Platform endpoints (reference within BigQuery SQL)

**Additional SQL Commands:**
```sql
-- Evaluate model
CALL BQ.ML.EVALUATE(MODEL `project.dataset.model_name`, (
  SELECT col1, col2, col3, target_column
  FROM `project.dataset.test_table`
));

-- Make predictions
SELECT *
FROM ML.PREDICT(MODEL `project.dataset.model_name`, (
  SELECT col1, col2, col3
  FROM `project.dataset.new_data`
));
```

**Training Location:** BigQuery (no external infrastructure needed; training compute is within BigQuery's managed environment).

---

## Amazon Redshift ML

**Official Documentation:** https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_MODEL.html (AWS, 2026)

**CREATE MODEL Syntax:**
```sql
CREATE MODEL model_schema.model_name
FROM { table_name | (SELECT ...) }
TARGET column_name
FUNCTION function_name(data_type [, ...])
IAM_ROLE default
AUTO ON
PROBLEM_TYPE BINARY_CLASSIFICATION
SETTINGS (
  S3_BUCKET 'amzn-s3-demo-bucket'
);
```

**Supported Model Types (via MODEL_TYPE clause):**
- `XGBOOST` — gradient boosting (regression/classification)
- `MLP` — multilayer perceptron (neural network)
- `LINEAR_LEARNER` — linear/logistic regression
- `KMEANS` — clustering
- `FORECAST` — time series forecasting (requires HORIZON, FREQUENCY parameters)

**Supported Problem Types (via PROBLEM_TYPE clause):**
- `REGRESSION` — continuous numeric prediction
- `BINARY_CLASSIFICATION` — two-class classification
- `MULTICLASS_CLASSIFICATION` — multi-class classification (auto-detected if problem_type not specified and AUTO ON)

**Generated Prediction Function:**
After training completes, Redshift ML creates a prediction function in the same schema as the model:
```sql
SELECT model_schema.function_name(col1_value, col2_value, ...) AS prediction;
```

**Key Parameters:**
- `AUTO ON/OFF` — If ON, Redshift auto-selects algorithm, preprocessors, hyperparameters. If OFF, must specify MODEL_TYPE, PROBLEM_TYPE, PREPROCESSORS, and HYPERPARAMETERS.
- `OBJECTIVE` — Optimization metric (MSE for regression, F1 for classification, etc.). Default is task-dependent.
- `MAX_RUNTIME` — Maximum training time in seconds (default 5400 = 90 minutes).
- `MAX_CELLS` — Maximum training data size in cells (rows × columns; default 1,000,000).
- `S3_BUCKET` — Required; intermediate data and trained model stored here.

**Training Location:** Amazon SageMaker (data exported to S3, model training runs asynchronously in SageMaker; predictions invoked from Redshift).

**Important Note (2026):** Amazon Redshift will end support for Python UDFs after June 30, 2026. Native SQL models (CREATE MODEL) remain fully supported.

---

## Side-by-Side Comparison

| Feature | BigQuery ML | Redshift ML |
|---------|-------------|------------|
| **Training Location** | BigQuery (managed) | Amazon SageMaker (managed, async) |
| **Model Types** | 10+ built-in + external + imported | 5 native (XGBOOST, MLP, LINEAR_LEARNER, KMEANS, FORECAST) |
| **CREATE Syntax** | `CREATE [OR REPLACE] MODEL ... OPTIONS(model_type=...)` | `CREATE MODEL ... [MODEL_TYPE] ... PROBLEM_TYPE ... SETTINGS (S3_BUCKET ...)` |
| **Prediction API** | `ML.PREDICT(MODEL, SELECT ...)` | User-defined function (generated by Redshift ML) |
| **Evaluation** | `ML.EVALUATE()` / `ML.CONFUSION_MATRIX()` / etc. | Via SageMaker metrics or manual queries on predictions |
| **Supported Frameworks** | ONNX, TensorFlow, custom XGBoost | XGBoost, MLP, Linear Learner, KMeans, ARIMA (Forecast) |
| **Deployment** | Pure SQL, no infrastructure | Pure SQL, but relies on SageMaker backend |
| **Scalability** | BigQuery (petabyte-scale data) | Redshift (warehouse-scale, limited by cluster size) |

---

**Caveats / limits:**
- **BigQuery ML model types:** "Internally trained" models (linear/logistic, K-means, etc.) are simpler and run fully within BigQuery. External models (AutoML, DNN, XGBoost) require Agent Platform and may have higher latency.
- **Imported models (BigQuery):** Require the model to be pre-trained and stored in GCS (Google Cloud Storage) in a supported format (ONNX, TensorFlow saved_model, etc.).
- **Redshift ML async behavior:** CREATE MODEL returns immediately after exporting training data to S3; actual training happens asynchronously in SageMaker. Monitor training progress via Redshift system views.
- **Redshift ML cost:** Training incurs SageMaker costs; ensure S3 bucket and IAM role are correctly configured to avoid silent failures or runaway costs.
- **SQL-only deployment:** Both platforms require the data to live in the warehouse (BigQuery or Redshift). For data in external systems, federated queries or ETL pipelines are needed.
- **Feature engineering:** BigQuery ML and Redshift ML support basic transformations in SQL (UDFs, CASE statements), but advanced feature engineering (e.g., embedding generation) may require external preprocessing.
- **Limited hyperparameter tuning:** BigQuery's built-in models have limited hyperparameter exposure; Redshift ML's AUTO ON does implicit tuning, but explicit control is restricted.
- **Model versioning:** Neither platform has a native model registry equivalent to Vertex AI or SageMaker Model Registry. Version models manually (e.g., append timestamps to model names).

**Recommendation:**
- **BigQuery ML:** Use for rapid prototyping on large datasets; cite the CREATE MODEL and ML.PREDICT syntax from the official Google Cloud docs (dated 2026). Mark as **reference (not executed)** since BigQuery access is required.
- **Redshift ML:** Use when models are co-located with data in a Redshift warehouse; note the async training pattern and SageMaker backend. Cite the CREATE MODEL syntax from AWS docs (dated 2026).
- **SQL snippets:** Include complete, realistic examples for both platforms with comments explaining each clause (TARGET, FUNCTION, PROBLEM_TYPE, S3_BUCKET, etc.). No fabricated result rows; reference "expected output" only if from official docs.
- **Trade-offs section:** Emphasize that BigQuery ML is simpler (pure BigQuery, no external infra) but has fewer model type options; Redshift ML integrates with SageMaker (more models, more control) but adds cost and operational complexity.
- **Dates:** Both official documentation (Google Cloud and AWS) were verified 2026-08-15 to 2026-09-01; cite these sources directly.
