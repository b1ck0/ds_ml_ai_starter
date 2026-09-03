# Curriculum — the backlog

The master plan. Each bullet below is a candidate **chapter** (or chapter section). The architect
(Opus) turns one bullet at a time into an approved `specs/SPEC-*.md`, then the pipeline writes it.
Status of each chapter is tracked by its spec, not here. This file is the *what to cover*; the specs
are the *contract for each piece*.

Spec id convention: `SPEC-<SUBJECT>-<n>-<slug>` where SUBJECT ∈ {DS, ML, AGENT, SDLC}.

Every subject folder is organised under the same five sections:
**Theory · Local Environment Setup · Worked Examples · Cloud Environment Setup · Production Considerations.**

---

## 1. Data Science (`01-data-science/`) — prefix `DS`

### Theory
- Hypothesis testing; imputation; feature engineering; regression; classification (binary,
  multi-class, multi-label); class imbalance; undersampling; ensembles; forecasting; AutoML; feature
  selection; overfitting; regularization.

### Local Environment Setup
- PyCharm, Python, `pip`/venv, pandas, NumPy, Matplotlib, scikit-learn, Jupyter Notebook.

### Worked Examples
- **Hypothesis testing & EDA** — a good textbook dataset for comparing distributions and doing EDA
  (Titanic as a candidate; researcher to confirm a better one if it exists). Distribution comparison,
  t-test / chi-square, p-values, effect size.
- **Imputation** — mean imputer; *why* impute at all; alternatives (median, KNN, model-based,
  indicator columns), and their failure modes.
- **Collinearity** — why it's dangerous; keeping feature count minimal; what an independent variable
  is vs a label; VIF / correlation heatmaps.
- **Train / Validation / Holdout split** — why we split; framed on simple (non-temporal) regression &
  classification where each row is independent and there is no future-into-past leakage.
- **Regression** — R², RMSE, MSE, MAE; Linear Regression, Random Forest, Gradient Boosting; the key
  difference between bagging (forest) and boosting and *why boosting often wins*. Dataset: **NYC taxi
  fare prediction** (also great for feature-engineering coordinates into 2D map sectors). Fairness of
  a model: residual distribution should be ~normal; y-vs-ŷ plot vs the 45° line. Feature importance
  via coefficients. Feature scaling: MinMax vs StandardScaler and what it affects. Which models favour
  numeric vs categorical features. One-hot encoding vs categorical/ordinal encoding — when each.
- **Classification** — Precision, Recall, Accuracy, F1, confusion matrix, AUC-ROC, AUC-PR; Logistic
  Regression, Random Forest, Gradient Boosting; class imbalance. Dataset: **Titanic** with feature
  engineering. Also identify good **multi-class** and **multi-label** datasets. Undersampling the
  majority class + training an **ensemble** (voting) to predict the minority class. Same scaling /
  transformation notes as regression.
- **Forecasting** — similarities with regression, but the key difference in **how we split**
  (non-overlapping windows, no leakage of the future). Autoregressive models, autocorrelation,
  seasonality, trend. Dataset: a **composite synthetic signal**; for each, say which forecasting model
  fits best and how to normalize/scale:
  - linear trend + sine wave (10% of amplitude A) on top
  - linear trend + random noise (10% of A)
  - sine wave + random noise (10% of A)
  - quadratic trend + random noise (10% of A)
- **Feature Selection (advanced)** — the problem it solves; knee/elbow method, forward selection,
  backward selection; recurring theme: a working model with the **minimum** number of features.
- **AutoML (advanced)** — introduce an open-source AutoML framework; what it does and how (researcher
  to pick a current, maintained one).
- **Model Registry** — MLflow: tracking model performance across runs.
- **Feature Store** — Feast (or another open-source store): slow vs fast features in production, a
  unified SDK.
- **Bayesian inference** (SPEC-DS-19) — the other paradigm: prior × likelihood → posterior, sampled
  with PyMC. Bayesian linear regression with Gaussian noise and an AR(1) time-series model on the
  owner's own use-cases; credible vs. confidence intervals; posterior-predictive bands; MCMC
  diagnostics (R-hat, ESS) with a deliberately non-converging model as the pitfall. Ties back to DS-5
  (regression) and DS-9 (the AR model returns).

### Cloud Environment Setup
- **Google Vertex AI** — notebooks, training via Vertex AI Pipelines, MLOps advantages over ad-hoc
  notebooks, where models deploy, the model registry.
- **Azure ML** — same shape as Vertex AI.
- **SageMaker** — same shape as Vertex AI.
- **Inference types** — batch (Airflow / Dataflow pipeline with the model embedded, or endpoint
  `batch_predict`) vs online (Docker container + REST API with the model inside).
- **Production monitoring** — concept drift, data drift, model drift and how to model for them; when
  to trigger retraining (MLOps pipeline as the big win); how to decide to swap the production model
  (compare new vs old on recent data + a golden dataset).
- **In-database deployment** — BigQuery ML, Amazon Redshift ML: no containers/endpoints, just SQL.

---

## 2. Machine Learning (`02-machine-learning/`) — prefix `ML`

### Theory
- Neural networks; gradient descent; neurons; activation functions; dense layers; dropout;
  convolution layers; LSTM/GRU; transformer; quantized models; model fine-tuning; encoder–decoder;
  autoencoder; tokenizers; word2vec; cosine similarity; Euclidean distance.

### Local Environment Setup
- Python, TensorFlow, PyTorch, torchvision.

### Worked Examples
- **Computer Vision**
  - Image classification — MNIST + torchvision.
  - Object detection — COCO + torchvision.
  - Semantic segmentation — COCO + torchvision.
  - Metrics — mAP, mAR, IoU, and other important ones.
- **Natural Language**
  - Text classification — pretrained DistilBERT checkpoint, inference only (SPEC-ML-8).
  - Text generation — distilgpt2, a decoder model (SPEC-ML-9).
  - Fine-tuning a transformer — train DistilBERT end to end on dair-ai/emotion (6-way emotion
    classification): explicit PyTorch loop + the HF `Trainer`, loss/accuracy curves, save/reload,
    inference (SPEC-ML-13).
  - Text & NLP metrics — classification, generation (BLEU/ROUGE/BERTScore), and retrieval/similarity
    metrics for text (SPEC-ML-14).
- **LLMs**
  - Transformer (from the inside).
  - Text generation.
- **Reinforcement Learning** (SPEC-ML-15) — MDP (state/action/reward/return); policy, value, and the
  Bellman equation; ε-greedy exploration; TD learning — Q-learning (off-policy) vs. SARSA (on-policy);
  a runnable tabular Q-learning agent on a tiny chess-derived environment (real `python-chess` legality,
  a King-vs-Rook corner-capture task), evaluated against random/greedy baselines; DQN, policy gradients,
  and self-play + MCTS (AlphaZero/MuZero) — explained and grounded, not executed, with an honest
  compute-gap caveat. Chess through-line, prerequisites SPEC-ML-1 and SPEC-DS-14.

### Cloud Environment Setup
- Google / AWS / Azure — blob storage, GPU training, TPU training.

---

## 3. Agentic Engineering (`03-agentic-engineering/`) — prefix `AGENT`

### Theory
- Vector databases; RAG; MCP; context window.

### Local Environment Setup
- Python, Google Agent Development Kit (ADK), pgvector, FastAPI, FastMCP.

### Worked Examples
- **MCP** — a simple MCP server as a database query layer: the agent asks for data, the MCP builds
  the query and returns it.
- **RAG** — a simple RAG app over PDFs.
- **Invoice Agent** — give it a PDF invoice; it extracts the fields and uses the MCP to write them to
  the database.
- **Elders Tribunal App** — a multi-agent app where different LLMs debate any topic you give and
  report the consensus back.

### Cloud Environment Setup
- GCP / AWS / Azure — the appropriate services for deploying such applications.

---

## 4. AI-assisted SDLC (`04-ai-assisted-sdlc/`) — prefix `SDLC`

### Theory
- Prompts; hooks; rules; gates; sub-agents; tools; skills.

### Local Environment Setup
- Java, Claude Code.

### Worked Examples
- Creating a new Java project + setting up all SDLC documents so the agents follow the SOP:
  researcher subagent, QA subagent, implementer subagent, architect. (This repo's own `.claude/`
  scaffold is a live reference example.)

---

## Notes for the architect
- Sequence within a subject roughly top-to-bottom; earlier chapters are prerequisites for later ones.
- Some plan items contain **claims to verify** (best dataset, current AutoML framework, right
  generative NLP model, exact metric formulas, current package versions). These MUST go through a
  Haiku research brief before the chapter is written — flag them in the chapter spec.
