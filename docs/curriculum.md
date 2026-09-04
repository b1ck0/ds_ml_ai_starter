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
- **Trustworthy probabilities on imbalanced data** (SPEC-DS-20) — the rare-event practitioner's
  toolkit for when a model must drive real decisions: **out-of-time validation** (split by time, not
  a random shuffle — the honest, lower number); the **Brier score** (+ Murphy decomposition and the
  base-rate-robust Brier skill score); **precision@top-N** and **lift** (the metric a capacity-limited
  team actually lives by); and **isotonic calibration fit on a true-prevalence hold-out** to fix the
  base-rate shift that undersampling introduces — with the King & Zeng prior-correction as the
  analytic intuition and a Platt-scaling contrast (and the honest note that isotonic's step function
  can tie ranks, where Platt preserves them exactly). Builds on DS-4/DS-6/DS-8; relates to DS-17.

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
- **How this repo was built** (SPEC-SDLC-3) — a self-referential case study: this book was produced
  by exactly the governed multi-agent pipeline it teaches (Opus scopes/reviews/merges · Sonnet writes
  one chapter · Haiku grounds · a fresh Sonnet reviews). Walks one real chapter from spec → NOTE →
  draft → gate → commit, maps every role/gate to a real file, and is honest about where it broke
  (the `guard.sh` false positive, the render-lint's origin, the rate-limit/concurrency lesson, a
  grounding conflict on a package version). Every claim is checkable with `git log`.
- **Governing an AI-built Rails e-store** (SPEC-SDLC-4) — the same scaffold on a different stack
  (Ruby on Rails 8), driving two security-sensitive features — user authentication and checkout —
  through the spec → implement → review → gate → merge loop. Wires security-first gates (RuboCop,
  RSpec, and Brakeman for a static security scan) and shows the fresh reviewer catching an
  authorization hole (an IDOR) that all three automated gates miss — because a static scanner can't
  see an authorization gap. A full runnable-in-Rails project tree (`code/rails-estore/`) with a
  stubbed payment seam, so it needs no external account and no live credentials. Extended with two
  more specialist agents — **seo-optimizer** (titles/canonical/Open Graph/Product JSON-LD/sitemap) and
  **frontend-qa** (axe/WCAG accessibility, valid/responsive HTML) — a product-catalog surface for them
  to work on, and a polished **standalone macOS README** so the example runs from zero on a Mac. Also
  ships a **verified `docker compose up`** local run (the project completed into a bootable Rails 8 app,
  actually built + booted + tested in Docker) with a Docker primer for newcomers and real screenshots
  of the running store.

---

## Notes for the architect
- Sequence within a subject roughly top-to-bottom; earlier chapters are prerequisites for later ones.
- Some plan items contain **claims to verify** (best dataset, current AutoML framework, right
  generative NLP model, exact metric formulas, current package versions). These MUST go through a
  Haiku research brief before the chapter is written — flag them in the chapter spec.
