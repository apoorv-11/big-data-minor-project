# FraudShield 🛡️
### Fraud Detection and Transaction Analytics using Big Data & Spark MLlib

FraudShield is an end-to-end Big Data fraud detection and analytics platform built on the **PaySim** mobile-money dataset (6.36 million transactions). It leverages a scalable **Medallion Architecture (Bronze → Silver → Gold)** implemented in **PySpark**, engineers real-time and rolling behavioral features, trains and evaluates class-weighted machine learning models on **PR-AUC**, and serves analytics through an interactive **Streamlit dashboard** with automated **Airflow orchestration**.

---

## 🏛️ System Architecture

```
                                      ┌──────────────────────────────────────────────┐
                                      │             Raw PaySim CSV                   │
                                      │       6.36M transactions (~470MB)            │
                                      └──────────────────────┬───────────────────────┘
                                                             │
                                                             ▼
                                      ┌──────────────────────────────────────────────┐
                                      │           BRONZE LAYER (Parquet)             │
                                      │  Raw schema + Ingestion audit metadata       │
                                      │  (_ingest_ts, _source_file, _batch_id)       │
                                      └──────────────────────┬───────────────────────┘
                                                             │
                                                             ▼
                                      ┌──────────────────────────────────────────────┐
                                      │           SILVER LAYER (Parquet)             │
                                      │  Type casting, data hygiene, deduplication   │
                                      │  Bad records routed to silver_rejects        │
                                      └──────────────────────┬───────────────────────┘
                                                             │
                                                             ▼
                                      ┌──────────────────────────────────────────────┐
                                      │        SILVER FEATURE ENGINEERING            │
                                      │  Balance errors, 24h rolling velocity windows│
                                      │  Day, hour, weekend, destination aggregations│
                                      └──────────────────────┬───────────────────────┘
                                                             │
                                                             ▼
                                      ┌──────────────────────────────────────────────┐
                                      │            GOLD LAYER (Parquet)              │
                                      │  • gold_features (ML-ready feature matrix)   │
                                      │  • gold_kpis (Fraud rates by type/hour/day)  │
                                      │  • gold_account_risk (Top destination mules) │
                                      │  • gold_daily_summary (Daily trends & totals)│
                                      └──────────────┬───────────────────────────────┘
                                                     │
                             ┌───────────────────────┴───────────────────────────────┐
                             ▼                                                       ▼
               ┌───────────────────────────┐                           ┌───────────────────────────┐
               │    Spark MLlib Engine     │                           │   Streamlit Dashboard     │
               │  • Logistic Regression    │                           │  6 Tabs:                  │
               │  • Random Forest          │                           │  1. Overview KPIs         │
               │  • Gradient Boosted Trees │                           │  2. Fraud Patterns        │
               │  • Time split (1-600 / >) │                           │  3. Account Risk          │
               │  • Class weight 400x      │                           │  4. Model Performance     │
               │  • PR-AUC Evaluation      │                           │  5. Live Stream Monitor   │
               └─────────────┬─────────────┘                           │  6. Lineage & Quality     │
                             │                                         └───────────────────────────┘
                             ▼
               ┌───────────────────────────┐
               │   Batch & Stream Alerts   │
               │  Kafka replay + Streaming │
               └───────────────────────────┘
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.12+
- OpenJDK 17
- [uv](https://github.com/astral-sh/uv) (or standard Python virtualenv)
- Docker & Docker Compose (optional, for Kafka)

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/fraudshield.git
cd fraudshield

# Create virtual environment and install dependencies
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt setuptools
```

### 2. Run the Full Batch Pipeline
Run the end-to-end pipeline (Bronze → Silver → Features → Gold → ML Train → Batch Predict) with a single command:
```bash
# Direct Python execution (no make required):
python -m src.pipeline.run_all

# Or via Makefile (requires `sudo apt install make`):
make all
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run src/dashboard/app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

---

## 📊 Medallion Layers

| Layer | Directory | Description |
|-------|-----------|-------------|
| **Raw** | `data/raw/` | Source PaySim log file (`PS_..._log.csv`). |
| **Bronze** | `data/bronze/` | Snappy-compressed Parquet with ingestion audit timestamps. |
| **Silver** | `data/silver/` | Validated clean transactions with reject logging in `data/silver_rejects/`. |
| **Silver Features** | `data/silver_features/` | PySpark window aggregations, rolling 24h count & amounts, balance error features. |
| **Gold** | `data/gold/` | Aggregated KPI tables (`gold_kpis`, `gold_account_risk`, `gold_daily_summary`, `gold_features`). |
| **Predictions** | `data/predictions/` | Test set predictions with `prob_fraud` and `is_alert` threshold flags. |

---

## 🧠 Machine Learning & Imbalance Handling

### The Severe Imbalance Challenge
- In PaySim, fraudulent mobile transactions comprise only **~0.13%** (8,197 fraud cases out of 6.36 million).
- Accuracy is completely misleading: predicting 0 on everything achieves **99.87% accuracy** while missing 100% of fraud.
- We optimize and evaluate primarily on **PR-AUC (Precision-Recall Area Under the Curve)** and **Recall**, ensuring actual fraudsters are caught while managing analyst alert queues.

### Class Balancing
- Implemented using class weighting in Spark MLlib with a `fraud_weight` of ~400:
  $$\text{classWeight} = \begin{cases} 400.0, & \text{if } \text{isFraud} = 1 \\ 1.0, & \text{otherwise} \end{cases}$$

### Preventing Data Leakage
1. **Time-Based Split:** Training on Steps 1–600 (~6.26M rows) and testing on Steps 601–743 (~103k rows). Future steps are never leaked to the past.
2. **Feature Exclusions:** `isFlaggedFraud` and raw identifier strings (`nameOrig`, `nameDest`) are removed from model feature vectors.

---

## ⚡ Near-Real-Time Kafka Streaming

1. **Start Kafka Broker:**
   ```bash
   make up   # Launches Zookeeper & Kafka via docker compose
   ```
2. **Replay Transactions:**
   ```bash
   python -m src.ingestion.kafka_producer  # Or: make stream-produce
   ```
3. **Score Stream:**
   ```bash
   python -m src.streaming.stream_score    # Or: make stream-score
   ```

---

## 🗂️ Project Structure

```
minor-project/
├── airflow/
│   └── dags/
│       └── fraudshield_dag.py       # Airflow orchestration DAG
├── config/
│   └── config.yaml                  # Central configuration
├── data/                            # Bronze, Silver, Gold data artifacts
├── docs/
│   ├── problem_statement.md         # Formal project problem statement
│   └── viva_notes.md                # Viva defense & technical questions guide
├── src/
│   ├── common/                      # Spark session, logger, config loader
│   ├── ingestion/                   # Bronze batch reader & Kafka producer
│   ├── transform/                   # Silver cleaning, PySpark features, Gold build
│   ├── ml/                          # Model training, evaluation, batch predictions
│   ├── streaming/                   # Spark Structured Streaming scorer
│   ├── dashboard/                   # Streamlit + Plotly interactive UI
│   └── pipeline/                    # End-to-end pipeline runner
├── tests/
│   └── smoke_test.py                # Environment & Spark validation
├── Makefile                         # One-command CLI build targets
└── requirements.txt                 # Python dependencies
```

---

## 🎓 University Viva Defense Guide
For full viva preparation, architectural justifications, and detailed Q&A, see [docs/viva_notes.md](docs/viva_notes.md).
