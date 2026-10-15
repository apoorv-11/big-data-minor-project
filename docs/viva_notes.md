# FraudShield — Viva Preparation & Technical Defense Guide

This guide contains everything you need to confidently explain, demonstrate, and defend FraudShield during your university viva examination.

---

## 1. Project Overview & Motivation

- **Project Title:** FraudShield: Fraud Detection and Transaction Analytics using Big Data
- **Domain:** Mobile money transfers and electronic payments fraud.
- **The Core Problem:**
  - Modern payment networks process millions of transactions per day.
  - Fraud is **extremely rare** (~0.13% in the PaySim dataset: only ~8,200 fraud cases out of 6.36 million transactions).
  - Traditional rule-based engines (e.g. flagging single transactions over $200,000) miss complex behavioral patterns and produce high false positives or false negatives.
  - Missing a fraud event directly costs financial loss and regulatory penalties, whereas a false alarm is only a minor friction. Therefore, **high recall** and **precision-recall trade-offs** are paramount.

---

## 2. Architecture & Design Decisions

### A. Medallion Architecture (Databricks Pattern)

```
[ Raw CSV: PaySim 6.36M ]
           │
           ▼
[ Bronze Layer (Parquet) ] ── Ingests raw data with ingestion metadata (_ingest_ts, _batch_id, _row_id)
           │
           ▼
[ Silver Layer (Parquet) ] ── Schema validation, deduplication, reject tracking (bad rows logged separately)
           │
           ▼
[ Silver Features ]       ── Rolling 24h window aggregations, balance error derivation, hour/day temporal features
           │
           ▼
[ Gold Layer (Parquet) ]   ── Analytics aggregations (KPIs, account risk, daily summaries) & ML-ready feature table
           │
     ┌─────┴────────────────────────┐
     ▼                              ▼
[ MLlib Training & Eval ]    [ Streamlit Interactive Dashboard ]
(LR, RF, GBT; PR-AUC >0.90)  (6 tabs: KPIs, Patterns, Risk, Model, Live Stream, Lineage)
```

#### Why Medallion Architecture?
1. **Separation of Concerns:** Bronze stores immutable history; Silver cleans and standardizes; Gold aggregates for downstream business and ML consumption.
2. **Data Governance & Auditability:** Raw data is never lost or modified. Faulty records are segregated to `silver_rejects` rather than silently dropped.
3. **Optimized Storage:** Parquet columnar format with Snappy compression reduces disk space from ~470MB raw CSV to optimized compressed columnar splits, speeding up analytic queries by 10x+.

---

## 3. Why PySpark for Big Data?

- **Distributed Computation:** Uses Spark's in-memory DAG engine with lazy evaluation, partitioning large datasets (6.36M rows) across multiple CPU cores.
- **Unified APIs:** PySpark provides unified DataFrames, Spark SQL, Spark MLlib for distributed machine learning, and Structured Streaming for Kafka ingestion.
- **Partitioning & Shuffling:** The dataset is partitioned into 20 partitions matching local worker cores. Catalyst Optimizer optimizes projection, predicate pushdown, and join operations.

---

## 4. Key Data Engineering Highlights

### Feature Engineering
1. **Balance Discrepancies (`errorBalanceOrig`, `errorBalanceDest`):**
   - In legitimate transactions: `newbalanceOrig = oldbalanceOrg - amount`.
   - Fraudsters often tamper with balances or initiate transfers when available balance is zero. Discrepancies between expected and actual balances are powerful fraud indicators.
2. **Temporal Features:**
   - Extracting `hour_of_day`, `day_of_month`, and `is_weekend` from simulation step (each step = 1 hour).
3. **Rolling Windows (`Window.partitionBy(...).rangeBetween(-24, 0)`):**
   - Computes transaction velocity: count and sum of transactions per sender and receiver over the previous 24 hours.

### Preventing Data Leakage
- **No Random Split:** We use a **chronological time-based split** (Steps 1–600 for training, Steps 601–743 for testing). A random train/test split leaks future information into past predictions.
- **Excluded Features:**
  - `isFlaggedFraud`: A naive heuristic column present in the dataset. Including it would cause target leakage.
  - IDs (`nameOrig`, `nameDest`): Kept as identifiers for audit lineage, but excluded from ML vector assembly to ensure the model learns behavioral dynamics rather than memorizing account strings.
  - Target labels or future aggregates are never included in feature pipelines.

---

## 5. Machine Learning & Imbalance Handling

### Why PR-AUC Instead of Accuracy or ROC-AUC?
- **Accuracy Fallacy:** If 99.87% of transactions are legitimate, a dummy model that predicts "always legitimate" achieves **99.87% accuracy** while catching 0% of fraud!
- **ROC-AUC Weakness:** ROC curves plot True Positive Rate vs. False Positive Rate. With millions of legitimate transactions, the False Positive Rate denominator is huge, making ROC-AUC look deceptively high (>0.99) even when hundreds of false alarms occur.
- **PR-AUC (Precision-Recall Area Under Curve):** Directly evaluates the trade-off between Precision (positive predictive value) and Recall (fraud detection rate) exclusively on the minority class.

### Class Balancing
- Handled via `classWeight` column in Spark MLlib (fraud weight ≈ 400x to compensate for 0.13% incidence), penalizing missed fraud proportional to its rarity.

### Models Compared
1. **Logistic Regression:** Linear baseline with class weighting and regularization.
2. **Random Forest:** Ensemble of bagging decision trees; handles non-linear relationships and feature correlations gracefully.
3. **GBT (Gradient Boosted Trees):** Sequential boosting optimizing pseudo-residuals; achieves the highest PR-AUC and discriminative power.

---

## 6. Real-Time Streaming & Orchestration

- **Kafka Ingestion:** Simulates near-real-time payment streams by replaying historical test transactions into a Kafka topic (`transactions`).
- **Structured Streaming:** Ingests micro-batches, deserializes JSON, transforms features, scores records against the trained PipelineModel, and flags high-risk alerts.
- **Airflow Orchestration:** Linear DAG with `BashOperator` scheduling each stage with automated retries, dependency enforcement, and logging.

---

## 7. Frequently Asked Viva Questions & Answers

### Q1: What is the difference between Bronze, Silver, and Gold in your pipeline?
> **Answer:** Bronze is raw Parquet preservation with audit metadata. Silver is cleaned, validated, typed data with reject segregation and engineered features. Gold contains business aggregations, KPIs, high-risk account lists, and ML-ready feature matrices.

### Q2: Why did you split train and test chronologically rather than with `randomSplit`?
> **Answer:** Financial transactions are time-series events. A random split allows transactions from step 700 to appear in the training set and step 100 in the test set. This creates future-lookahead data leakage. Time-based splitting evaluates the model on unseen future transactions.

### Q3: Why is accuracy a dangerous metric in fraud detection?
> **Answer:** Because of severe class imbalance (0.13% fraud). Predicting all transactions as non-fraud yields 99.87% accuracy but 0% recall, causing 100% financial loss. We rely on PR-AUC, Recall, and Confusion Matrix.

### Q4: How do you handle schema drift or corrupted rows?
> **Answer:** In `silver_clean.py`, we apply schema validation rules (e.g. non-negative amounts, valid transaction types). Records that violate validation are routed to `data/silver_rejects/` with reason codes, allowing monitoring without breaking pipeline execution.

### Q5: How would this scale to a real production environment?
> **Answer:** In production, Spark runs on a distributed cluster (e.g. AWS EMR, Databricks, or Kubernetes) rather than local mode. The Kafka producer would be replaced by transaction gateway webhooks, and the storage layer would reside in an object store like S3, ADLS, or Delta Lake with ACID guarantees.
