# FraudShield: Big Data Fraud Detection & Transaction Analytics
## High-Impact 6-Slide Presentation & NotebookLM Source Guide

> **How to Use this Document with NotebookLM:**
> 1. Upload this file directly as a source into **Google NotebookLM** (`notebooklm.google.com`).
> 2. **Audio Overview (Deep Dive Podcast):** Click "Generate" on the Audio Overview to get an engaging, podcast-style discussion between two AI hosts explaining the project's architecture, trade-offs, and results.
> 3. **Slide Deck Generation:** Ask NotebookLM: *"Create a tight 6-slide executive presentation based on this document, highlighting the What, How, and Why of each slide."*
> 4. **Viva / Defense Prep:** Ask NotebookLM: *"Act as an external university examiner. Quiz me on the engineering decisions, class imbalance techniques, and data leakage safeguards in FraudShield."*

---

# Part 1: Executive Summary

- **Title:** FraudShield: Scalable Fraud Detection and Transaction Analytics using Big Data
- **Domain:** Mobile Money & Digital Payments (PaySim financial transaction simulation).
- **Scale:** 6.36 million transactions (~470 MB raw CSV) spanning 743 simulation hours (~30 days).
- **Core Dilemma:** Extreme class imbalance (only **0.129%** fraud; 8,213 fraud events out of 6,362,620 transactions, moving **$12.05 Billion** in stolen funds).
- **Solution:** An enterprise-grade, end-to-end **Medallion Lakehouse Pipeline** (Bronze → Silver → Gold) built in **PySpark**, engineered with distributed rolling window features, trained via **Spark MLlib** tree ensembles with 400x class weighting, automated with **Apache Airflow**, and served via an interactive **Streamlit** dashboard with **Kafka** streaming.
- **Primary Metric:** **PR-AUC (Precision-Recall Area Under Curve)** rather than deceptive Accuracy.

---

# Part 2: The 6-Slide Presentation Structure (What, How, Why)

---

### Slide 1: The Problem & The Accuracy Fallacy
- **Slide Title:** FraudShield: Detecting the 0.13% at Big Data Scale
- **Visual:** Split graphic: Left shows 99.87% legitimate traffic vs. 0.13% fraud ($12.05B at risk); Right shows "Naive 99.9% Model" ($0 caught) vs. "FraudShield" (100% recall).
- **The WHAT:**
  - In a 6.36 million transaction dataset, fraud occurs in only **8,213 transactions** (~0.129%).
  - Fraud is strictly concentrated in two types: `TRANSFER` (4,097 cases) and `CASH_OUT` (4,116 cases). It is zero in `PAYMENT`, `CASH_IN`, and `DEBIT`.
  - **The Accuracy Fallacy:** A trivial model predicting "always legitimate" achieves **99.87% accuracy** while missing 100% of fraud, causing catastrophic financial losses.
- **The HOW:**
  - We discard Accuracy and ROC-AUC as primary optimization metrics.
  - We optimize exclusively on **PR-AUC (Precision-Recall AUC)**, minority-class Recall, and business-focused confusion matrices.
- **The WHY:**
  - In fraud operations, missing a transaction (False Negative) costs money and compliance fines, whereas an alert (False Positive) is merely an investigator review. PR-AUC isolates performance exclusively on the rare positive class without being distorted by millions of easy True Negatives.
- **Speaker Script:**
  > *"Good morning. In mobile money, fraud is an adversarial needle in a 6.36-million-transaction haystack. If a bank’s detection system simply classified every transaction as legitimate, it would achieve 99.87% accuracy, yet allow twelve billion dollars to be stolen. That is why FraudShield rejects accuracy and is built entirely on Precision-Recall dynamics and big data engineering."*

---

### Slide 2: End-to-End Medallion Lakehouse Architecture
- **Slide Title:** The Medallion Lakehouse Architecture (Bronze → Silver → Gold)
- **Visual:** Horizontal architecture diagram showing: Raw CSV (470MB) → Bronze (Snappy Parquet + Audit metadata) → Silver (Cleaned + Reject Quarantine) → Gold (Analytic Marts & ML Matrix).
- **The WHAT:**
  - **Bronze Layer:** Ingests raw data into compressed Parquet, appending audit lineage: `_ingest_ts`, `_source_file`, `_batch_id`, and `_row_id`.
  - **Silver Layer:** Schema validation, type casting, and quarantine tracking for corrupt records in `silver_rejects`.
  - **Gold Layer:** Four pre-materialized business cubes: `gold_kpis`, `gold_account_risk` (top 500 money mules), `gold_daily_summary`, and `gold_features`.
- **The HOW:**
  - Built with **PySpark 3.5** running local parallelism across CPU cores.
  - Storage: Snappy-compressed Parquet partitioned into 16–20 splits matching available executor threads.
  - Rejection quarantine pattern:
    ```python
    clean_df = df.filter(is_valid)
    rejects_df = df.filter(~is_valid).withColumn("reject_reason", F.lit("Validation failure"))
    rejects_df.write.mode("overwrite").parquet("data/silver_rejects")
    ```
- **The WHY:**
  - Processing raw CSV directly into ML models is slow, brittle, and violates banking compliance. The Medallion pattern guarantees immutability in Bronze, compliance auditability in Silver (no silent `dropna`), and sub-second dashboard rendering in Gold.
- **Speaker Script:**
  > *"To handle 6.36 million rows efficiently, we implemented the Databricks Medallion Architecture. Bronze preserves raw data with audit metadata. Silver cleans and routes corrupt records into a dedicated quarantine table rather than silently dropping them. Gold pre-calculates analytical cubes so our dashboard and ML models never perform redundant, full-table scans."*

---

### Slide 3: Advanced Feature Engineering — Balance Invariance & Window Velocity
- **Slide Title:** PySpark Feature Engineering: Accounting Physics & Velocity Windows
- **Visual:** Mathematical formulas for balance discrepancies alongside a diagram of a sliding 24-hour time window aggregating sender/receiver activity.
- **The WHAT:**
  1. **Accounting Invariance (Balance Discrepancies):**
     - Origin Error: $\text{errorBalanceOrig} = \text{newbalanceOrig} + \text{amount} - \text{oldbalanceOrg}$
     - Destination Error: $\text{errorBalanceDest} = \text{oldbalanceDest} + \text{amount} - \text{newbalanceDest}$
  2. **Behavioral Velocity Profiles (Rolling 24-Hour Windows):**
     - Sender velocity: transaction count and total volume over the past 24 hours (`txn_count_orig_last_24`, `txn_amount_orig_last_24`).
     - Recipient fan-in: distinct senders and cumulative inbound funds (`dest_distinct_senders`, `dest_total_inbound`).
- **The HOW:**
  - Implemented in PySpark using distributed window definitions partitioned by account:
    ```python
    w_24 = Window.partitionBy("nameOrig").orderBy("step").rangeBetween(-24, 0)
    df = df.withColumn("txn_count_orig_last_24", F.count("*").over(w_24) - 1)
    df = df.withColumn("errorBalanceOrig", F.round(F.col("newbalanceOrig") + F.col("amount") - F.col("oldbalanceOrg"), 2))
    ```
- **The WHY:**
  - Static features miss the fraud lifecycle. Fraudsters test stolen accounts with micro-transfers before draining them, or funnel multiple stolen balances into a single money mule account. Window functions expose transaction velocity, while balance discrepancies catch forced balance wipes.
- **Speaker Script:**
  > *"Raw transaction amounts alone cannot catch sophisticated attackers. We engineered two distinct feature categories: First, accounting invariance—in legitimate transfers, balance changes match the transaction amount, whereas fraud accounts exhibit severe discrepancies. Second, rolling 24-hour window functions that capture burst velocities and money-mule aggregation networks."*

---

### Slide 4: ML Rigor — Time-Split Leakage Prevention & Class Weighting
- **Slide Title:** Machine Learning Rigor: Preventing Leakage & Cost-Sensitive Learning
- **Visual:** Timeline split diagram (Steps 1–600 Train vs. Steps 601–743 Test) contrasted with an incorrect random split. Class weighting scale graphic showing 400x penalty for fraud.
- **The WHAT:**
  - **Time-Based Splitting:** Chronological split at step 600 (~Day 25):
    - Training Set: Steps 1–600 = **6,259,039 transactions**
    - Testing Set: Steps 601–743 = **103,565 transactions**
  - **Feature Exclusions:** Strictly excluded `isFlaggedFraud` (naive target leak) and account IDs (`nameOrig`, `nameDest`).
  - **Cost-Sensitive Balancing:** 400x penalty applied to minority fraud class.
- **The HOW:**
  - Spark MLlib Pipeline: `StringIndexer` → `OneHotEncoder` → `VectorAssembler` → Model:
    ```python
    fraud_weight = 400.0  # proportional to 1 / fraud_rate
    train_df = train_df.withColumn(
        "classWeight", F.when(F.col("isFraud") == 1, F.lit(fraud_weight)).otherwise(F.lit(1.0))
    )
    pipeline = Pipeline(stages=[type_indexer, type_encoder, assembler, rf_model])
    ```
- **The WHY:**
  - Standard random train/test splitting on time-series transactions causes temporal data leakage, where the model peeks at future transactions from known mule accounts. Cost-sensitive weighting avoids the data destruction of random undersampling (which throws away 99% of data) and avoids the synthetic hallucinations of SMOTE.
- **Speaker Script:**
  > *"A critical flaw in academic machine learning is random train/test splitting. In financial time-series, random splitting leaks future account states into the past. We enforce a strict chronological cutoff at step 600. Furthermore, rather than throwing away 99% of our data through undersampling, we injected a 400x class weight into Spark’s loss function, heavily penalizing missed fraud."*

---

### Slide 5: Benchmark Results, Confusion Matrix & Explainability
- **Slide Title:** Model Benchmarks: PR-AUC Evaluation & Feature Importance
- **Visual:** Comparative performance table, confusion matrix graphic (101,973 TN, 1,592 TP, 0 FP, 0 FN), and feature importance horizontal bar chart.
- **The WHAT:**
  - Benchmark on Unseen Test Set (103,565 rows):

| Model | PR-AUC (Primary) | ROC-AUC | Weighted F1 | Test Recall | Test Precision |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.6833 | 0.9685 | 0.9395 | 90.7% | 98.5% |
| **Random Forest** (Best) | **1.0000** | **1.0000** | **1.0000** | **100.0%** | **100.0%** |
| **GBTClassifier** | **1.0000** | **1.0000** | **1.0000** | **100.0%** | **100.0%** |

  - **Top Predictive Features:**
    1. `errorBalanceOrig`: **34.24%** (Accounting anomaly at origin)
    2. `newbalanceOrig`: **11.57%** (Residual balance post-transaction)
    3. `amount`: **9.78%** (Capital magnitude)
    4. `oldbalanceOrg`: **7.47%** (Pre-transaction origin liquidity)
- **The HOW:**
  - Evaluated with PySpark `BinaryClassificationEvaluator(metricName="areaUnderPR")`.
  - Feature importances extracted directly from the fitted tree ensemble.
- **The WHY:**
  - Tree ensembles outperform linear models because fraud operates on non-linear thresholds (e.g. large amount *combined* with balance wipeout). Financial regulations require explainability; feature attribution proves the model decisions are anchored in accounting discrepancies rather than random noise.
- **Speaker Script:**
  > *"On the unseen test set of 103,565 transactions, our Random Forest classifier achieved a 1.000 PR-AUC, catching all 1,592 fraud cases with zero false positives. Feature attribution reveals that errorBalanceOrig drives over 34% of the model’s decisions. While synthetic simulator artifacts contribute to this separation, our feature engineering cleanly isolates the criminal signature."*

---

### Slide 6: Production Operations, Serving & Viva Defense Takeaways
- **Slide Title:** Production Serving: Airflow DAG, Kafka Streaming & Streamlit Dashboard
- **Visual:** Split screen showing the Apache Airflow DAG execution graph on the left and the 6-Tab Streamlit analyst dashboard on the right.
- **The WHAT:**
  - **Batch Orchestration:** Apache Airflow DAG (`fraudshield_dag.py`) executing `@daily` across all 6 Medallion stages with automated retries.
  - **Near-Real-Time Streaming:** Kafka producer streaming test-slice transactions paired with Spark Structured Streaming micro-batch scoring.
  - **Analyst Command Center:** Streamlit dashboard with 6 tabs: Overview KPIs, Fraud Patterns, Account Risk, Model Performance with dynamic threshold slider, Live Stream Monitor, and Data Lineage.
  - **Engineering Quality:** 6/6 pytest tests passing, reproducible via single command (`python -m src.pipeline.run_all` or `make all`), and clean GitHub repo (< 200 KB).
- **The HOW:**
  - Decoupled architecture: Airflow automates the heavy batch compute, Parquet provides high-speed columnar data storage, and Streamlit serves cached analytical views.
- **The WHY:**
  - A machine learning model is useless if it sits in a Jupyter Notebook. FraudShield proves end-to-end viability: from data engineering pipelines and workflow orchestration down to an investigator’s interactive triage dashboard.
- **Speaker Script:**
  > *"Finally, FraudShield is not a one-off notebook—it is a production-grade system. Our Airflow DAG manages idempotent daily runs, while our Kafka and Spark Structured Streaming pipeline demonstrates sub-second transaction scoring. Through our Streamlit dashboard, fraud analysts can investigate high-risk destination accounts and test probability thresholds in real time. Thank you, and I look forward to your questions."*

---

# Part 3: Technical "What, How, and Why" Quick Matrix

| Dimension | The WHAT (Design Choice) | The HOW (Implementation) | The WHY (Theoretical Rationale) |
|---|---|---|---|
| **Storage Engine** | Apache Parquet (Snappy) | `df.write.parquet(...)` in 16–20 partitions. | Columnar compression cuts I/O by >70%, enables dictionary encoding, and accelerates analytic scans. |
| **Data Architecture** | 3-Tier Medallion + Quarantine | Bronze (raw), Silver (clean), Gold (marts), and `silver_rejects`. | Eliminates redundant ETL scans, preserves immutable audit history, and complies with financial regulations. |
| **Imbalance Strategy** | Cost-Sensitive Weighting (400x) | `classWeight` column passed to MLlib estimators. | Retains all 6.25M legitimate transactions without the data destruction of undersampling or SMOTE artifacts. |
| **Data Splitting** | Chronological Time Split | Steps 1–600 (Train) / Steps 601–743 (Test). | Eliminates temporal lookahead bias; prevents models from peeking at future destination mule accounts. |
| **Evaluation Metric** | PR-AUC (Precision-Recall) | Evaluated on minority fraud class. | ROC-AUC is misleadingly high (>0.99) due to 99.87% True Negatives. PR-AUC directly tracks fraud capture vs false alarms. |
| **Feature Extraction** | Balance Invariance + Windowing | `errorBalanceOrig` math identity + 24-hr sliding window. | Exposes account-draining anomalies and burst transaction velocities across millions of transactions in parallel. |

---

# Part 4: Examiner Viva Defense Q&A Cheatsheet

### Q1: Why not use SMOTE or Random Undersampling?
> **Answer:** *"Undersampling discards over 99.8% of legitimate user transactions, crippling the model's ability to learn normal behavior. SMOTE synthesizes points in high-dimensional feature spaces, creating artificial transactions that violate financial logic (e.g. impossible balance combinations). 400x cost-sensitive weighting penalizes missed fraud proportionally while keeping all 6.25 million training records intact."*

### Q2: Why did your Random Forest achieve a 1.000 PR-AUC? Is it overfitting?
> **Answer:** *"The 1.000 PR-AUC was measured on the unseen test set (103,565 transactions from steps 601–743), not the training set. It succeeds because our engineered feature, `errorBalanceOrig`, captures the primary mechanic of fraud in PaySim: attackers completely drain origin accounts to zero, creating an unmistakable accounting error that decision trees split cleanly. We acknowledge this as a characteristic of simulator dynamics, while validating that our pipeline generalizes effectively."*

### Q3: What is the purpose of the `silver_rejects` table?
> **Answer:** *"In banking, silently dropping corrupt rows using `dropna()` is illegal under compliance standards like BCBS 239. The quarantine pattern routes invalid transactions to `silver_rejects` with timestamp and rejection codes, allowing data engineering teams to diagnose upstream errors without failing pipeline execution."*

### Q4: Why is `isFlaggedFraud` excluded from the feature set?
> **Answer:** *"`isFlaggedFraud` is a naive rule that flags transfers over 200,000 units, catching only 16 cases out of 8,213. Including it would cause target leakage and bias the model toward a simplistic heuristic. We excluded it along with raw account IDs (`nameOrig`, `nameDest`) so the model learns behavioral patterns rather than memorizing account strings."*

### Q5: How would this architecture scale to 500 million transactions?
> **Answer:** *"Because FraudShield is built on Apache Spark's Catalyst engine, scaling is horizontal. On a cloud cluster (AWS EMR or Databricks), the same code distributes across hundreds of nodes. Furthermore, our window functions partition strictly by account (`nameOrig`/`nameDest`), meaning memory footprints are localized per account rather than requiring an expensive global shuffle."*
