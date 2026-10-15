"""
FraudShield Dashboard — Streamlit + Plotly
==========================================
6 tabs showing real pipeline outputs:
  1. Overview KPIs
  2. Fraud Patterns (by type, hour, day, amount bucket)
  3. Account Risk (top risky destinations)
  4. Model Performance (metrics, confusion matrix, PR curve, feature importance)
  5. Live Stream Monitor (auto-refreshing scored transactions)
  6. Data Lineage / Pipeline Status

Usage:  streamlit run src/dashboard/app.py --server.port 8501
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---- Add project root to path so we can import config ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG

# ============================================================
#  Page config
# ============================================================
st.set_page_config(
    page_title="FraudShield Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stMetric { background: #1e293b; border-radius: 0.5rem; padding: 1rem; }
    h1, h2, h3 { color: #e2e8f0; }
    .block-container { max-width: 1400px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  Helper: safe Parquet loader
# ============================================================
def load_parquet(path: str, label: str = "") -> pd.DataFrame | None:
    """Load a Parquet directory into pandas.  Returns None if missing."""
    if not os.path.exists(path):
        st.warning(f"⚠️ {label or path} not found. Run the corresponding pipeline step first.")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Error reading {label or path}: {e}")
        return None


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================
#  Sidebar
# ============================================================
st.sidebar.title("🛡️ FraudShield")
st.sidebar.markdown("Fraud Detection & Analytics")
st.sidebar.markdown("---")

tab_names = [
    "📊 Overview KPIs",
    "🔍 Fraud Patterns",
    "⚠️ Account Risk",
    "🤖 Model Performance",
    "📡 Live Stream",
    "🔗 Data Lineage",
]
selected_tab = st.sidebar.radio("Navigate", tab_names)

st.sidebar.markdown("---")
st.sidebar.caption("FraudShield v1.0 | Built with PySpark + Streamlit")

# ---- Paths ----
gold_dir = CONFIG["data"]["gold_dir"]
models_dir = CONFIG["data"]["models_dir"]
predictions_dir = CONFIG["data"]["predictions_dir"]
streaming_dir = CONFIG["data"]["streaming_dir"]
bronze_dir = CONFIG["data"]["bronze_dir"]
silver_dir = CONFIG["data"]["silver_dir"]
rejects_dir = CONFIG["data"]["silver_rejects_dir"]

# ============================================================
#  Tab 1: Overview KPIs
# ============================================================
if selected_tab == tab_names[0]:
    st.title("📊 Overview KPIs")

    kpi_path = f"{gold_dir}/gold_kpis"
    kpis = load_parquet(kpi_path, "Gold KPIs")
    if kpis is not None:
        # Calculate totals from KPI by type
        by_type = kpis[kpis["dimension"] == "type"]
        total_txns = int(by_type["total_txns"].sum())
        fraud_count = int(by_type["fraud_count"].sum())
        fraud_amount = float(by_type["fraud_amount"].sum())
        fraud_rate = fraud_count / total_txns * 100 if total_txns > 0 else 0

        # Rejected rows count
        reject_count = 0
        if os.path.exists(rejects_dir):
            try:
                rejects = pd.read_parquet(rejects_dir)
                reject_count = len(rejects)
            except Exception:
                pass

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Transactions", f"{total_txns:,}")
        col2.metric("Fraud Cases", f"{fraud_count:,}")
        col3.metric("Fraud Rate", f"{fraud_rate:.4f}%")
        col4.metric("Fraud Amount", f"${fraud_amount:,.2f}")
        col5.metric("Rejected Rows", f"{reject_count:,}")

        st.markdown("---")

        # Daily summary
        daily = load_parquet(f"{gold_dir}/gold_daily_summary", "Daily Summary")
        if daily is not None:
            fig = px.bar(
                daily.sort_values("day"),
                x="day", y="fraud_count",
                title="Fraud Count by Day",
                labels={"day": "Day", "fraud_count": "Fraud Count"},
                color="fraud_count",
                color_continuous_scale="Reds",
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.line(
                    daily.sort_values("day"),
                    x="day", y="total_txns",
                    title="Daily Transaction Volume",
                    labels={"day": "Day", "total_txns": "Transactions"},
                )
                fig2.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                fig3 = px.line(
                    daily.sort_values("day"),
                    x="day", y="fraud_amount",
                    title="Daily Fraud Amount ($)",
                    labels={"day": "Day", "fraud_amount": "Amount"},
                )
                fig3.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig3, use_container_width=True)


# ============================================================
#  Tab 2: Fraud Patterns
# ============================================================
elif selected_tab == tab_names[1]:
    st.title("🔍 Fraud Patterns")

    kpis = load_parquet(f"{gold_dir}/gold_kpis", "Gold KPIs")
    if kpis is not None:
        col1, col2 = st.columns(2)

        # By type
        by_type = kpis[kpis["dimension"] == "type"].copy()
        with col1:
            fig = px.bar(
                by_type.sort_values("fraud_count", ascending=False),
                x="dimension_value", y="fraud_count",
                title="Fraud Count by Transaction Type",
                color="fraud_rate_pct",
                color_continuous_scale="YlOrRd",
                labels={"dimension_value": "Type", "fraud_count": "Fraud Count"},
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # By hour
        by_hour = kpis[kpis["dimension"] == "hour_of_day"].copy()
        by_hour["dimension_value"] = by_hour["dimension_value"].astype(int)
        with col2:
            fig = px.bar(
                by_hour.sort_values("dimension_value"),
                x="dimension_value", y="fraud_count",
                title="Fraud Count by Hour of Day",
                labels={"dimension_value": "Hour", "fraud_count": "Fraud Count"},
                color="fraud_rate_pct",
                color_continuous_scale="YlOrRd",
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        # By amount bucket
        by_bucket = kpis[kpis["dimension"] == "amount_bucket"].copy()
        with col1:
            fig = px.pie(
                by_bucket,
                names="dimension_value",
                values="fraud_count",
                title="Fraud Distribution by Amount Bucket",
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Fraud rate by type
        with col2:
            fig = px.bar(
                by_type.sort_values("fraud_rate_pct", ascending=False),
                x="dimension_value", y="fraud_rate_pct",
                title="Fraud Rate (%) by Transaction Type",
                labels={"dimension_value": "Type", "fraud_rate_pct": "Fraud Rate %"},
                color="fraud_rate_pct",
                color_continuous_scale="Reds",
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
#  Tab 3: Account Risk
# ============================================================
elif selected_tab == tab_names[2]:
    st.title("⚠️ Top Risky Destination Accounts")

    risk = load_parquet(f"{gold_dir}/gold_account_risk", "Account Risk")
    if risk is not None:
        st.markdown(f"**{len(risk)} accounts** with known fraud inflow:")

        # Top 20 bar chart
        top20 = risk.head(20)
        fig = px.bar(
            top20,
            x="nameDest", y="total_fraud_amount",
            title="Top 20 Fraud-Receiving Accounts",
            labels={"nameDest": "Account", "total_fraud_amount": "Total Fraud Amount ($)"},
            color="fraud_txn_count",
            color_continuous_scale="Reds",
        )
        fig.update_layout(template="plotly_dark", height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Full table
        st.dataframe(
            risk.head(100).style.format({
                "total_fraud_amount": "${:,.2f}",
                "avg_fraud_amount": "${:,.2f}",
            }),
            height=400,
            use_container_width=True,
        )


# ============================================================
#  Tab 4: Model Performance
# ============================================================
elif selected_tab == tab_names[3]:
    st.title("🤖 Model Performance")

    metrics = load_json(f"{models_dir}/metrics.json")
    curves = load_json(f"{models_dir}/curves.json")

    if metrics is None:
        st.warning("⚠️ No metrics found. Run `make train` first.")
    else:
        # Best model info
        st.subheader(f"Best Model: {metrics['best_model']}")
        col1, col2, col3 = st.columns(3)
        best_results = metrics["all_results"][metrics["best_model"]]
        col1.metric("PR-AUC", f"{best_results['PR-AUC']:.4f}")
        col2.metric("ROC-AUC", f"{best_results['ROC-AUC']:.4f}")
        col3.metric("F1 Score", f"{best_results['F1']:.4f}")

        st.markdown("---")

        # All models comparison
        st.subheader("Model Comparison")
        comparison = pd.DataFrame(metrics["all_results"]).T
        st.dataframe(comparison.style.highlight_max(axis=0, color="#22c55e30"),
                     use_container_width=True)

        st.markdown("---")

        # Confusion Matrix
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm = metrics.get("confusion_matrix", {})
            tn = cm.get("actual_0_pred_0", 0)
            fp = cm.get("actual_0_pred_1", 0)
            fn = cm.get("actual_1_pred_0", 0)
            tp = cm.get("actual_1_pred_1", 0)

            fig = go.Figure(data=go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Pred: Legit", "Pred: Fraud"],
                y=["Actual: Legit", "Actual: Fraud"],
                text=[[f"TN={tn:,}", f"FP={fp:,}"],
                      [f"FN={fn:,}", f"TP={tp:,}"]],
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            ))
            fig.update_layout(template="plotly_dark", height=350,
                              title="Confusion Matrix (Default Threshold)")
            st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        with col2:
            st.subheader("Feature Importances")
            fi = metrics.get("feature_importances", {})
            if fi:
                fi_df = pd.DataFrame(
                    sorted(fi.items(), key=lambda x: -x[1]),
                    columns=["Feature", "Importance"],
                )
                fig = px.bar(
                    fi_df.head(15),
                    x="Importance", y="Feature",
                    orientation="h",
                    title="Top 15 Features",
                    color="Importance",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(template="plotly_dark", height=350,
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

        # PR Curve
        if curves and curves.get("pr_curve"):
            st.subheader("Precision-Recall Curve")
            pr_df = pd.DataFrame(curves["pr_curve"])
            fig = px.line(
                pr_df, x="recall", y="precision",
                title=f"PR Curve (AUC={best_results['PR-AUC']:.4f})",
                labels={"recall": "Recall", "precision": "Precision"},
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Threshold slider
        if curves and curves.get("pr_curve"):
            st.subheader("Threshold Explorer")
            threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)
            above = [p for p in curves["pr_curve"] if p["threshold"] >= threshold]
            if above:
                closest = min(above, key=lambda x: abs(x["threshold"] - threshold))
                c1, c2, c3 = st.columns(3)
                c1.metric("Threshold", f"{closest['threshold']:.3f}")
                c2.metric("Precision", f"{closest['precision']:.4f}")
                c3.metric("Recall", f"{closest['recall']:.4f}")

        st.markdown("---")
        st.subheader("Train/Test Split")
        st.markdown(f"""
        - **Train:** steps 1–{metrics.get('train_max_step', '?')} ({metrics.get('train_rows', '?'):,} rows)
        - **Test:** steps {metrics.get('test_min_step', '?')}–743 ({metrics.get('test_rows', '?'):,} rows)
        - Split by **time** (not random) to avoid data leakage.
        """)


# ============================================================
#  Tab 5: Live Stream Monitor
# ============================================================
elif selected_tab == tab_names[4]:
    st.title("📡 Live Stream Monitor")

    if not os.path.exists(streaming_dir) or not any(
        f.endswith(".parquet") for f in os.listdir(streaming_dir)
        if os.path.isfile(os.path.join(streaming_dir, f))
    ):
        st.warning("⚠️ No streaming data found. Start the streaming pipeline first:\n"
                   "```\nmake up              # Start Kafka\n"
                   "make stream-produce  # Replay transactions\n"
                   "make stream-score    # Score with model\n```")
    else:
        # Auto-refresh
        refresh = st.checkbox("Auto-refresh (every 5s)", value=True)
        if refresh:
            st.empty()

        stream_df = pd.read_parquet(streaming_dir)
        total = len(stream_df)
        alerts = len(stream_df[stream_df.get("is_alert", pd.Series(dtype=int)) == 1])

        col1, col2, col3 = st.columns(3)
        col1.metric("Scored Transactions", f"{total:,}")
        col2.metric("Alerts", f"{alerts:,}", delta=f"{alerts/max(total,1)*100:.1f}%")
        col3.metric("Alert Rate", f"{alerts/max(total,1)*100:.2f}%")

        st.markdown("---")

        # Recent alerts
        st.subheader("Recent Alerts")
        alert_df = stream_df[stream_df.get("is_alert", pd.Series(dtype=int)) == 1]
        if len(alert_df) > 0:
            st.dataframe(
                alert_df.sort_values("scored_at", ascending=False).head(50),
                use_container_width=True,
            )
        else:
            st.info("No alerts yet.")

        # Probability distribution
        if "prob_fraud" in stream_df.columns:
            fig = px.histogram(
                stream_df, x="prob_fraud", nbins=50,
                title="Fraud Probability Distribution (Stream)",
                labels={"prob_fraud": "P(Fraud)"},
                color_discrete_sequence=["#ef4444"],
            )
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)

        if refresh:
            import time
            time.sleep(CONFIG["dashboard"]["refresh_interval"])
            st.rerun()


# ============================================================
#  Tab 6: Data Lineage / Pipeline Status
# ============================================================
elif selected_tab == tab_names[5]:
    st.title("🔗 Data Lineage & Pipeline Status")

    st.markdown("""
    ```
    Raw CSV → Bronze (Parquet) → Silver (Clean) → Features → Gold → ML Model → Predictions
                                                                  ↘ Kafka Stream → Scored Events
    ```
    """)

    layers = {
        "Raw CSV": CONFIG["data"]["raw_dir"],
        "Bronze": bronze_dir,
        "Silver": silver_dir,
        "Silver Rejects": rejects_dir,
        "Gold Features": f"{gold_dir}/gold_features",
        "Gold KPIs": f"{gold_dir}/gold_kpis",
        "Gold Account Risk": f"{gold_dir}/gold_account_risk",
        "Gold Daily Summary": f"{gold_dir}/gold_daily_summary",
        "Predictions": predictions_dir,
        "Streaming": streaming_dir,
    }

    rows = []
    for name, path in layers.items():
        if os.path.exists(path):
            try:
                if name == "Raw CSV":
                    import glob
                    csvs = glob.glob(f"{path}/*.csv")
                    count = "CSV file(s) found" if csvs else "No CSV"
                    status = "✅"
                else:
                    df = pd.read_parquet(path)
                    count = f"{len(df):,} rows"
                    status = "✅"
            except Exception:
                count = "Error reading"
                status = "❌"
        else:
            count = "Not generated"
            status = "⏳"
        rows.append({"Layer": name, "Status": status, "Count": count})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Model info
    st.subheader("Model Status")
    model_path = f"{models_dir}/best_pipeline_model"
    if os.path.exists(model_path):
        st.success(f"✅ Trained model saved at `{model_path}`")
        metrics = load_json(f"{models_dir}/metrics.json")
        if metrics:
            st.json(metrics.get("all_results", {}))
    else:
        st.warning("⏳ No trained model. Run `make train`.")
