import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Visualizations", layout="wide")
st.title("Anomaly Detection Visualizations")

# ---- Paths saved by app.py ----
SCORED_PATH = "outputs/scored_output.csv"
REPORT_PATH = "outputs/inspection_report.csv"

# ---- Check files exist ----
if not os.path.exists(SCORED_PATH):
    st.warning("Scored dataset not found. Run scoring first on the main page.")
    st.stop()

df = pd.read_csv(SCORED_PATH)

# Convert date
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# ----------------------------
# 1) Score Distribution
# ----------------------------
st.subheader("Global Anomaly Score Distribution")

fig, ax = plt.subplots()
ax.hist(df["anomaly_score"].dropna(), bins=60)
ax.set_xlabel("Anomaly Score (lower = more anomalous)")
ax.set_ylabel("Count")
st.pyplot(fig)

st.caption("Lower anomaly scores represent more suspicious observations.")

# ----------------------------
# 2) Meter-Level Time Series
# ----------------------------
st.subheader("Meter-Level Time Series View")

meter_list = sorted(df["meter_id"].astype(str).unique())
selected_meter = st.selectbox("Select Meter", meter_list)

meter_df = df[df["meter_id"].astype(str) == selected_meter].sort_values("date")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(meter_df["date"], meter_df["daily_mean_power"], label="Daily Mean Power")

if "rolling_mean_30" in meter_df.columns:
    ax2.plot(meter_df["date"], meter_df["rolling_mean_30"], label="30-Day Rolling Mean")

# highlight anomalies
if "anomaly_flag_global" in meter_df.columns:
    anomalies = meter_df[meter_df["anomaly_flag_global"] == 1]
    ax2.scatter(anomalies["date"], anomalies["daily_mean_power"], label="Flagged Anomalies")

ax2.legend()
ax2.set_xlabel("Date")
ax2.set_ylabel("Power")
st.pyplot(fig2)

# ----------------------------
# 3) Risk Ranking (from report)
# ----------------------------
st.subheader("Inspection Priority Ranking (Risk Score)")

if not os.path.exists(REPORT_PATH):
    st.info("Inspection report not found yet. Run scoring first to generate outputs/inspection_report.csv.")
    st.stop()

report = pd.read_csv(REPORT_PATH)

if "risk_score" in report.columns:
    report = report.sort_values("risk_score", ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.bar(report["meter_id"].astype(str), report["risk_score"].astype(float))
    ax3.set_ylabel("Risk Score (0–100)")
    ax3.set_xlabel("Meter ID")
    st.pyplot(fig3)
else:
    st.info("risk_score column not found in inspection report.")