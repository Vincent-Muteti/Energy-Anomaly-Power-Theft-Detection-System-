import json
import os
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Inspection Prioritization", layout="wide")

st.title("Electricity Anomaly Detection and Inspection Prioritization")
st.caption("Machine learning–driven anomaly detection for electricity inspection prioritization.")

DATA_MODE = st.sidebar.radio("Data mode", ["Use repo data", "Upload CSVs"])
st.sidebar.markdown("---")

POWER_PATH = "power_multi_household_daily.csv"
WEATHER_PATH = "nairobi_weather_2007_2008.csv"

ARTIFACT_DIR = "artifacts"
MODELS_PATH = f"{ARTIFACT_DIR}/models_by_meter.joblib"
SCALERS_PATH = f"{ARTIFACT_DIR}/scalers_by_meter.joblib"
META_PATH = f"{ARTIFACT_DIR}/metadata.json"

# Outputs directory
OUTPUT_DIR = "outputs"
SCORED_PATH = f"{OUTPUT_DIR}/scored_output.csv"
REPORT_PATH = f"{OUTPUT_DIR}/inspection_report.csv"
META_OUT_PATH = f"{OUTPUT_DIR}/run_metadata.json"


def generate_alert(row: pd.Series) -> str:
    if row["risk_level"] in ["High", "Medium"]:
        last_date = row["last_anomaly_date"]
        last_date_str = last_date.date().isoformat() if pd.notna(last_date) else "N/A"
        return (
            f"ALERT: Meter {row['meter_id']} is {row['risk_level']} risk "
            f"(Risk Score: {row['risk_score']}). "
            f"Anomalous days: {int(row['total_anomalies'])}; "
            f"Max streak: {int(row['max_streak_days'])} days; "
            f"Last anomaly: {last_date_str}. "
            "Recommended for inspection review."
        )
    return "No immediate inspection required."


def compute_max_streak(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    df = df.sort_values(["meter_id", "date"]).copy()
    df["prev_flag"] = df.groupby("meter_id")[flag_col].shift(1)
    df["start_streak"] = (df[flag_col].eq(1) & ~df["prev_flag"].eq(1))
    df["streak_group"] = df.groupby("meter_id")["start_streak"].cumsum()

    streak_lengths = (
        df[df[flag_col] == 1]
        .groupby(["meter_id", "streak_group"])
        .size()
        .reset_index(name="streak_len")
    )

    if streak_lengths.empty:
        return pd.DataFrame({"meter_id": df["meter_id"].unique(), "max_streak_days": 0})

    max_streak = streak_lengths.groupby("meter_id")["streak_len"].max().reset_index()
    max_streak.columns = ["meter_id", "max_streak_days"]
    return max_streak


@st.cache_data(show_spinner=False)
def load_artifacts():
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    models_by_meter = joblib.load(MODELS_PATH)
    scalers_by_meter = joblib.load(SCALERS_PATH)
    return meta, models_by_meter, scalers_by_meter


def load_data(power_df, weather_df, start_date, end_date):
    power_df["date"] = pd.to_datetime(power_df["date"])
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    df = power_df.merge(weather_df, on="date", how="left")
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    df = df.dropna().sort_values(["meter_id", "date"])
    return df


def add_features(df, rolling_window_days):
    w = int(rolling_window_days)
    df = df.sort_values(["meter_id", "date"]).copy()

    df["rolling_mean_30"] = df.groupby("meter_id")["daily_mean_power"].transform(
        lambda x: x.rolling(w, min_periods=1).mean()
    )
    df["rolling_std_30"] = df.groupby("meter_id")["daily_mean_power"].transform(
        lambda x: x.rolling(w, min_periods=1).std()
    )
    df["residual_30"] = df["daily_mean_power"] - df["rolling_mean_30"]
    df["z_score_30"] = df["residual_30"] / (df["rolling_std_30"] + 1e-6)
    return df


def score(df, features, models_by_meter, scalers_by_meter):
    df["anomaly_score"] = np.nan

    for meter_id, g in df.groupby("meter_id"):
        key = str(meter_id)
        if key not in models_by_meter:
            continue
        X = g[features].fillna(0)
        Xs = scalers_by_meter[key].transform(X)
        df.loc[g.index, "anomaly_score"] = models_by_meter[key].decision_function(Xs)

    return df


def make_report(df, threshold):
    df["anomaly_flag_global"] = (df["anomaly_score"] <= threshold).astype(int)

    max_streak = compute_max_streak(df, "anomaly_flag_global")

    report = df.groupby("meter_id").agg(
        total_anomalies=("anomaly_flag_global", "sum"),
        percent_anomalous=("anomaly_flag_global", "mean"),
        worst_anomaly_score=("anomaly_score", "min"),
        avg_anomaly_score=("anomaly_score", "mean"),
        last_anomaly_date=("date", lambda x: x[df.loc[x.index, "anomaly_flag_global"].eq(1)].max()),
    ).reset_index()

    report = report.merge(max_streak, on="meter_id", how="left")
    report["max_streak_days"] = report["max_streak_days"].fillna(0).astype(int)

    score_raw = -report["worst_anomaly_score"]
    smin, smax = score_raw.min(), score_raw.max()
    report["risk_score"] = 100 * (score_raw - smin) / (smax - smin + 1e-9)
    report["risk_score"] = (report["risk_score"] + report["max_streak_days"] * 5).clip(0, 100).round(1)

    report["risk_level"] = pd.cut(
        report["risk_score"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"],
    )

    report["alert_message"] = report.apply(generate_alert, axis=1)
    return report


# --------------------
# App Flow
# --------------------

meta, models_by_meter, scalers_by_meter = load_artifacts()

run_btn = st.sidebar.button("Run Scoring")

if DATA_MODE == "Upload CSVs":
    power_file = st.sidebar.file_uploader("Upload power CSV", type=["csv"])
    weather_file = st.sidebar.file_uploader("Upload weather CSV", type=["csv"])
else:
    power_file = None
    weather_file = None


tab0, tab1, tab2, tab3 = st.tabs(["How to Use", "Inspection Report", "Alerts", "Export"])

with tab0:
    st.subheader("How to use this app")

    st.markdown("""
1. Choose **Data mode** in the sidebar:
   - **Use repo data** (demo mode), or
   - **Upload CSVs** (your own datasets).
2. Click **Run Scoring** to compute anomaly scores.
3. View prioritized meters in the Inspection Report tab.
4. Download the final inspection CSV from the Export tab.
""")

    st.warning("""
Upload Mode Requirements:
- Power CSV must contain: meter_id, date, daily_mean_power
- Weather CSV must contain: date and required weather columns used in training
- Column names must match exactly.
""")

    st.caption("Note: This system flags statistical anomalies. It does not confirm electricity theft.")


if run_btn:
    if DATA_MODE == "Upload CSVs":
        if power_file is None or weather_file is None:
            st.error("Upload both CSV files.")
            st.stop()
        power_df = pd.read_csv(power_file)
        weather_df = pd.read_csv(weather_file)
    else:
        power_df = pd.read_csv(POWER_PATH)
        weather_df = pd.read_csv(WEATHER_PATH)

    df = load_data(power_df, weather_df, meta["start_date"], meta["end_date"])
    df = add_features(df, meta["rolling_window_days"])
    df = score(df, meta["features"], models_by_meter, scalers_by_meter)

    report = make_report(df, meta["global_threshold"])

    with tab1:
        st.dataframe(report, use_container_width=True)

    with tab2:
        st.dataframe(
            report[["meter_id", "risk_level", "risk_score", "alert_message"]],
            use_container_width=True,
        )

    with tab3:
        csv_bytes = report.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Inspection Report",
            csv_bytes,
            file_name="Inspection_Report.csv",
            mime="text/csv",
        )
else:
    with tab1:
        st.info("Run scoring first (sidebar → Run Scoring).")
    with tab2:
        st.info("Run scoring first (sidebar → Run Scoring).")
    with tab3:
        st.info("Run scoring first (sidebar → Run Scoring).")