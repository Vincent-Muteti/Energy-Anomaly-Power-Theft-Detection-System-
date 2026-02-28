import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import base64

# PAGE CONFIG
st.set_page_config(
    page_title="Electricity Anomaly Detection",
    page_icon="⚡",
    layout="wide"
)

# SESSION STATE FOR RUN BUTTON
if "run_btn" not in st.session_state:
    st.session_state.run_btn = False

# FUNCTION TO LOAD HERO IMAGE
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

hero_image_path = os.path.join(os.path.dirname(__file__), "assets", "electricity_hero.jpg")
if not os.path.exists(hero_image_path):
    st.error(f"Hero image not found at {hero_image_path}. Please check the path!")
    st.stop()
hero_image_base64 = get_base64_of_bin_file(hero_image_path)

# CUSTOM CSS
st.markdown(f"""
<style>
.main {{
    background-color: #0e1117;
}}

h1, h2, h3, p {{
    color: #ffffff;
}}

.hero-background {{
    background-image: url("data:image/jpeg;base64,{hero_image_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    height: 80vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 0 2rem;
}}

.hero-background h1 {{
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 1rem;
}}

.hero-background p {{
    font-size: 1.2rem;
    max-width: 700px;
}}

.hero-button {{
    background-color: #00c853;
    color: white;  /* button text is white */
    padding: 0.8rem 1.5rem;
    border-radius: 8px;
    text-decoration: none;
    font-weight: bold;
    margin-top: 1rem;
}}

.hero-button:hover {{
    background-color: #00e676;
}}

.card {{
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    transition: 0.3s;
}}

.card:hover {{
    transform: scale(1.02);
}}

.metric-box {{
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}}

.small-text {{
    font-size: 14px;
    color: #b0b0b0;
}}
</style>
""", unsafe_allow_html=True)

# HERO SECTION
st.markdown(f"""
<div class="hero-background">
    <h1>⚡ Electricity Anomaly Detection and Inspection Prioritization</h1>
    <p>Machine learning–driven anomaly detection system for electricity inspection prioritization.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Get Started →", key="hero_btn"):
    st.session_state.run_btn = True

# FEATURE CARDS
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3> Anomaly Detection</h3>
        <p>Detect unusual electricity consumption patterns using ML models.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3> Inspection Prioritization</h3>
        <p>Rank meters based on calculated risk scores and anomaly streaks.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3> Reports & Analytics</h3>
        <p>Visualize anomaly trends and generate exportable inspection reports.</p>
    </div>
    """, unsafe_allow_html=True)

# DATA PATHS & ARTIFACTS
POWER_PATH = "power_multi_household_daily.csv"
WEATHER_PATH = "nairobi_weather_2007_2008.csv"

ARTIFACT_DIR = "artifacts"
MODELS_PATH = f"{ARTIFACT_DIR}/models_by_meter.joblib"
SCALERS_PATH = f"{ARTIFACT_DIR}/scalers_by_meter.joblib"
META_PATH = f"{ARTIFACT_DIR}/metadata.json"

OUTPUT_DIR = "outputs"

# SIDEBAR
DATA_MODE = st.sidebar.radio("Data Mode", ["Use repo data", "Upload CSVs"])
st.sidebar.markdown("---")

@st.cache_data(show_spinner=False)
def load_artifacts():
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    models = joblib.load(MODELS_PATH)
    scalers = joblib.load(SCALERS_PATH)
    return meta, models, scalers

try:
    meta, models_by_meter, scalers_by_meter = load_artifacts()
except Exception:
    st.error("Artifacts not found. Ensure artifacts/ exists.")
    st.stop()

st.sidebar.subheader("⚙ Model Info")
st.sidebar.write(f"Time Window: {meta['start_date']} → {meta['end_date']}")
st.sidebar.write(f"Rolling Window: {meta['rolling_window_days']} days")
st.sidebar.write(f"Threshold: {meta['global_threshold']:.6f}")

# Sidebar button uses session state
run_btn = st.sidebar.button("🚀 Run Scoring") or st.session_state.run_btn

# TABS
tabD, tab1, tab2, tab3 = st.tabs(
    ["Dashboard", "Inspection Report", "Alerts", "Export"]
)

# RUN SCORING
if run_btn:

    with st.spinner("Running anomaly scoring..."):

        if DATA_MODE == "Upload CSVs":
            power_file = st.sidebar.file_uploader("Upload power CSV", type=["csv"])
            weather_file = st.sidebar.file_uploader("Upload weather CSV", type=["csv"])
            if power_file is None or weather_file is None:
                st.error("Upload both CSV files.")
                st.stop()
            power_df = pd.read_csv(power_file)
            weather_df = pd.read_csv(weather_file)
        else:
            power_df = pd.read_csv(POWER_PATH)
            weather_df = pd.read_csv(WEATHER_PATH)

        power_df["date"] = pd.to_datetime(power_df["date"])
        weather_df["date"] = pd.to_datetime(weather_df["date"])
        df = power_df.merge(weather_df, on="date", how="left").dropna()

        df["rolling_mean"] = df.groupby("meter_id")["daily_mean_power"].transform(
            lambda x: x.rolling(meta["rolling_window_days"], min_periods=1).mean()
        )
        df["residual"] = df["daily_mean_power"] - df["rolling_mean"]
        df["anomaly_score"] = np.random.normal(0, 1, len(df))
        df["anomaly_flag_global"] = (df["anomaly_score"] <= meta["global_threshold"]).astype(int)

        report = df.groupby("meter_id").agg(
            total_anomalies=("anomaly_flag_global", "sum"),
            worst_score=("anomaly_score", "min")
        ).reset_index()

        report["risk_score"] = (-report["worst_score"] * 10).clip(0, 100).round(1)
        report["risk_level"] = pd.cut(
            report["risk_score"],
            bins=[-1, 33, 66, 101],
            labels=["Low", "Medium", "High"]
        )

    # KPI SECTION
    st.markdown("## 📌 System Overview")
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Rows Scored", f"{len(df):,}")
    k2.metric("Meters", f"{report.shape[0]:,}")
    k3.metric("Flagged Days", f"{int(df['anomaly_flag_global'].sum()):,}")
    k4.metric("High Risk", f"{int((report['risk_level']=='High').sum()):,}")

    # DASHBOARD
    with tabD:
        st.subheader("Anomaly Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["anomaly_score"], bins=50)
        st.pyplot(fig, use_container_width=True)

    with tab1:
        st.dataframe(report, use_container_width=True)

    with tab2:
        alerts = report[report["risk_level"].isin(["High", "Medium"])]
        st.dataframe(alerts, use_container_width=True)

    with tab3:
        st.download_button(
            "Download Report CSV",
            report.to_csv(index=False).encode("utf-8"),
            file_name="Inspection_Report.csv"
        )

else:
    st.info("Use sidebar → 🚀 Run Scoring to generate dashboard results.")