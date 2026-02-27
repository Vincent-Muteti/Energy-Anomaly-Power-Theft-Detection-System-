import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------
# Config (edit if needed)
# -----------------------
POWER_PATH = "power_multi_household_daily.csv"
WEATHER_PATH = "nairobi_weather_2007_2008.csv"

START_DATE = "2007-01-01"
END_DATE = "2008-12-31"

ROLLING_WINDOW_DAYS = 30
PER_METER_CONTAMINATION = 0.02   # used only to stabilize per-meter scoring
GLOBAL_ANOMALY_RATE = 0.02       # top 2% most anomalous days overall

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

FEATURES = [
    "residual_30",
    "z_score_30",
    "daily_std_power",
    "voltage_std",
    "intensity_mean",
    "tmax",
    "tmin",
    "prcp",
]

# -----------------------
# Load data
# -----------------------
power_df = pd.read_csv(POWER_PATH)
weather_df = pd.read_csv(WEATHER_PATH)

power_df["date"] = pd.to_datetime(power_df["date"])
weather_df["date"] = pd.to_datetime(weather_df["date"])

df = power_df.merge(weather_df, on="date", how="left")
df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
df = df.dropna().copy()
df = df.sort_values(["meter_id", "date"]).copy()

# -----------------------
# Feature engineering
# -----------------------
w = int(ROLLING_WINDOW_DAYS)

df["rolling_mean_30"] = df.groupby("meter_id")["daily_mean_power"].transform(
    lambda x: x.rolling(w, min_periods=1).mean()
)
df["rolling_std_30"] = df.groupby("meter_id")["daily_mean_power"].transform(
    lambda x: x.rolling(w, min_periods=1).std()
)
df["residual_30"] = df["daily_mean_power"] - df["rolling_mean_30"]
df["z_score_30"] = df["residual_30"] / (df["rolling_std_30"] + 1e-6)

# -----------------------
# Train per-meter models
# -----------------------
models_by_meter = {}
scalers_by_meter = {}

df["anomaly_score"] = np.nan

for meter_id, g in df.groupby("meter_id"):
    X = g[FEATURES].fillna(0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=150,
        contamination=float(PER_METER_CONTAMINATION),
        random_state=42
    )
    iso.fit(Xs)

    scores = iso.decision_function(Xs)

    models_by_meter[str(meter_id)] = iso
    scalers_by_meter[str(meter_id)] = scaler
    df.loc[g.index, "anomaly_score"] = scores

# -----------------------
# Global threshold (top X% anomalous)
# -----------------------
global_threshold = float(df["anomaly_score"].quantile(GLOBAL_ANOMALY_RATE))

# -----------------------
# Save artifacts
# -----------------------
joblib.dump(models_by_meter, os.path.join(ARTIFACT_DIR, "models_by_meter.joblib"))
joblib.dump(scalers_by_meter, os.path.join(ARTIFACT_DIR, "scalers_by_meter.joblib"))

meta = {
    "trained_at": datetime.utcnow().isoformat() + "Z",
    "start_date": START_DATE,
    "end_date": END_DATE,
    "rolling_window_days": ROLLING_WINDOW_DAYS,
    "per_meter_contamination": PER_METER_CONTAMINATION,
    "global_anomaly_rate": GLOBAL_ANOMALY_RATE,
    "global_threshold": global_threshold,
    "features": FEATURES,
}

with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Training complete.")
print(f"Meters trained: {len(models_by_meter)}")
print(f"Global threshold: {global_threshold}")
print(f"Artifacts saved to: {ARTIFACT_DIR}/")