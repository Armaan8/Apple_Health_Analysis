# pages/09_Train_Test_Validate_Insights.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

from utils.data_loader import load_health_data

# -------------------
# Helpers
# -------------------
def safe_mape(y_true, y_pred):
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def ensure_date(df, date_col="date"):
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df[date_col].dt.year
    return df

def generate_shift_insight(feature, delta, corr, importance):
    if abs(delta) < 0.2:  # ignore tiny shifts
        return None

    if "HRV" in feature or "HeartRateVariability" in feature:
        if delta > 0:
            return f"Heart rate variability increased (+{delta:.1f}), suggesting better recovery and autonomic balance, which supports a lower resting heart rate."
        else:
            return f"Heart rate variability decreased ({delta:.1f}), indicating more stress or reduced recovery, which tends to elevate resting heart rate."

    if "HeartRate" in feature and "WalkingHeartRateAverage" not in feature:
        if delta > 0:
            return f"Average heart rate rose (+{delta:.1f}), often a sign of higher cardiovascular strain, which can raise resting heart rate."
        else:
            return f"Average heart rate fell ({delta:.1f}), reflecting less strain, which usually lowers resting heart rate."

    if "WalkingHeartRateAverage" in feature:
        if delta > 0:
            return f"Walking heart rate average increased (+{delta:.1f}), suggesting walks were more intense, which could elevate resting heart rate."
        else:
            return f"Walking heart rate average decreased ({delta:.1f}), meaning walks were easier, which supports lower resting heart rate."

    if "HRFitnessIndex" in feature:
        if delta > 0:
            return f"Fitness index improved (+{delta:.1f}), reflecting stronger cardiovascular capacity and supporting lower resting heart rate."
        else:
            return f"Fitness index declined ({delta:.1f}), pointing to reduced fitness, which can raise resting heart rate."

    if "WalkingStepLength" in feature or "MobilityIndex" in feature:
        if delta > 0:
            return f"Mobility metrics improved (+{delta:.1f}), suggesting stronger gait efficiency, which indirectly supports a healthier resting heart rate."
        else:
            return f"Mobility metrics declined ({delta:.1f}), possibly reflecting reduced gait efficiency, which can contribute to higher resting heart rate."

    if "WalkingDoubleSupportPercentage" in feature:
        if delta > 0:
            return f"Double support percentage increased (+{delta:.1f}), showing more cautious walking. This usually signals reduced stability, indirectly raising resting heart rate."
        else:
            return f"Double support percentage decreased ({delta:.1f}), indicating more confident gait, which supports a healthier resting heart rate."

    if "BasalEnergyBurned" in feature or "EnergyBalance" in feature:
        if delta > 0:
            return f"Energy expenditure rose (+{delta:.1f}), meaning the body was burning more baseline calories, which can support lower resting heart rate."
        else:
            return f"Energy expenditure dropped ({delta:.1f}), possibly reflecting reduced activity, which may contribute to higher resting heart rate."

    if "HeadphoneAudioExposure" in feature:
        if delta > 0:
            return f"Audio exposure increased (+{delta:.1f}), which might link to lifestyle stress or environmental noise, potentially nudging resting heart rate upward."
        else:
            return f"Audio exposure decreased ({delta:.1f}), suggesting a calmer environment, which may help resting heart rate stay lower."

    # fallback
    direction = "increased" if delta > 0 else "decreased"
    return f"{feature} {direction} ({delta:.1f}), influencing resting heart rate according to its learned importance."

# -------------------
# Streamlit App
# -------------------
st.set_page_config(page_title="Train/Test/Validate Insights", layout="wide")
st.title("Train-Test-Validate Influence Insights on Resting Heart Rate")

df = load_health_data()
if df is None or df.empty:
    st.error("No data loaded.")
    st.stop()

df = ensure_date(df, "date")
df = df.dropna(axis=1, how="all")

if "RestingHeartRate" not in df.columns:
    st.error("No RestingHeartRate column found.")
    st.stop()

# Candidate features
candidate_features = [
    c for c in df.columns
    if c not in ["date", "RestingHeartRate", "year"] and pd.api.types.is_numeric_dtype(df[c])
]
if not candidate_features:
    st.error("No numeric features available to model.")
    st.stop()

# Year check
years = sorted(df["year"].unique())
if 2023 not in years or 2024 not in years:
    st.error("This page requires at least 2023 and 2024 data.")
    st.stop()

train_df = df[df["year"] == 2023].copy()
validate_df = df[df["year"] == 2024].copy()

# Drop missing targets
train_df = train_df.dropna(subset=["RestingHeartRate"])
validate_df = validate_df.dropna(subset=["RestingHeartRate"])

if train_df.empty or validate_df.empty:
    st.error("Not enough non-missing RestingHeartRate values in selected years.")
    st.stop()

# In-sample split for train/test within 2023
cutoff_idx = int(len(train_df) * 0.8)
train_in, test_in = train_df.iloc[:cutoff_idx], train_df.iloc[cutoff_idx:]

X_train, y_train = train_in[candidate_features], train_in["RestingHeartRate"]
X_test, y_test = test_in[candidate_features], test_in["RestingHeartRate"]
X_val, y_val = validate_df[candidate_features], validate_df["RestingHeartRate"]

# Train model
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])
pipe.fit(X_train, y_train)

# Predict
pred_test = pipe.predict(X_test)
pred_val = pipe.predict(X_val)

# Metrics
mae_test = mean_absolute_error(y_test, pred_test)
r2_test = r2_score(y_test, pred_test) if len(y_test) > 1 else np.nan
mape_test = safe_mape(y_test, pred_test)

mae_val = mean_absolute_error(y_val, pred_val)
r2_val = r2_score(y_val, pred_val) if len(y_val) > 1 else np.nan
mape_val = safe_mape(y_val, pred_val)

# -------------------
# Outputs
# -------------------
st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("2023 Test MAE", f"{mae_test:.2f}")
col2.metric("2023 Test MAPE", f"{mape_test:.1f}%")
col3.metric("2023 Test R²", f"{r2_test:.3f}")

col4, col5, col6 = st.columns(3)
col4.metric("2024 Validation MAE", f"{mae_val:.2f}")
col5.metric("2024 Validation MAPE", f"{mape_val:.1f}%")
col6.metric("2024 Validation R²", f"{r2_val:.3f}")

# Average comparison
avg_train = y_train.mean()
avg_val = y_val.mean()
st.subheader("Average Resting Heart Rate Comparison")
st.markdown(f"- 2023 average: **{avg_train:.1f} bpm**")
st.markdown(f"- 2024 average: **{avg_val:.1f} bpm**")
st.markdown(f"- Change: **{avg_val - avg_train:+.1f} bpm**")

# Feature importance
rf = pipe.named_steps["model"]
importances = rf.feature_importances_
fi_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
})
fi_df = fi_df.sort_values("Importance", ascending=False).head(10)

st.subheader("Top 10 Influential Features (from 2023 training)")
fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="Feature Importance")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(fi_df, use_container_width=True)

# Year-over-year shifts
st.subheader("Year-over-Year Feature Shifts")
shift_records = []
for _, row in fi_df.iterrows():
    f = row["Feature"]
    imp = row["Importance"]
    mean_train = train_df[f].mean()
    mean_val = validate_df[f].mean()
    delta = mean_val - mean_train
    corr = np.corrcoef(df[f].fillna(0), df["RestingHeartRate"].fillna(0))[0,1]
    shift_records.append((f, mean_train, mean_val, delta, imp, corr))

shift_df = pd.DataFrame(shift_records, columns=["Feature","Mean_2023","Mean_2024","Delta","Importance","Corr"])
st.dataframe(shift_df[["Feature","Mean_2023","Mean_2024","Delta","Importance"]], use_container_width=True)

# Insights
st.subheader("Actionable Insights Explaining 2024 Resting HR")
for _, row in shift_df.iterrows():
    insight = generate_shift_insight(row['Feature'], row['Delta'], row['Corr'], row['Importance'])
    if insight:
        st.markdown(f"- {insight}")

