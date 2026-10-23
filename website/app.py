
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import average_precision_score

st.set_page_config(page_title="Sepsis AI", page_icon="🩺", layout="wide")

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
RESULTS = BASE / "results"
DATA = BASE / "processed"

st.title("🩺 Early Sepsis Detection")
st.caption("Explainable AI research prototype | PhysioNet 2019")

st.warning(
    "Research prototype only. Not validated for clinical use. "
    "Do not use this dashboard to make patient-care decisions."
)

@st.cache_resource
def load_xgb():
    model = xgb.XGBClassifier()
    model.load_model(str(MODELS / "xgboost_tuned_candidate.json"))
    return model

@st.cache_resource
def load_lstm():
    import tensorflow as tf
    return tf.keras.models.load_model(MODELS / "lstm_sepsis.keras")

@st.cache_data
def load_comparison():
    return pd.read_csv(RESULTS / "final_model_comparison.csv")

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Model Comparison", "Risk Prediction", "Explainability"
])

with tab1:
    st.subheader("Project Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients processed", "40,336")
    c2.metric("Hourly records", "1.55 million")
    c3.metric("Input features", "52")
    st.write(
        "This project compares XGBoost and LSTM models for predicting "
        "sepsis-labelled hours from ICU observations."
    )
    st.info(
        "The models predict dataset labels, not confirmed clinical diagnoses. "
        "Performance depends on the threshold and evaluation method."
    )

with tab2:
    st.subheader("Held-out Test Results")
    comparison = load_comparison()
    st.dataframe(comparison.round(4), use_container_width=True)
    st.bar_chart(comparison.set_index("Model")[["PR-AUC", "ROC-AUC", "F1"]])
    st.caption(
        "Recall and precision use the listed thresholds. "
        "The tuned XGBoost threshold was selected using validation data."
    )

with tab3:
    st.subheader("Predict from uploaded observations")
    st.write(
        "Upload a CSV containing the model's required feature columns. "
        "For LSTM, provide at least 24 chronological hourly rows for one patient."
    )
    model_choice = st.selectbox("Model", ["Tuned XGBoost", "LSTM"])
    uploaded = st.file_uploader("Upload feature CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        features = joblib.load(MODELS / "lstm_features.joblib")

        xgb_features = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST",
        "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
        "Bilirubin_direct", "Glucose", "Lactate", "Magnesium",
        "Phosphate", "Potassium", "Bilirubin_total", "TroponinI",
        "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
        "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
        "HR_prev1", "HR_mean6", "O2Sat_prev1", "O2Sat_mean6",
        "Temp_prev1", "Temp_mean6", "SBP_prev1", "SBP_mean6",
        "MAP_prev1", "MAP_mean6", "Resp_prev1", "Resp_mean6"
        ]
        if model_choice == "Tuned XGBoost":
            missing = sorted(set(xgb_features) - set(df.columns))
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                model = load_xgb()
                probability = model.predict_proba(df[xgb_features])[:, 1]
                st.dataframe(pd.DataFrame({
                    "Row": np.arange(1, len(probability) + 1),
                    "Predicted probability": probability,
                    "Flag at threshold 0.13": probability >= 0.13
                }))
                st.caption("Threshold 0.13 is a research threshold, not a clinical alarm threshold.")

        else:
            if len(df) < 24:
                st.error("LSTM requires at least 24 chronological hourly rows.")
            elif not set(features).issubset(df.columns):
                st.error("CSV is missing one or more required LSTM features.")
            else:
                imputer = joblib.load(MODELS / "lstm_imputer.joblib")
                scaler = joblib.load(MODELS / "lstm_scaler.joblib")
                X = scaler.transform(imputer.transform(df[features])).astype(np.float32)
                windows = np.lib.stride_tricks.sliding_window_view(
                    X, window_shape=24, axis=0
                ).transpose(0, 2, 1)
                probabilities = load_lstm().predict(windows, verbose=0).ravel()
                st.dataframe(pd.DataFrame({
                    "Ending hour": np.arange(24, len(df) + 1),
                    "Predicted probability": probabilities,
                    "Flag at threshold 0.50": probabilities >= 0.50
                }))

with tab4:
    st.subheader("Global Feature Importance (SHAP)")
    chart = RESULTS / "shap_feature_importance.png"
    if chart.exists():
        st.image(str(chart), caption="Mean absolute SHAP values on a test-record sample")
    else:
        st.info("SHAP chart not found. Generate it from the notebook first.")
    st.caption(
        "SHAP describes model influence, not causation. "
        "Time-related features such as ICULOS may strongly influence predictions."
    )
