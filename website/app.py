
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

    # Display the evaluation metrics in a table.
    st.dataframe(comparison.round(4), use_container_width=True)

    # Compare PR-AUC and F1.
    st.markdown("#### Precision–Recall Performance")
    st.bar_chart(
        comparison.set_index("Model")[["PR-AUC", "F1"]],
        stack=False,
        height=350
    )

    # Compare ROC-AUC separately.
    st.markdown("#### ROC-AUC Comparison")
    st.bar_chart(
        comparison.set_index("Model")[["ROC-AUC"]],
        height=300
    )

    # Explain the evaluation thresholds.
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
                # FINAL: Predict sepsis-label probabilities with tuned XGBoost.
                model = load_xgb()
                X_input = df[xgb_features]
                probability = model.predict_proba(X_input)[:, 1]

                # Display predictions for every uploaded row.
                st.dataframe(pd.DataFrame({
                    "Row": np.arange(1, len(probability) + 1),
                    "Predicted probability": probability,
                    "Flag at threshold 0.13": probability >= 0.13
                }))

                st.caption(
                    "Threshold 0.13 is a research threshold, "
                    "not a clinical alarm threshold."
                )

                # Explain the row with the highest predicted probability.
                st.markdown("### Explanation for the highest-scoring row")
                row_idx = int(np.argmax(probability))
                explainer = shap.TreeExplainer(model)
                shap_values = np.asarray(
                    explainer.shap_values(X_input.iloc[[row_idx]])
                ).reshape(-1)

                # Rank features by absolute SHAP impact.
                explanation = pd.DataFrame({
                    "Feature": xgb_features,
                    "SHAP value": shap_values
                })
                explanation["Absolute impact"] = (
                    explanation["SHAP value"].abs()
                )
                explanation = explanation.sort_values(
                    "Absolute impact", ascending=False
                ).head(10).sort_values("SHAP value")

                # Positive and negative bars show direction of influence.
                st.bar_chart(
                    explanation.set_index("Feature")["SHAP value"],
                    horizontal=True
                )
                st.caption(
                    "Positive SHAP values push the model output higher; "
                    "negative values push it lower. These are contributions "
                    "on the model's raw output scale, not percentage points."
                )


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
                            # Explain the highest-scoring LSTM prediction.
            st.subheader("LSTM: Individual Prediction Explanation")

            # Select the window with the highest predicted probability.
            best_idx = int(np.argmax(probabilities))
            best_window = windows[best_idx:best_idx + 1]

            # Explain how input features influence this prediction.
            st.write(
                f"Highest predicted probability: "
                f"{probabilities[best_idx]:.4f}"
            )
            st.caption(
                "This is a perturbation-based explanation, not a "
                "clinical diagnosis."
            )

            # Estimate feature influence by masking one feature at a time.
            baseline = float(probabilities[best_idx])
            impacts = []

            for feature_idx, feature_name in enumerate(features):
                # Replace this feature with zero across the 24-hour window.
                modified = best_window.copy()
                modified[:, :, feature_idx] = 0.0

                # Measure the change in model prediction.
                changed_prob = float(
                    load_lstm().predict(modified, verbose=0).ravel()[0]
                )
                impacts.append({
                    "Feature": feature_name,
                    "Prediction change": changed_prob - baseline,
                    "Absolute impact": abs(changed_prob - baseline)
                })

            # Display the features with the largest estimated influence.
            impact_df = pd.DataFrame(impacts).sort_values(
                "Absolute impact", ascending=False
            )
            st.dataframe(impact_df.head(10), use_container_width=True)

                    # Display LSTM feature impacts in a responsive chart.
        # Horizontal bars keep feature names readable when zooming.
        st.markdown("#### Feature Impact (Top 10)")

        chart_data = impact_df.head(10).set_index("Feature")[
            "Prediction change"
        ]

        st.bar_chart(
            chart_data,
            horizontal=True,
            height=450,
            use_container_width=True
        )


with tab4:
    st.subheader("Global Feature Importance (SHAP)")

    chart = RESULTS / "shap_feature_importance.png"

    if chart.exists():
        st.image(
            str(chart),
            caption="Mean absolute SHAP values on a test-record sample",
            width="stretch"
        )
    else:
        st.info("SHAP chart not found. Generate it from the notebook first.")

    st.markdown("### How to understand this chart")

    st.write(
        "SHAP explains how strongly each feature influences the "
        "model's predictions across the sampled records. Longer bars "
        "indicate greater average influence."
    )

    st.markdown("**Key observations from our results:**")
    st.markdown("""
    - **ICULOS:** Time elapsed in the ICU.
    - **HospAdmTime:** Time between hospital admission and ICU admission,
      represented in the dataset as a time value.
    - **Temp_mean6:** Six-hour rolling average body temperature.
    - **Resp_mean6:** Six-hour rolling average respiratory rate.
    """)

    st.info(
        "Important: Feature importance shows model influence, not causation. "
        "It does not prove that a feature causes sepsis or confirm a diagnosis."
    )

    st.caption(
        "These explanations summarize the model's behavior on a sample; "
        "they are not individual-patient explanations."
    )

