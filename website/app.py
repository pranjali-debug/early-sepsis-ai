
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
    st.subheader("Predict Sepsis Risk")

    uploaded_file = st.file_uploader(
        "Upload patient hourly data (CSV)",
        type=["csv"],
        key="risk_upload"
    )

    model_choice = st.selectbox(
        "Choose prediction model",
        ["Tuned XGBoost", "LSTM"]
    )

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("Uploaded data preview")
        st.dataframe(data.head(), use_container_width=True)

        if model_choice == "Tuned XGBoost":
            # Load the tuned XGBoost model and feature list.
            model_path = os.path.join(
                MODEL_DIR, "xgboost_tuned_candidate.json"
            )
            model = xgb.XGBClassifier()
            model.load_model(model_path)

            features_path = os.path.join(
                MODEL_DIR, "lstm_features.joblib"
            )
            # Use the XGBoost feature list if your project saves one.
            if os.path.exists(
                os.path.join(MODEL_DIR, "xgb_features.joblib")
            ):
                features = joblib.load(
                    os.path.join(MODEL_DIR, "xgb_features.joblib")
                )
            else:
                st.error(
                    "XGBoost feature list file is missing. "
                    "Use your project's saved XGBoost feature list."
                )
                st.stop()

            X = data.reindex(columns=features, fill_value=0)
            probabilities = model.predict_proba(X)[:, 1]

            results = data.copy()
            results["Sepsis Risk Probability"] = probabilities
            results["Predicted Sepsis"] = (
                probabilities >= 0.13
            ).astype(int)

            st.markdown("### Prediction Results")
            st.dataframe(results, use_container_width=True)

            st.markdown("### Individual Prediction Explanation")
            sample_index = st.number_input(
                "Select row to explain",
                min_value=0,
                max_value=len(X) - 1,
                value=0,
                step=1
            )

            explainer = shap.TreeExplainer(model)
            shap_values = np.asarray(
                explainer.shap_values(
                    X.iloc[[int(sample_index)]]
                )
            ).reshape(-1)

            explanation = pd.DataFrame({
                "Feature": features,
                "SHAP value": shap_values
            })
            explanation["Absolute impact"] = (
                explanation["SHAP value"].abs()
            )
            explanation = explanation.nlargest(
                10, "Absolute impact"
            ).sort_values("SHAP value")

            st.caption(
                "Positive SHAP values push the model output higher; "
                "negative values push it lower. These are model "
                "contributions, not causal effects."
            )
            st.bar_chart(
                explanation.set_index("Feature")["SHAP value"],
                horizontal=True,
                height=450
            )

        else:
            # Load the saved LSTM model and preprocessing objects.
            lstm_model = load_lstm_model()
            lstm_features = joblib.load(
                os.path.join(MODEL_DIR, "lstm_features.joblib")
            )
            lstm_imputer = joblib.load(
                os.path.join(MODEL_DIR, "lstm_imputer.joblib")
            )
            lstm_scaler = joblib.load(
                os.path.join(MODEL_DIR, "lstm_scaler.joblib")
            )

            X_raw = data.reindex(
                columns=lstm_features, fill_value=np.nan
            )
            X_imputed = lstm_imputer.transform(X_raw)
            X_scaled = lstm_scaler.transform(X_imputed)

            if len(X_scaled) < 24:
                st.warning(
                    "LSTM requires at least 24 hourly records."
                )
            else:
                windows = np.lib.stride_tricks.sliding_window_view(
                    X_scaled, window_shape=24, axis=0
                ).transpose(0, 2, 1)

                probabilities = lstm_model.predict(
                    windows, verbose=0
                ).reshape(-1)

                results = data.iloc[23:].copy().reset_index(drop=True)
                results["Sepsis Risk Probability"] = probabilities
                results["Predicted Sepsis"] = (
                    probabilities >= 0.50
                ).astype(int)

                st.markdown("### Prediction Results")
                st.dataframe(results, use_container_width=True)

                st.markdown("### Individual Prediction Explanation")
                sample_index = st.number_input(
                    "Select prediction row to explain",
                    min_value=0,
                    max_value=len(windows) - 1,
                    value=0,
                    step=1,
                    key="lstm_sample_index"
                )

                sample_window = windows[int(sample_index)]
                base_probability = float(
                    probabilities[int(sample_index)]
                )

                impacts = []
                for feature_index, feature_name in enumerate(
                    lstm_features
                ):
                    masked_window = sample_window.copy()
                    masked_window[:, feature_index] = 0

                    changed_probability = float(
                        lstm_model.predict(
                            masked_window[np.newaxis, ...],
                            verbose=0
                        ).reshape(-1)[0]
                    )

                    impacts.append({
                        "Feature": feature_name,
                        "Prediction change": (
                            changed_probability - base_probability
                        ),
                        "Absolute impact": abs(
                            changed_probability - base_probability
                        )
                    })

                # Keep the 10 features with the largest estimated impact.
                impact_df = pd.DataFrame(impacts).nlargest(
                    10, "Absolute impact"
                ).sort_values("Prediction change")

                st.caption(
                    "Perturbation-based sensitivity estimate, not SHAP "
                    "or evidence of causation."
                )
                st.bar_chart(
                    impact_df.set_index("Feature")[
                        "Prediction change"
                    ],
                    horizontal=True,
                    height=450
                )

    else:
        st.info("Upload a CSV file to generate predictions.")




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

