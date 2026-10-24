# Early Sepsis Prediction Using Machine Learning

**Learning from ICU time-series data to investigate early sepsis risk prediction.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-LSTM-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Deployed%20With-Streamlit-red?logo=streamlit&logoColor=white)](https://streamlit.io/)

**[Live Demo](https://early-sepsis-ai-fscmkkxxngu9nqd5driffc.streamlit.app/) · [GitHub Repository](https://github.com/pranjali-debug/early-sepsis-ai)**

---

## The Idea

Sepsis can progress rapidly, making early risk assessment an important research problem. This project investigates whether patterns in routinely collected ICU measurements can help identify patients at risk.

Using the PhysioNet 2019 dataset, the project compares a tree-based machine learning model with a sequential deep learning model and presents their results through an interactive web application.

> **Research question:** How do XGBoost and LSTM compare when predicting sepsis from highly imbalanced, time-dependent ICU data?

## Dataset at a Glance

**Source:** [PhysioNet — Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)

| Dataset characteristic | Value |
|---|---:|
| Unique patients | 40,336 |
| Hourly records | 1,552,210 |
| Positive sepsis-labelled records | 27,916 |
| Positive record rate | 1.80% |

The low positive-label rate makes class imbalance a central challenge. The raw dataset is not included in this repository; obtain it from the official source and follow its terms.

## Methodology

1. **Data processing** — Process patient records in batches and validate the resulting dataset.
2. **Feature engineering** — Derive clinical and temporal features from hourly measurements.
3. **Patient-level splitting** — Keep patients separate across training, validation, and test sets to reduce patient-level data leakage.
4. **Model development** — Compare XGBoost with an LSTM that processes sequences of 24 time steps and 25 selected features.
5. **Threshold selection** — Tune the XGBoost decision threshold using validation data.
6. **Evaluation** — Compare performance on held-out test patients.
7. **Explainability** — Use SHAP to inspect global feature importance for XGBoost.

## Results

Held-out test performance from the recorded experiment:

| Model | PR-AUC | ROC-AUC | Recall |
|---|---:|---:|---:|
| XGBoost — baseline | 0.0909 | 0.8016 | 0.5756 |
| XGBoost — tuned threshold | 0.1003 | 0.8138 | **0.8845** |
| LSTM | **0.1067** | **0.8166** | 0.7991 |

### What the results tell us

- **LSTM achieved the highest PR-AUC and ROC-AUC** among the evaluated models.
- **Tuned XGBoost achieved the highest recall**, but its precision was only 0.0316.
- The results illustrate why model evaluation should consider class imbalance and the trade-off between missed positive cases and false alarms.

*Metrics are from the saved test evaluation results. The tuned XGBoost threshold was 0.13; the baseline and LSTM thresholds were 0.50.*

## Explore the Application

**[Launch the Early Sepsis Prediction App](https://early-sepsis-ai-fscmkkxxngu9nqd5driffc.streamlit.app/)**

The Streamlit application provides:

- Model performance comparison.
- Interactive risk prediction using saved models.
- Available model explainability visualizations.

## Repository Structure

```text
early-sepsis-ai/
├── Early_Sepsis_Cleaned.ipynb
├── app.py
├── requirements.txt
├── models/
├── results/
├── README.md
└── .gitignore
```

- `Early_Sepsis_Cleaned.ipynb` — Research workflow and evaluation.
- `app.py` — Streamlit application.
- `models/` — Saved models and required preprocessing artifacts.
- `results/` — Evaluation metrics and visualizations.
- `requirements.txt` — Python dependencies.

*The structure above describes the intended organization. Only files actually committed to the repository are available to users.*

## Run Locally

```bash
git clone https://github.com/pranjali-debug/early-sepsis-ai.git
cd early-sepsis-ai
python -m venv .venv
```

Activate the environment, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ensure the model and preprocessing files required by the application are available. Reproducing the full experiment additionally requires the dataset, compatible dependencies, and the relevant processing workflow.

## Limitations and Responsible Use

- The dataset is highly imbalanced, and the tuned XGBoost model produces many false positives.
- Performance on this dataset does not establish generalization to other hospitals or patient populations.
- SHAP feature importance describes model behaviour; it does not establish causation.
- The models have not been clinically validated.

**This project is for educational and research purposes only. It is not a diagnostic tool and must not be used to make medical decisions.**

## Dataset Reference

Reyna, M. A., et al. (2019). *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.*

[Dataset and access information](https://physionet.org/content/challenge-2019/1.0.0/) · [DOI: 10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857)

## Author

**Pranjali Vijay Kumar Sadafule**  
B.Tech, Computer Science Engineering — AI Specialization

[GitHub Profile](https://github.com/pranjali-debug)
