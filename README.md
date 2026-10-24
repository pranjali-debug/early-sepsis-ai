# Early Sepsis Prediction | Machine Learning & Clinical Time-Series Analysis

<p align="center">
  <strong>From hourly ICU observations to data-driven risk estimation.</strong>
</p>

<p align="center">
  <a href="https://early-sepsis-ai-fscmkkxxngu9nqd5driffc.streamlit.app/">Live Demo</a> •
  <a href="https://github.com/pranjali-debug/early-sepsis-ai">GitHub Repository</a> •
  <a href="https://physionet.org/content/challenge-2019/1.0.0/">Dataset</a>
</p>

An end-to-end machine learning research project investigating sepsis risk estimation from longitudinal ICU data. The project compares **XGBoost and Long Short-Term Memory (LSTM)** models, evaluates their performance under severe class imbalance, and explores model explainability through SHAP.

---

## Research at a Glance

| Component | Description |
|---|---|
| Research domain | Healthcare AI and predictive modelling |
| Dataset | PhysioNet/CinC Challenge 2019 |
| Learning task | Binary classification of hourly sepsis labels |
| Models | XGBoost and LSTM |
| Evaluation | PR-AUC, ROC-AUC, precision, recall and F1-score |
| Explainability | SHAP for global XGBoost feature importance |
| Deployment | Interactive Streamlit application |

## 01 · Research Motivation

Sepsis is a potentially life-threatening condition in which timely recognition is important. ICU data contains evolving physiological measurements and laboratory observations that may provide useful signals for estimating risk.

However, this is not a straightforward classification problem. Clinical measurements are frequently missing, positive labels are uncommon, and observations from the same patient are temporally dependent.

**Research question:** How do a tree-based model using engineered features and a recurrent neural network using sequential measurements compare when estimating sepsis risk from ICU data?

The project investigates this question through a reproducible modelling and evaluation workflow.

## 02 · Dataset

**Source:** [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)

| Dataset characteristic | Recorded value |
|---|---:|
| Unique patients | 40,336 |
| Hourly records | 1,552,210 |
| Positive sepsis-labelled records | 27,916 |
| Positive record prevalence | 1.80% |
| Processed Parquet batches | 404 |

The processed data contains hourly clinical measurements, patient identifiers and the `SepsisLabel` target.

The dataset is highly imbalanced: approximately 98.2% of hourly records have a negative label. Consequently, accuracy alone would be inadequate for evaluating model performance.

The raw dataset is not included in this repository. Obtain it from the official source and comply with its licensing and attribution requirements.

## 03 · Methodology

The project follows these stages:

1. **Data engineering:** Process patient files in batches and validate the resulting dataset.
2. **Feature engineering:** Construct clinical and temporal features for model input.
3. **Patient-level partitioning:** Separate patients into training, validation and test sets.
4. **Preprocessing:** Apply model-specific transformations, including imputation and scaling for the LSTM pipeline.
5. **Model development:** Train and compare XGBoost and LSTM configurations.
6. **Threshold selection:** Use validation data to investigate the trade-off between recall and precision.
7. **Held-out evaluation:** Compare saved model predictions on the test patients.
8. **Interpretability and deployment:** Examine global XGBoost feature importance and demonstrate predictions through Streamlit.

### Data leakage prevention

The recorded split contains 28,235 training patients, 6,050 validation patients and 6,051 test patients, totalling 40,336.

Patient-level separation is used to reduce the risk of the same patient's observations appearing across multiple splits. Preprocessing is fitted on training data, and validation data is used for model selection and threshold tuning rather than final reporting.

## 04 · Experimental Results

The following metrics were recorded on the held-out test set.

| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost Baseline | 0.50 | 0.0909 | 0.8016 | 0.0640 | 0.5756 | 0.1152 |
| XGBoost Tuned | 0.13 | 0.1003 | 0.8138 | 0.0316 | **0.8845** | 0.0610 |
| LSTM | 0.50 | **0.1067** | **0.8166** | 0.0562 | 0.7991 | 0.1050 |

*Values are rounded from the saved experimental results.*

### Interpretation

- **LSTM achieved the highest PR-AUC and ROC-AUC** among the three evaluated configurations.
- **Tuned XGBoost achieved the highest recall**, but its precision fell to approximately 3.16%.
- The baseline XGBoost model achieved the highest F1-score in this comparison.
- The results illustrate that increasing recall does not necessarily improve precision or overall model utility.

These findings are specific to the evaluated dataset, model configurations and thresholds. They do not establish that one model is clinically superior.

## 05 · Explainable AI

The project uses **SHAP (SHapley Additive exPlanations)** to inspect global feature importance for XGBoost.

This analysis helps investigate which input features contribute to model predictions across the evaluated observations. It supports model inspection but does not establish causality or clinical significance.

## 06 · Interactive Demonstration

Explore the deployed application:

**[Launch Early Sepsis Prediction](https://early-sepsis-ai-fscmkkxxngu9nqd5driffc.streamlit.app/)**

The application includes:

- Project overview and model comparison
- Risk prediction using saved models
- Available explainability visualizations

The application demonstrates model outputs; it is not a clinical decision-support system.

## 07 · Repository Structure

```text
early-sepsis-ai/
├── Early_Sepsis_Cleaned.ipynb
├── app.py
├── requirements.txt
├── README.md
├── models/
├── results/
└── .gitignore
```

- `Early_Sepsis_Cleaned.ipynb` — modelling methodology and experimental analysis.
- `app.py` — Streamlit interface.
- `models/` — saved models and required preprocessing artifacts.
- `results/` — evaluation metrics and visualizations.
- `requirements.txt` — Python dependencies.

This is the intended high-level structure; verify the actual files committed to the repository before publishing.

## 08 · Run Locally

**Requirements:** Python and the required model artifacts.

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

Ensure the saved models and preprocessing files expected by the application are available. Reproducing the full experiment additionally requires the dataset and the appropriate processing workflow.

## 09 · Limitations & Research Scope

- Severe class imbalance affects precision and threshold selection.
- Model performance may change across hospitals and patient populations.
- The reported metrics do not by themselves establish clinical usefulness.
- Further calibration, external validation and evaluation using the official challenge utility metric would strengthen the analysis.
- Explainability results should not be interpreted as evidence of causation.

### Medical disclaimer

This project is intended for **educational and research purposes only**. It has not been clinically validated and must not be used to diagnose sepsis, recommend treatment or make patient-care decisions.

## 10 · Dataset Reference

Reyna, M. A., et al. (2019). *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.*

**Dataset:** https://doi.org/10.13026/v64v-d857

Please refer to the official dataset documentation for the complete citation and applicable license terms.

---

## Author

**Pranjali Vijay Kumar Sadafule**  
B.Tech Computer Science Engineering · AI Specialization

[GitHub Profile](https://github.com/pranjali-debug)

*Building interpretable, data-driven machine learning systems for real-world problems.*
