# Early Sepsis Prediction Using Machine Learning

**A machine learning project for estimating sepsis risk from hourly ICU patient data using XGBoost and LSTM.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red)](https://streamlit.io/)

**Live Demo:** [Open the Early Sepsis Prediction App](https://early-sepsis-ai-fscmkkxxngu9nqd5driffc.streamlit.app/)

**GitHub Repository:** [early-sepsis-ai](https://github.com/pranjali-debug/early-sepsis-ai)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Explainable AI](#explainable-ai)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation and Setup](#installation-and-setup)
- [How to Use](#how-to-use)
- [Reproducibility and Data Leakage Prevention](#reproducibility-and-data-leakage-prevention)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)
- [Dataset Citation](#dataset-citation)
- [Author](#author)

---

## Overview

Early Sepsis Prediction is a machine learning project that investigates whether patterns in routinely collected intensive care unit (ICU) patient measurements can help estimate the risk of sepsis.

The project uses the **PhysioNet/Computing in Cardiology Challenge 2019 dataset**, which contains hourly clinical measurements from ICU patients.

Two machine learning approaches are evaluated:

- **XGBoost:** A gradient-boosted decision tree model using engineered clinical features.
- **LSTM (Long Short-Term Memory):** A recurrent neural network designed to learn patterns from sequential patient measurements.

A Streamlit web application provides an interface for exploring model results, entering patient measurements, obtaining model-generated risk estimates, and viewing explainability information.

The project demonstrates an end-to-end machine learning workflow, from data processing and model evaluation to deployment of a demonstration application.

> **Important:** This is an educational and research prototype. It is not a clinically validated diagnostic system and must not be used to make medical decisions.

## Problem Statement

Sepsis is a serious condition associated with a potentially life-threatening response to infection. Identifying concerning changes early may help healthcare professionals assess patients more effectively.

ICU monitoring produces large amounts of time-dependent clinical data, including vital signs, laboratory measurements, and patient-related information.

The challenge is to investigate whether machine learning models can identify patterns associated with sepsis while accounting for missing measurements, imbalanced labels, and changes in patient condition over time.

This project compares a tree-based model with a sequential deep learning model and evaluates their performance on held-out patient data.

## Objectives

The main objectives are to:

1. Process the PhysioNet 2019 dataset efficiently.
2. Explore missing values and the distribution of sepsis labels.
3. Engineer features from hourly clinical measurements.
4. Prevent patient overlap between training, validation, and test sets.
5. Train and evaluate XGBoost and LSTM models.
6. Investigate decision thresholds for imbalanced classification.
7. Compare models using precision, recall, F1-score, ROC-AUC, and PR-AUC.
8. Explore model explainability using SHAP.
9. Develop a web interface for demonstrating saved models and their predictions.

## Dataset

### Source

The project uses the publicly available **PhysioNet/Computing in Cardiology Challenge 2019** dataset.

- **Official dataset:** https://physionet.org/content/challenge-2019/1.0.0/
- **Official data description:** https://archive.physionet.org/challenge/2019/
- **Dataset version:** 1.0.0
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0), subject to the dataset's terms.

The dataset contains records from 40,336 ICU patients across two training subsets, `training_setA` and `training_setB`. Each patient's file contains hourly clinical measurements and a `SepsisLabel` target. See the official dataset documentation for details.

### Dataset Statistics

The following statistics were obtained during this project's data processing and validation.

| Statistic | Value |
|---|---:|
| Unique patients | 40,336 |
| Hourly records | 1,552,210 |
| Columns in processed records | 42 |
| Positive sepsis-labelled hourly records | 27,916 |
| Positive hourly-record rate | Approximately 1.80% |
| Full processed Parquet batches | 404 |

The 42 processed columns include the patient identifier, clinical features, and target label. The model feature matrix uses 52 engineered features for XGBoost after feature engineering.

### Important Dataset Notes

- The dataset contains missing clinical measurements.
- Sepsis-positive hourly records represent a small fraction of the observations.
- Individual patient files contain time-series data rather than independent, unrelated samples.
- The dataset is intended for research and benchmarking, not direct clinical deployment.

**The raw dataset is not included in this GitHub repository.** Download it from the official PhysioNet source and follow its terms. This keeps the repository smaller and avoids unnecessary duplication of the source data.

## Project Workflow

The overall workflow is:

1. **Data acquisition:** Obtain the patient records from PhysioNet.
2. **Data processing:** Process patient files in batches to reduce memory usage.
3. **Data validation:** Check record counts, patient identifiers, labels, and processed files.
4. **Feature engineering:** Construct features from clinical measurements and their temporal patterns.
5. **Patient-level splitting:** Separate patients into training, validation, and test sets.
6. **Preprocessing:** Prepare the features for each model, including the LSTM's sequential inputs.
7. **Model training:** Train XGBoost and LSTM models.
8. **Threshold selection:** Evaluate candidate decision thresholds using validation data.
9. **Model evaluation:** Evaluate the saved models on held-out test patients.
10. **Explainability:** Examine global feature importance using SHAP for XGBoost.
11. **Deployment:** Use the saved models in a Streamlit demonstration application.

The existing processed files and saved artifacts allow the analysis and demo to be inspected without repeating the entire computationally expensive training workflow.

## Machine Learning Models

### 1. XGBoost

XGBoost is a gradient-boosting algorithm that builds an ensemble of decision trees.

In this project, it uses engineered features derived from patient measurements. The final feature matrix contains 52 features, excluding the patient identifier and target label.

Two configurations are evaluated:

- **XGBoost baseline:** Uses a decision threshold of 0.50.
- **XGBoost tuned:** Uses a decision threshold of 0.13, selected during validation to improve recall.

The lower threshold makes the tuned model more likely to identify positive cases, but it also produces more false-positive predictions.

### 2. LSTM

LSTM is a recurrent neural network architecture designed to learn dependencies in sequential data.

For this project, the LSTM receives sequences containing 24 time steps and 25 selected features per time step.

The LSTM pipeline uses saved preprocessing objects for imputation and scaling. These objects were fitted using training data rather than the held-out test data.

The LSTM is evaluated alongside XGBoost to investigate whether sequential modelling offers benefits for this time-dependent prediction problem.

## Results

The following results were recorded on the held-out test set.

| Model | Threshold | PR-AUC | ROC-AUC | Recall | Precision | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost baseline | 0.50 | 0.0909 | 0.8016 | 0.5756 | 0.0640 | 0.1152 |
| XGBoost tuned | 0.13 | 0.1003 | 0.8138 | 0.8845 | 0.0316 | 0.0610 |
| LSTM | 0.50 | 0.1067 | 0.8166 | 0.7991 | 0.0562 | 0.1050 |

*Metrics are rounded for readability. Results correspond to the saved evaluation artifacts from this experiment.*

### Interpretation

- **LSTM achieved the highest PR-AUC** among the three evaluated configurations, at approximately 0.1067.
- **LSTM achieved the highest ROC-AUC**, at approximately 0.8166.
- **Tuned XGBoost achieved the highest recall**, at approximately 0.8845.
- The tuned XGBoost model's precision is low, illustrating the trade-off between identifying more positive cases and generating false alarms.
- The baseline XGBoost model has a higher F1-score than the tuned model in this particular test evaluation.

Because the dataset is highly imbalanced, PR-AUC, recall, precision, and the number of false positives are important alongside ROC-AUC.

A high recall value alone does not mean that a model is sufficiently accurate for clinical use. These results are experimental and require further validation.

## Explainable AI

The project uses **SHAP (SHapley Additive exPlanations)** to examine global feature importance for the XGBoost model.

SHAP helps investigate which features contribute to the model's predictions across the evaluated data.

The saved global feature-importance visualization is available in the project's results artifacts.

The explainability analysis is intended to support model inspection and understanding. Feature importance does not establish causation, clinical relevance, or the reason for an individual patient's medical condition.

## Web Application

The project includes a Streamlit web application for demonstrating the saved models.

**Launch the application:** https://early-sepsis-ai-fscmkkxxngu9nqd5driffc.streamlit.app/

The application includes the following areas:

- **Overview:** Introduces the project and its purpose.
- **Model Comparison:** Displays model performance for comparison.
- **Risk Prediction:** Accepts input measurements and displays model-generated risk estimates.
- **Explainability:** Presents available model-interpretation information.

The website uses saved model and preprocessing artifacts rather than requiring the full dataset to be uploaded through the interface.

The deployed application is a demonstration tool. Its predictions should not be interpreted as medical diagnoses.

## Project Structure

The intended repository structure is shown below. Exact filenames may differ depending on which artifacts are included in the published repository.

```text
early-sepsis-ai/
│
├── Early_Sepsis_Cleaned.ipynb
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── xgboost_sepsis.json
│   ├── xgboost_tuned_candidate.json
│   ├── lstm_sepsis.keras
│   ├── lstm_features.joblib
│   ├── lstm_imputer.joblib
│   └── lstm_scaler.joblib
│
└── results/
    ├── final_model_comparison.csv
    ├── xgboost_test_metrics.csv
    ├── lstm_test_metrics.csv
    └── shap_feature_importance.png
```

**Note:** This is a reference structure, not a claim that every listed file is already committed to GitHub. The application must be supplied with every artifact it actually loads. Large files can be hosted separately when appropriate.

The full raw dataset, temporary checkpoints, runtime logs, and unnecessary generated files do not need to be committed.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Tabular data processing |
| NumPy | Numerical computation |
| Scikit-learn | Preprocessing and evaluation utilities |
| XGBoost | Gradient-boosted tree model |
| TensorFlow / Keras | LSTM implementation |
| Joblib | Saving and loading preprocessing objects |
| SHAP | Model explainability |
| Matplotlib | Result visualizations |
| Jupyter Notebook / Google Colab | Development and experimentation |
| Streamlit | Interactive web application |
| Git and GitHub | Version control and project hosting |

## Installation and Setup

### Prerequisites

- Python compatible with the project's dependency versions.
- Git.
- The project source files.
- Required saved model and preprocessing artifacts for running the application.

### 1. Clone the Repository

```bash
git clone https://github.com/pranjali-debug/early-sepsis-ai.git
cd early-sepsis-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain the compatible versions of the libraries actually used by the application and notebook.

### 4. Verify Model Artifacts

Ensure that the saved models and preprocessing files expected by `app.py` are available at the paths configured in the application.

If any required artifacts are hosted separately, download them and place them in the appropriate directories before launching the application.

### 5. Run the Web Application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that address in your browser.

### Running the Notebook

Open `Early_Sepsis_Cleaned.ipynb` in Jupyter Notebook or Google Colab.

The notebook documents the data-processing and modelling workflow. Some stages require the original dataset and previously generated processed files. Expensive training and data-processing stages should only be run when intentionally reproducing the experiment with the necessary resources.

## How to Use

1. Open the [live application](https://early-sepsis-ai-fscmkkxxngu9nqd5driffc.streamlit.app/).
2. Read the overview and project limitations.
3. Visit the model comparison section to inspect the recorded evaluation metrics.
4. Open the risk-prediction section and enter supported example measurements.
5. Review the displayed output and available explainability information.
6. Use the notebook and results files to understand how the models were developed and evaluated.

Use only appropriate demonstration inputs. Do not enter identifiable patient information or rely on the output to make healthcare decisions.

## Reproducibility and Data Leakage Prevention

Several design choices are important for reliable evaluation.

### Patient-Level Data Splitting

Patients are assigned to separate training, validation, and test groups. This reduces the risk of the same patient's records appearing in multiple sets.

The recorded split contains:

| Split | Unique patients |
|---|---:|
| Training | 28,235 |
| Validation | 6,050 |
| Test | 6,051 |
| **Total** | **40,336** |

### Separation of Model Development and Testing

- Training data is used for fitting the model and preprocessing steps.
- Validation data is used for model selection and decision-threshold tuning.
- Test data is reserved for the final reported evaluation.

### Preprocessing

The saved LSTM imputer and scaler are fitted on training data. Applying the same fitted transformations to validation and test data helps prevent information from those sets from leaking into preprocessing.

### Saved Artifacts

The project retains saved model and evaluation artifacts so that predictions and reported results can be inspected without automatically repeating model training.

Reproducing the experiment requires the appropriate data, dependencies, saved split definitions, and compatible code.

## Limitations

This project has several important limitations:

- **Class imbalance:** Positive sepsis-labelled hourly records make up approximately 1.80% of all processed hourly records.
- **False positives:** The tuned XGBoost configuration has low precision at its selected threshold.
- **Temporal modelling:** LSTM performance depends on sequence construction, missing-data handling, and the availability of appropriate historical measurements.
- **Dataset scope:** Results on the PhysioNet 2019 dataset may not generalize to different hospitals, patient populations, or clinical workflows.
- **Evaluation metric:** PR-AUC, ROC-AUC, recall, precision, and F1-score do not by themselves establish clinical usefulness. The official challenge also defines a utility-based evaluation for early sepsis prediction.
- **Threshold selection:** Changing the decision threshold changes the balance between false positives and false negatives.
- **Explainability:** SHAP feature importance does not demonstrate causality.
- **Deployment validation:** The web application has been tested as a demonstration, but this does not constitute clinical validation or a formal medical-device assessment.

## Future Improvements

Potential future work includes:

1. Evaluating additional threshold settings and reporting confusion matrices.
2. Comparing models using the official PhysioNet challenge utility metric.
3. Performing external validation on an appropriate independent dataset.
4. Investigating calibration and uncertainty estimates.
5. Improving the LSTM training and sequence-generation pipeline.
6. Adding more detailed evaluation visualizations.
7. Improving automated testing and documenting exact dependency versions.
8. Evaluating robustness across different patient groups and missing-data patterns.
9. Conducting appropriate clinical and ethical review before any real-world use.

These are proposed improvements and should not be interpreted as already completed.

## Disclaimer

This project is developed for educational and research purposes only.

It is **not a medical device, is not clinically validated, and is not intended to diagnose sepsis, recommend treatment, replace healthcare professionals, or support real-world clinical decisions**.

Predictions may be incorrect, and model performance may differ substantially from the reported experimental results when applied to other data or settings.

## Dataset Citation

The project uses the following dataset:

Reyna, M. A., Josef, C., Jeter, R., Shashikumar, S., Moody, B., Westover, M. B., Sharma, A., Nemati, S., & Clifford, G. D. (2019). *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019* (version 1.0.0). PhysioNet. https://doi.org/10.13026/v64v-d857

Official dataset page: https://physionet.org/content/challenge-2019/1.0.0/

Please consult the official source for the full license terms and citation requirements.

## Author

**Pranjali Vijay Kumar Sadafule**

B.Tech Computer Science Engineering — AI Specialization

GitHub: https://github.com/pranjali-debug

---

If you find this project useful for learning about machine learning, time-series data, model evaluation, or explainable AI, feel free to explore the repository and the accompanying notebook.

**Project:** Early Sepsis Prediction Using Machine Learning  
**Application:** https://early-sepsis-ai-fscmkkxxngu9nqd5driffc.streamlit.app/
