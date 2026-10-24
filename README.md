# Early Sepsis Prediction Using Machine Learning

A machine learning project that compares **XGBoost and LSTM** models to estimate sepsis risk from hourly ICU patient data. Includes a Streamlit web application for interactive predictions and model comparison.

**Live Demo:** [Early Sepsis Prediction App](https://early-sepsis-ai-fscmkkxxngu9nqd5driffc.streamlit.app/)

**GitHub:** [early-sepsis-ai](https://github.com/pranjali-debug/early-sepsis-ai)

## Overview

Sepsis is a potentially life-threatening condition where early risk assessment is important. This project explores machine learning techniques for identifying patterns associated with sepsis using clinical time-series data.

### Objectives
- Process and analyze ICU patient data.
- Engineer features and prevent patient-level data leakage.
- Train and compare XGBoost and LSTM models.
- Evaluate model performance on held-out patient data.
- Explore feature importance using SHAP.
- Deploy a demonstration application using Streamlit.

## Dataset

**Source:** [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)

| Statistic | Value |
|---|---:|
| Patients | 40,336 |
| Hourly records | 1,552,210 |
| Positive sepsis-labelled records | 27,916 |
| Positive record rate | 1.80% |

The raw dataset is not included in this repository. Please obtain it from the official source and follow its license terms.

## Models and Results

Evaluation on the held-out test set:

| Model | PR-AUC | ROC-AUC | Recall |
|---|---:|---:|---:|
| XGBoost Baseline | 0.0909 | 0.8016 | 0.5756 |
| XGBoost Tuned | 0.1003 | 0.8138 | 0.8845 |
| LSTM | **0.1067** | **0.8166** | 0.7991 |

The LSTM achieved the highest PR-AUC and ROC-AUC in this experiment. Tuned XGBoost achieved the highest recall, with lower precision. Results reflect a highly imbalanced dataset.

## Web Application

The Streamlit application provides:
- Project overview and model comparison.
- Interactive risk prediction using saved models.
- Available model explainability information.

[**Open the Live Application**](https://early-sepsis-ai-fscmkkxxngu9nqd5driffc.streamlit.app/)

## Project Structure

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

The `models/` directory contains the required saved models and preprocessing artifacts. The `results/` directory contains evaluation metrics and visualizations. Exact files depend on the artifacts included in the repository.

## Technologies Used

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- TensorFlow/Keras
- SHAP, Matplotlib
- Streamlit, Git, GitHub

## Run Locally

```bash
git clone https://github.com/pranjali-debug/early-sepsis-ai.git
cd early-sepsis-ai
python -m venv .venv
```

Activate the virtual environment, then install dependencies and launch the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ensure the saved model and preprocessing files required by the application are available before launching it.

## Limitations

- The dataset is highly imbalanced.
- The tuned XGBoost model has low precision despite high recall.
- Results may not generalize to other hospitals or patient populations.
- The models and application have not been clinically validated.

## Disclaimer

**This project is for educational and research purposes only.** It is not a medical device and must not be used to diagnose sepsis or make clinical decisions.

## Dataset Citation

Reyna, M. A., et al. (2019). *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.* PhysioNet.

https://doi.org/10.13026/v64v-d857

## Author

**Pranjali Vijay Kumar Sadafule**

B.Tech Computer Science Engineering — AI Specialization

[GitHub Profile](https://github.com/pranjali-debug)
