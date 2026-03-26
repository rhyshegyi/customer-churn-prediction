# Customer Churn Prediction

Predict whether a telecom customer will churn using machine learning. This project demonstrates an end-to-end ML workflow — from exploratory analysis to production-ready inference with validated input/output schemas.

## Problem Statement

Customer churn costs telecom companies billions annually. Predicting which customers are likely to leave enables targeted retention campaigns. This project builds a classification model to identify at-risk customers and wraps inference in a robust validation layer using Pydantic.

## Dataset

**Telco Customer Churn** — 7,043 customers with 21 features.

| Feature Group | Examples |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod |
| Services | PhoneService, InternetService, OnlineSecurity, TechSupport |
| Charges | MonthlyCharges, TotalCharges |
| Target | Churn (Yes/No) |

**Source:** [Kaggle / IBM](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Approach

1. **EDA** — Explored feature distributions, class imbalance (26% churn), and key predictors
2. **Preprocessing** — `ColumnTransformer` with `StandardScaler` (numeric) and `OneHotEncoder` (categorical)
3. **Model Benchmarking** — Compared Logistic Regression, Random Forest, and Gradient Boosting using 5-fold cross-validated ROC-AUC
4. **Hyperparameter Tuning** — `GridSearchCV` on the best classifier
5. **Pydantic Validation** — Input/output schemas with type checking, field constraints, and custom validators

## Results

| Model | CV ROC-AUC |
|---|---|
| Logistic Regression | ~0.84 |
| Random Forest | ~0.83 |
| **Gradient Boosting** | **~0.85** |

*Key predictors: Contract type, tenure, MonthlyCharges, InternetService, TotalCharges*

## Project Structure

```
├── data/                ← dataset (auto-downloaded)
├── notebooks/
│   └── 01_eda.ipynb     ← exploratory data analysis
├── src/
│   ├── schemas.py       ← Pydantic input/output models
│   ├── train.py         ← training pipeline
│   ├── predict.py       ← validated inference
│   └── evaluate.py      ← metrics & plots
├── tests/
│   └── test_predict.py  ← validation tests
├── models/              ← saved model artifacts
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python src/train.py

# 4. Evaluate
python src/evaluate.py

# 5. Run inference tests
python tests/test_predict.py
```

## Technologies

Python · Pandas · Scikit-Learn · Pydantic · Matplotlib · Seaborn · Jupyter
