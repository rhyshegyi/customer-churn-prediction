"""
predict.py — ML inference with Pydantic validation.

Loads the trained churn pipeline and wraps prediction calls
with Pydantic input/output validation.

Usage:
    python src/predict.py
"""
import os
import json
import pandas as pd
import joblib
from pydantic import ValidationError

# Use absolute import-safe path resolution
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

# Add project root to path for imports
import sys
sys.path.insert(0, _SRC_DIR)
from schemas import CustomerInput, ChurnPrediction

# Load model once at module level
MODEL_PATH = os.path.join(_PROJECT_DIR, "models", "churn_pipeline.joblib")
model = joblib.load(MODEL_PATH)


def predict_churn(payload: dict) -> dict:
    """Validate input, run inference, and return validated output as JSON.

    Args:
        payload: Dictionary with customer feature values.

    Returns:
        Dictionary from ChurnPrediction.model_dump().

    Raises:
        pydantic.ValidationError: If input fails validation.
    """
    # 1. Validate input
    customer = CustomerInput(**payload)

    # 2. Convert to DataFrame
    row = customer.model_dump()
    df = pd.DataFrame([row])

    # 3. Inference
    pred_label = model.predict(df)[0]
    pred_proba = model.predict_proba(df)[0]

    prediction_str = "Churn" if pred_label == 1 else "No Churn"
    churn_prob = float(pred_proba[1])

    # 4. Validate & serialize output
    result = ChurnPrediction(
        prediction=prediction_str,
        churn_probability=round(churn_prob, 4),
    )

    return result.model_dump()


# ── Demo ──
if __name__ == "__main__":
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 358.2,
    }

    print("=== Valid Prediction ===")
    result = predict_churn(sample)
    print(json.dumps(result, indent=2))

    print("\n=== Invalid Input (tenure = -5) ===")
    bad = sample.copy()
    bad["tenure"] = -5
    try:
        predict_churn(bad)
    except ValidationError as e:
        print(e)
