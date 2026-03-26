"""
test_predict.py — Tests for the Pydantic churn prediction validation layer.

Verifies that valid inputs return predictions and invalid inputs
raise clear Pydantic ValidationErrors.

Usage:
    python tests/test_predict.py
"""
import sys
import os
import json

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import ValidationError
from predict import predict_churn

# ═══════════════════════════════════════════════════════════════════
# Test data
# ═══════════════════════════════════════════════════════════════════
VALID_CUSTOMER = {
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

passed = 0
failed = 0


def run_test(name: str, fn):
    global passed, failed
    print(f"\n{'─' * 60}")
    print(f"TEST: {name}")
    print(f"{'─' * 60}")
    try:
        fn()
        passed += 1
        print("  ✅ PASSED")
    except AssertionError as e:
        failed += 1
        print(f"  ❌ FAILED: {e}")


# ═══════════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════════
def test_valid_prediction():
    """A well-formed input should return a valid prediction dict."""
    result = predict_churn(VALID_CUSTOMER)
    print(f"  Result: {json.dumps(result, indent=4)}")
    assert "prediction" in result
    assert result["prediction"] in ("Churn", "No Churn")
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert "model_version" in result


def test_invalid_type():
    """Passing a bad type should raise a ValidationError."""
    bad = VALID_CUSTOMER.copy()
    bad["MonthlyCharges"] = "not-a-number"
    try:
        predict_churn(bad)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        print(f"  Caught expected error ({e.error_count()} error(s)):")
        for err in e.errors():
            print(f"    → {err['loc']}: {err['msg']}")


def test_out_of_range():
    """Tenure of -5 should fail the ge=0 constraint."""
    bad = VALID_CUSTOMER.copy()
    bad["tenure"] = -5
    try:
        predict_churn(bad)
        assert False, "Should have raised ValidationError for tenure=-5"
    except ValidationError as e:
        print(f"  Caught expected error ({e.error_count()} error(s)):")
        for err in e.errors():
            print(f"    → {err['loc']}: {err['msg']}")
        assert any("tenure" in str(err["loc"]) for err in e.errors())


def test_invalid_literal():
    """An invalid contract type should fail Literal validation."""
    bad = VALID_CUSTOMER.copy()
    bad["Contract"] = "Weekly"
    try:
        predict_churn(bad)
        assert False, "Should have raised ValidationError for Contract='Weekly'"
    except ValidationError as e:
        print(f"  Caught expected error ({e.error_count()} error(s)):")
        for err in e.errors():
            print(f"    → {err['loc']}: {err['msg']}")
        assert any("Contract" in str(err["loc"]) for err in e.errors())


def test_missing_field():
    """Omitting required fields should raise a ValidationError."""
    bad = VALID_CUSTOMER.copy()
    del bad["MonthlyCharges"]
    del bad["tenure"]
    try:
        predict_churn(bad)
        assert False, "Should have raised ValidationError for missing fields"
    except ValidationError as e:
        print(f"  Caught expected error ({e.error_count()} error(s)):")
        for err in e.errors():
            print(f"    → {err['loc']}: {err['msg']}")


def test_custom_validator():
    """TotalCharges too low for tenure should be rejected."""
    bad = VALID_CUSTOMER.copy()
    bad["tenure"] = 24
    bad["MonthlyCharges"] = 100.0
    bad["TotalCharges"] = 10.0  # way too low
    try:
        predict_churn(bad)
        assert False, "Should have raised ValidationError for implausible TotalCharges"
    except ValidationError as e:
        print(f"  Caught expected error ({e.error_count()} error(s)):")
        for err in e.errors():
            print(f"    → {err['loc']}: {err['msg']}")
        assert any("TotalCharges" in str(err["loc"]) for err in e.errors())


def test_serialisation():
    """Output should be JSON-serialisable."""
    result = predict_churn(VALID_CUSTOMER)
    json_str = json.dumps(result)
    print(f"  JSON: {json_str}")
    assert json.loads(json_str) == result


# ═══════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_test("Valid prediction", test_valid_prediction)
    run_test("Invalid type", test_invalid_type)
    run_test("Out-of-range (tenure=-5)", test_out_of_range)
    run_test("Invalid literal (Contract='Weekly')", test_invalid_literal)
    run_test("Missing required fields", test_missing_field)
    run_test("Custom validator (TotalCharges too low)", test_custom_validator)
    run_test("JSON serialisation round-trip", test_serialisation)

    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'═' * 60}")

    sys.exit(1 if failed > 0 else 0)
