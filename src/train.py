"""
train.py — Training pipeline for Customer Churn prediction.

Loads the Telco Customer Churn dataset, builds sklearn pipelines,
benchmarks multiple classifiers, tunes the best one, and saves
the final model.

Usage:
    python src/train.py
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import warnings
warnings.filterwarnings("ignore")


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset."""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv")

    if not os.path.exists(data_path):
        print("Downloading Telco Customer Churn dataset from OpenML...")
        from sklearn.datasets import fetch_openml
        dataset = fetch_openml(data_id=42178, as_frame=True, parser="auto")
        df = dataset.frame
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"  Saved to {data_path}")
    else:
        df = pd.read_csv(data_path)

    return df


def preprocess(df: pd.DataFrame):
    """Clean data and split into X, y."""
    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target column
    target_col = "Churn"
    if target_col not in df.columns:
        # OpenML may name it differently
        for col in df.columns:
            if "churn" in col.lower():
                target_col = col
                break

    # Encode target
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # Fix TotalCharges (some rows have blank strings)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    return X, y


def build_pipeline(X: pd.DataFrame):
    """Build a ColumnTransformer + classifier pipeline."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def main():
    print("=" * 60)
    print("  Customer Churn Prediction — Training Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_data()
    print(f"\nDataset shape: {df.shape}")

    # 2. Preprocess
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Churn rate: {y.mean():.1%}")

    # 3. Build pipeline
    preprocessor, numeric_cols, categorical_cols = build_pipeline(X_train)
    print(f"\nNumeric features ({len(numeric_cols)}):     {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # 4. Benchmark models
    print("\n" + "-" * 60)
    print("Model Benchmarking (5-fold CV, ROC-AUC)")
    print("-" * 60)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
        results[name] = scores.mean()
        print(f"  {name:<25s}  AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(results, key=results.get)
    print(f"\n  Best model: {best_name} (AUC: {results[best_name]:.4f})")

    # 5. Hyperparameter tuning on best model
    print("\n" + "-" * 60)
    print(f"Hyperparameter Tuning — {best_name}")
    print("-" * 60)

    if "Gradient Boosting" in best_name:
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        }
        base_clf = GradientBoostingClassifier(random_state=42)
    elif "Random Forest" in best_name:
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20, None],
        }
        base_clf = RandomForestClassifier(random_state=42)
    else:
        param_grid = {
            "classifier__C": [0.1, 1, 10],
        }
        base_clf = LogisticRegression(max_iter=1000, random_state=42)

    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", base_clf)])
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV AUC: {grid.best_score_:.4f}")

    # 6. Evaluate on test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)

    print(f"\n  Test ROC-AUC: {test_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    # 7. Save model + test data
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "churn_pipeline.joblib")
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    # Save test data for evaluate.py
    test_path = os.path.join(models_dir, "test_data.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_path)
    print(f"Test data saved to {test_path}")

    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
