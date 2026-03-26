"""
evaluate.py — Model evaluation and reporting.

Loads the trained model and test data, prints metrics,
and generates diagnostic plots.

Usage:
    python src/evaluate.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_theme(style="whitegrid")


def main():
    print("=" * 60)
    print("  Customer Churn — Model Evaluation")
    print("=" * 60)

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    # Load model and test data
    model = joblib.load(os.path.join(models_dir, "churn_pipeline.joblib"))
    test_data = joblib.load(os.path.join(models_dir, "test_data.joblib"))
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)

    # Classification report
    print(f"\nTest ROC-AUC: {test_auc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ── Confusion Matrix ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No Churn", "Churn"],
        cmap="Blues", ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix")

    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, label=f"AUC = {test_auc:.3f}", linewidth=2)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "evaluation_plots.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plots saved to {os.path.join(models_dir, 'evaluation_plots.png')}")

    # ── Feature Importance ──
    clf = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    if hasattr(clf, "feature_importances_"):
        # Get feature names after preprocessing
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(clf.feature_importances_))]

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        fig, ax = plt.subplots(figsize=(10, 6))
        top_names = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        # Clean feature names
        clean_names = [n.replace("num__", "").replace("cat__", "") for n in top_names]

        ax.barh(range(len(clean_names)), top_importances[::-1], color="steelblue")
        ax.set_yticks(range(len(clean_names)))
        ax.set_yticklabels(clean_names[::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Feature importance plot saved.")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
