import os
import joblib
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


class EvaluationService:
    """
    Provides utilities for evaluating machine learning models,
    comparing their performance, and generating evaluation reports.
    """

    def __init__(self):

        self.results = []

        self.output_dir = "static/images/evaluation"

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    # ==========================================================
    # Add Model Result
    # ==========================================================

    def add_result(
        self,
        model_name,
        metrics
    ):

        self.results.append({

            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
            "ROC AUC": metrics["roc_auc"]

        })

    # ==========================================================
    # Create DataFrame
    # ==========================================================

    def get_dataframe(self):

        return pd.DataFrame(self.results)

    # ==========================================================
    # Print Comparison
    # ==========================================================

    def print_comparison(self):

        print()
        print("=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        print(self.get_dataframe())

    # ==========================================================
    # Save CSV
    # ==========================================================

    def save_csv(self):

        self.get_dataframe().to_csv(
            "reports/model_comparison.csv",
            index=False
        )

    # ==========================================================
    # Save Markdown
    # ==========================================================

    def save_markdown(self):

        with open(
            "reports/model_comparison.md",
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                self.get_dataframe().to_markdown(
                    index=False
                )
            )

    # ==========================================================
    # Best Model
    # ==========================================================

    def get_best_model(self):

        df = self.get_dataframe()

        return df.loc[
            df["Accuracy"].idxmax()
        ]

    # ==========================================================
    # Plot Confusion Matrix
    # ==========================================================

    def plot_confusion_matrix(
        self,
        model,
        X_test,
        y_test,
        model_name
    ):

        plt.figure(figsize=(6, 5))

        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test
        )

        plt.title(
            f"{model_name} Confusion Matrix"
        )

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            os.path.join(
                self.output_dir,
                filename
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ==========================================================
    # Plot ROC Curve
    # ==========================================================

    def plot_roc_curve(
        self,
        model,
        X_test,
        y_test,
        model_name
    ):

        plt.figure(figsize=(6, 5))

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test
        )

        plt.title(
            f"{model_name} ROC Curve"
        )

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + "_roc_curve.png"
        )

        plt.savefig(
            os.path.join(
                self.output_dir,
                filename
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ==========================================================
    # Plot Feature Importance
    # ==========================================================

    def plot_feature_importance(
        self,
        model,
        feature_names,
        model_name
    ):

        if not hasattr(model, "feature_importances_"):
            return

        importances = model.feature_importances_

        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))

        plt.bar(
            range(len(importances)),
            importances[indices]
        )

        plt.xticks(
            range(len(importances)),
            [feature_names[i] for i in indices],
            rotation=45,
            ha="right"
        )

        plt.title(
            f"{model_name} Feature Importance"
        )

        plt.xlabel("Features")
        plt.ylabel("Importance")

        plt.tight_layout()

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + "_feature_importance.png"
        )

        plt.savefig(
            os.path.join(
                self.output_dir,
                filename
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ==========================================================
    # Save Best Model
    # ==========================================================

    def save_best_model(
        self,
        model
    ):

        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            model,
            "models/best_model.pkl"
        )

        print()
        print("=" * 80)
        print("BEST MODEL SAVED")
        print("=" * 80)
        print("models/best_model.pkl")