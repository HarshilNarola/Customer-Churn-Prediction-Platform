from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


class EvaluationService:
    """
    Provides utilities for evaluating machine learning models,
    comparing their performance, generating reports and
    saving evaluation visualizations.
    """

    REPORT_DIRECTORY = Path("reports")
    MODEL_DIRECTORY = Path("models")
    IMAGE_DIRECTORY = Path("static/images/evaluation")

    def __init__(self) -> None:
        """
        Initialize the evaluation service.
        """

        self.results = []

        self.REPORT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True
        )

        self.MODEL_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True
        )

        self.IMAGE_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True
        )

    # ==========================================================
    # Add Model Result
    # ==========================================================

    def add_result(
        self,
        model_name: str,
        metrics: dict
    ) -> None:
        """
        Store the evaluation metrics of a trained model.
        """

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

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return all stored model evaluation results
        as a DataFrame.
        """

        return pd.DataFrame(self.results)

    # ==========================================================
    # Print Comparison
    # ==========================================================

    def print_comparison(self) -> None:
        """
        Print the comparison table of all models.
        """

        print()
        print("=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        print(self.get_dataframe())

    # ==========================================================
    # Save CSV
    # ==========================================================

    def save_csv(self) -> None:
        """
        Save model comparison as CSV.
        """

        self.get_dataframe().to_csv(

            self.REPORT_DIRECTORY / "model_comparison.csv",

            index=False

        )

    # ==========================================================
    # Save Markdown
    # ==========================================================

    def save_markdown(self) -> None:
        """
        Save model comparison as a Markdown table.
        """

        with open(

            self.REPORT_DIRECTORY / "model_comparison.md",

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

    def get_best_model(self) -> pd.Series:
        """
        Return the best-performing model based on accuracy.
        """

        dataframe = self.get_dataframe()

        return dataframe.loc[
            dataframe["Accuracy"].idxmax()
        ]

    # ==========================================================
    # Confusion Matrix
    # ==========================================================

    def plot_confusion_matrix(
        self,
        model,
        X_test,
        y_test,
        model_name: str
    ) -> None:
        """
        Save the confusion matrix plot.
        """

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

            self.IMAGE_DIRECTORY / filename,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

    # ==========================================================
    # ROC Curve
    # ==========================================================

    def plot_roc_curve(
        self,
        model,
        X_test,
        y_test,
        model_name: str
    ) -> None:
        """
        Save the ROC curve.
        """

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

            self.IMAGE_DIRECTORY / filename,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

    # ==========================================================
    # Feature Importance
    # ==========================================================

    def plot_feature_importance(
        self,
        model,
        feature_names,
        model_name: str
    ) -> None:
        """
        Save feature importance visualization for
        tree-based models.
        """

        if not hasattr(model, "feature_importances_"):
            return

        importances = model.feature_importances_

        indices = np.argsort(
            importances
        )[::-1]

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

        plt.xlabel("Features")

        plt.ylabel("Importance")

        plt.title(
            f"{model_name} Feature Importance"
        )

        plt.tight_layout()

        filename = (

            model_name.lower()

            .replace(" ", "_")

            + "_feature_importance.png"

        )

        plt.savefig(

            self.IMAGE_DIRECTORY / filename,

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
    ) -> None:
        """
        Save the best-performing trained model.
        """

        joblib.dump(

            model,

            self.MODEL_DIRECTORY / "best_model.pkl"

        )

        print()
        print("=" * 80)
        print("BEST MODEL SAVED")
        print("=" * 80)
        print(self.MODEL_DIRECTORY / "best_model.pkl")