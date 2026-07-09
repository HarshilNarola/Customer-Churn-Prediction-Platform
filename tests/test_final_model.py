"""
Test script for training, evaluating, and saving the
final machine learning model.
"""

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from services.preprocessing_service import PreprocessingService
from services.model_training_service import ModelTrainingService


# ==========================================================
# Constants
# ==========================================================

DATASET_PATH = "data/raw/customer_churn_dataset.csv"

MODEL_PATH = "models/best_model.pkl"

REPORT_PATH = "reports/final_model_report.csv"


# ==========================================================
# Helper Function
# ==========================================================

def print_section(title):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ==========================================================
# Main
# ==========================================================

def main():

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    print_section("LOADING DATASET")

    dataframe = pd.read_csv(

        DATASET_PATH

    )

    # ------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------

    print_section("PREPROCESSING DATA")

    preprocessing = PreprocessingService()

    X_train, X_test, y_train, y_test = (

        preprocessing.preprocess(

            dataframe

        )

    )

    # ------------------------------------------------------
    # Train Final Model
    # ------------------------------------------------------

    print_section("TRAINING FINAL MODEL")

    model = RandomForestClassifier(

        n_estimators=200,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    # ------------------------------------------------------
    # Evaluate Model
    # ------------------------------------------------------

    print_section("EVALUATING MODEL")

    trainer = ModelTrainingService()

    results = trainer._evaluate_model(

        model,

        X_test,

        y_test

    )

    print(f"Accuracy  : {results['accuracy']:.6f}")
    print(f"Precision : {results['precision']:.6f}")
    print(f"Recall    : {results['recall']:.6f}")
    print(f"F1 Score  : {results['f1_score']:.6f}")
    print(f"ROC AUC   : {results['roc_auc']:.6f}")

    # ------------------------------------------------------
    # Save Model
    # ------------------------------------------------------

    print_section("SAVING MODEL")

    joblib.dump(

        model,

        MODEL_PATH

    )

    print(f"Saved : {MODEL_PATH}")

    # ------------------------------------------------------
    # Save Report
    # ------------------------------------------------------

    print_section("SAVING REPORT")

    report = pd.DataFrame([{

        "Model": "Random Forest (Tuned)",

        "Accuracy": results["accuracy"],

        "Precision": results["precision"],

        "Recall": results["recall"],

        "F1 Score": results["f1_score"],

        "ROC AUC": results["roc_auc"]

    }])

    report.to_csv(

        REPORT_PATH,

        index=False

    )

    print(f"Saved : {REPORT_PATH}")

    # ------------------------------------------------------
    # Completion
    # ------------------------------------------------------

    print_section("FINAL MODEL COMPLETED")

    print("Final model trained, evaluated and saved successfully.")


if __name__ == "__main__":

    main()