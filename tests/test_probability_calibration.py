"""
Test script for evaluating probability calibration of the final model.
"""

import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score
)

from services.preprocessing_service import PreprocessingService
from services.probability_calibration_service import (
    ProbabilityCalibrationService
)
from utils.data_loader import DataLoader


# ==========================================================
# Constants
# ==========================================================

DATASET_PATH = "data/raw/customer_churn_dataset.csv"

BEST_MODEL_PATH = "models/best_model.pkl"

CALIBRATED_MODEL_PATH = "models/calibrated_best_model.pkl"


# ==========================================================
# Helper Functions
# ==========================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_metrics(model, X_test, y_test):

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy   : {accuracy_score(y_test, prediction):.6f}")

    print(f"ROC AUC    : {roc_auc_score(y_test, probability):.6f}")

    print(f"Brier Score: {brier_score_loss(y_test, probability):.6f}")


# ==========================================================
# Main
# ==========================================================

def main():

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    loader = DataLoader()

    dataframe = loader.load_data(DATASET_PATH)

    # ------------------------------------------------------
    # Preprocess Dataset
    # ------------------------------------------------------

    preprocessing_service = PreprocessingService()

    X_train, X_test, y_train, y_test = (

        preprocessing_service.preprocess(dataframe)

    )

    # ------------------------------------------------------
    # Load Best Model
    # ------------------------------------------------------

    model = joblib.load(BEST_MODEL_PATH)

    # ------------------------------------------------------
    # Before Calibration
    # ------------------------------------------------------

    print_section("BEFORE CALIBRATION")

    print_metrics(

        model,

        X_test,

        y_test

    )

    # ------------------------------------------------------
    # Probability Calibration
    # ------------------------------------------------------

    calibration_service = ProbabilityCalibrationService()

    calibrated_model = calibration_service.calibrate(

        model,

        X_train,

        y_train

    )

    # ------------------------------------------------------
    # After Calibration
    # ------------------------------------------------------

    print_section("AFTER CALIBRATION")

    print_metrics(

        calibrated_model,

        X_test,

        y_test

    )

    # ------------------------------------------------------
    # Save Calibrated Model
    # ------------------------------------------------------

    calibration_service.save(

        calibrated_model,

        CALIBRATED_MODEL_PATH

    )

    print_section("MODEL SAVED")

    if os.path.exists(CALIBRATED_MODEL_PATH):

        print("✅ Calibrated model saved successfully.")

        print(f"Location : {CALIBRATED_MODEL_PATH}")

    else:

        print("❌ Failed to save calibrated model.")


if __name__ == "__main__":

    main()