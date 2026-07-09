"""
Test script for verifying the preprocessing pipeline.
"""

import os

from services.preprocessing_service import PreprocessingService
from utils.data_loader import DataLoader


# ==========================================================
# Constants
# ==========================================================

DATASET_PATH = "data/raw/customer_churn_dataset.csv"

PIPELINE_PATH = "models/preprocessing_pipeline.pkl"


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
    # Dataset Shapes
    # ------------------------------------------------------

    print_section("TRAINING FEATURES")

    print(X_train.shape)

    print_section("TESTING FEATURES")

    print(X_test.shape)

    print_section("TRAINING LABELS")

    print(y_train.shape)

    print_section("TESTING LABELS")

    print(y_test.shape)

    # ------------------------------------------------------
    # Train/Test Split
    # ------------------------------------------------------

    total_samples = len(dataframe)

    print_section("TRAIN / TEST SPLIT")

    print(

        f"Training Samples : {len(X_train)} "

        f"({len(X_train) / total_samples:.2%})"

    )

    print(

        f"Testing Samples  : {len(X_test)} "

        f"({len(X_test) / total_samples:.2%})"

    )

    # ------------------------------------------------------
    # Feature Names
    # ------------------------------------------------------

    print_section("FEATURE COLUMNS")

    print(list(X_train.columns))

    # ------------------------------------------------------
    # Preview
    # ------------------------------------------------------

    print_section("ENCODED & SCALED TRAINING DATA")

    print(X_train.head())

    # ------------------------------------------------------
    # Pipeline Verification
    # ------------------------------------------------------

    print_section("PREPROCESSING PIPELINE")

    if os.path.exists(PIPELINE_PATH):

        print("✅ Preprocessing pipeline saved successfully.")

        print(f"Location : {PIPELINE_PATH}")

    else:

        print("❌ Preprocessing pipeline was not found.")


if __name__ == "__main__":

    main()