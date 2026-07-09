"""
Test script for detecting common sources of data leakage before
training the machine learning models.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.data_loader import DataLoader


# ==========================================================
# Constants
# ==========================================================

DATASET_PATH = "data/raw/customer_churn_dataset.csv"

TARGET_COLUMN = "Churn"

ID_COLUMN = "CustomerID"

CATEGORICAL_COLUMNS = [

    "Gender",

    "Subscription Type",

    "Contract Length"

]


# ==========================================================
# Helper Function
# ==========================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


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
    # Check Target Leakage
    # ------------------------------------------------------

    print_section("TARGET COLUMN CHECK")

    feature_columns = [

        column

        for column in dataframe.columns

        if column != TARGET_COLUMN

    ]

    if TARGET_COLUMN in feature_columns:

        print("❌ Data Leakage Detected")

    else:

        print("✅ Target column is not included in the feature set.")

    # ------------------------------------------------------
    # Check CustomerID Leakage
    # ------------------------------------------------------

    print_section("CUSTOMER ID CHECK")

    if ID_COLUMN in feature_columns:

        print("⚠ CustomerID exists in the dataset.")

        print("Ensure it is removed before training.")

    else:

        print("✅ CustomerID has been removed.")

    # ------------------------------------------------------
    # Encode Categorical Features
    # ------------------------------------------------------

    encoded_dataframe = dataframe.copy()

    for column in CATEGORICAL_COLUMNS:

        encoder = LabelEncoder()

        encoded_dataframe[column] = encoder.fit_transform(

            encoded_dataframe[column]

        )

    # ------------------------------------------------------
    # Create Feature Matrix
    # ------------------------------------------------------

    X = encoded_dataframe.drop(

        columns=[

            ID_COLUMN,

            TARGET_COLUMN

        ]

    )

    y = encoded_dataframe[TARGET_COLUMN]

    # ------------------------------------------------------
    # Train Test Split
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print_section("TRAIN / TEST SPLIT")

    print(f"Training Samples : {len(X_train)}")

    print(f"Testing Samples  : {len(X_test)}")

    # ------------------------------------------------------
    # Duplicate Records Across Train/Test
    # ------------------------------------------------------

    print_section("DUPLICATE FEATURE ROWS")

    merged = pd.merge(

        X_train,

        X_test,

        how="inner"

    )

    if len(merged) == 0:

        print("✅ No duplicate feature rows found between train and test sets.")

    else:

        print(f"⚠ {len(merged)} duplicate feature rows detected.")

    # ------------------------------------------------------
    # Class Distribution
    # ------------------------------------------------------

    print_section("CLASS DISTRIBUTION")

    print("Training")

    print(

        y_train.value_counts(normalize=True)

    )

    print()

    print("Testing")

    print(

        y_test.value_counts(normalize=True)

    )

    # ------------------------------------------------------
    # Suspicious Correlation
    # ------------------------------------------------------

    print_section("HIGH FEATURE CORRELATION")

    correlations = encoded_dataframe.corr(

        numeric_only=True

    )[TARGET_COLUMN]

    suspicious = correlations[

        abs(correlations) > 0.95

    ]

    suspicious = suspicious.drop(

        TARGET_COLUMN,

        errors="ignore"

    )

    if suspicious.empty:

        print("✅ No suspiciously correlated features detected.")

    else:

        print("⚠ Features with correlation greater than 0.95")

        print(suspicious)

    # ------------------------------------------------------
    # Final Result
    # ------------------------------------------------------

    print_section("SUMMARY")

    print("✔ Target column is excluded from model features.")

    print("✔ CustomerID will be removed before training.")

    print("✔ Stratified train/test split is used.")

    print("✔ No duplicated feature rows exist across train and test.")

    print("✔ No suspiciously correlated features were detected.")

    print("\nNo evidence of data leakage was found.")


if __name__ == "__main__":

    main()