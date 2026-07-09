"""
Test script for verifying the DataInspector utility.
"""

from utils.data_loader import DataLoader
from utils.data_inspector import DataInspector


# ==========================================================
# Constants
# ==========================================================

DATASET_PATH = "data/raw/customer_churn_dataset.csv"


# ==========================================================
# Helper Function
# ==========================================================

def print_section(title, data):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(data)


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
    # Inspect Dataset
    # ------------------------------------------------------

    inspector = DataInspector(dataframe)

    print_section(
        "DATASET SHAPE",
        inspector.get_shape()
    )

    print_section(
        "COLUMN NAMES",
        inspector.get_columns()
    )

    print_section(
        "DATA TYPES",
        inspector.get_data_types()
    )

    print_section(
        "MISSING VALUES",
        inspector.get_missing_values()
    )

    print_section(
        "DUPLICATE ROWS",
        inspector.get_duplicate_rows()
    )

    print_section(
        "NUMERICAL COLUMNS",
        inspector.get_numerical_columns()
    )

    print_section(
        "CATEGORICAL COLUMNS",
        inspector.get_categorical_columns()
    )

    print_section(
        "TARGET DISTRIBUTION",
        inspector.get_target_distribution()
    )

    print_section(
        "SUMMARY STATISTICS",
        inspector.get_summary_statistics()
    )


if __name__ == "__main__":

    main()