"""
Test script for generating Exploratory Data Analysis (EDA)
visualizations for the customer churn dataset.
"""

from services.eda_service import EDAService
from utils.data_loader import DataLoader


# ==========================================================
# Constants
# ==========================================================

DATASET_PATH = "data/raw/customer_churn_dataset.csv"

NUMERICAL_COLUMNS = [

    "Age",

    "Tenure",

    "Usage Frequency",

    "Support Calls",

    "Payment Delay",

    "Total Spend",

    "Last Interaction"

]

CATEGORICAL_COLUMNS = [

    "Gender",

    "Subscription Type",

    "Contract Length"

]


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

    dataframe = loader.load_data(

        DATASET_PATH

    )

    # ------------------------------------------------------
    # Initialize EDA Service
    # ------------------------------------------------------

    eda_service = EDAService(

        dataframe

    )

    # ------------------------------------------------------
    # Target Variable Analysis
    # ------------------------------------------------------

    print_section("TARGET VARIABLE ANALYSIS")

    eda_service.plot_target_distribution()

    eda_service.plot_target_percentage()

    # ------------------------------------------------------
    # Numerical Feature Analysis
    # ------------------------------------------------------

    print_section("NUMERICAL FEATURE ANALYSIS")

    for column in NUMERICAL_COLUMNS:

        print(f"Generating Histogram : {column}")

        eda_service.plot_histogram(column)

        print(f"Generating Boxplot   : {column}")

        eda_service.plot_boxplot(column)

    # ------------------------------------------------------
    # Numerical Feature vs Target
    # ------------------------------------------------------

    print_section("NUMERICAL FEATURES VS CHURN")

    for column in NUMERICAL_COLUMNS:

        print(f"Generating {column} vs Churn")

        eda_service.plot_boxplot_by_target(column)

    # ------------------------------------------------------
    # Categorical Feature Analysis
    # ------------------------------------------------------

    print_section("CATEGORICAL FEATURE ANALYSIS")

    for column in CATEGORICAL_COLUMNS:

        print(f"Generating Countplot : {column}")

        eda_service.plot_countplot(column)

    # ------------------------------------------------------
    # Categorical Feature vs Target
    # ------------------------------------------------------

    print_section("CATEGORICAL FEATURES VS CHURN")

    for column in CATEGORICAL_COLUMNS:

        print(f"Generating {column} vs Churn")

        eda_service.plot_countplot_by_target(column)

    # ------------------------------------------------------
    # Correlation Heatmap
    # ------------------------------------------------------

    print_section("CORRELATION HEATMAP")

    eda_service.plot_correlation_heatmap()

    # ------------------------------------------------------
    # Completion
    # ------------------------------------------------------

    print_section("EDA COMPLETED")

    print("All EDA visualizations generated successfully.")

    print("Location : static/images/eda/")


if __name__ == "__main__":

    main()