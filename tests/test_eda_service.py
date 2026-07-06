from services.eda_service import EDAService
from utils.data_loader import DataLoader

DATASET_PATH = "data/raw/customer_churn_dataset.csv"


def main():

    # ===========================
    # Load Dataset
    # ===========================

    loader = DataLoader()
    df = loader.load_data(DATASET_PATH)

    # ===========================
    # Initialize EDA Service
    # ===========================

    eda = EDAService(df)

    # ==========================================================
    # Target Variable Analysis
    # ==========================================================

    print("=" * 60)
    print("Generating Target Variable Analysis...")
    print("=" * 60)

    eda.plot_target_distribution()
    eda.plot_target_percentage()

    # ==========================================================
    # Numerical Feature Analysis
    # ==========================================================

    numerical_columns = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction",
    ]

    print()
    print("=" * 60)
    print("Generating Numerical Feature Analysis...")
    print("=" * 60)

    for column in numerical_columns:

        print(f"Generating Histogram for {column}")
        eda.plot_histogram(column)

        print(f"Generating Boxplot for {column}")
        eda.plot_boxplot(column)

    # ==========================================================
    # Numerical Feature vs Churn
    # ==========================================================

    print()
    print("=" * 60)
    print("Generating Numerical Feature vs Churn Analysis...")
    print("=" * 60)

    for column in numerical_columns:

        print(f"Generating {column} vs Churn")
        eda.plot_boxplot_by_target(column)

    # ==========================================================
    # Categorical Feature Analysis
    # ==========================================================

    categorical_columns = [
        "Gender",
        "Subscription Type",
        "Contract Length",
    ]

    print()
    print("=" * 60)
    print("Generating Categorical Feature Analysis...")
    print("=" * 60)

    for column in categorical_columns:

        print(f"Generating Countplot for {column}")
        eda.plot_countplot(column)

    # ==========================================================
    # Categorical Feature vs Churn
    # ==========================================================

    print()
    print("=" * 60)
    print("Generating Categorical Feature vs Churn Analysis...")
    print("=" * 60)

    for column in categorical_columns:

        print(f"Generating {column} vs Churn")
        eda.plot_countplot_by_target(column)
    

    print()
    print("=" * 60)
    print("Generating Correlation Heatmap...")
    print("=" * 60)

    eda.plot_correlation_heatmap()


if __name__ == "__main__":
    main()