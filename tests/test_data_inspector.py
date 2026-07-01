from utils.data_loader import DataLoader
from utils.data_inspector import DataInspector


DATASET_PATH = "data/raw/customer_churn_dataset.csv"


def main():

    # Load Dataset
    loader = DataLoader()
    df = loader.load_data(DATASET_PATH)

    # Inspect Dataset
    inspector = DataInspector(df)

    print("=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(inspector.get_shape())

    print()

    print("=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(inspector.get_columns())

    print()

    print("=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(inspector.get_data_types())

    print()

    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(inspector.get_missing_values())

    print()

    print("=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)
    print(inspector.get_duplicate_rows())

    print()

    print("=" * 60)
    print("NUMERICAL COLUMNS")
    print("=" * 60)
    print(inspector.get_numerical_columns())

    print()

    print("=" * 60)
    print("CATEGORICAL COLUMNS")
    print("=" * 60)
    print(inspector.get_categorical_columns())

    print()

    print("=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)
    print(inspector.get_target_distribution())

    print()

    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(inspector.get_summary_statistics())


if __name__ == "__main__":
    main()