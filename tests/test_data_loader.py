"""
Test script for verifying the DataLoader utility.
"""

from utils.data_loader import DataLoader


# ==========================================================
# Constants
# ==========================================================

DATASET_PATH = "data/raw/customer_churn_dataset.csv"


# ==========================================================
# Main
# ==========================================================

def main():

    loader = DataLoader()

    dataframe = loader.load_data(DATASET_PATH)

    print("\n" + "=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print("\nDataset Shape")
    print(dataframe.shape)

    print("\nFirst 5 Rows")
    print(dataframe.head())


if __name__ == "__main__":

    main()