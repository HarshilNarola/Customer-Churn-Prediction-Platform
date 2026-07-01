from utils.data_loader import DataLoader

DATASET_PATH = "data/raw/customer_churn_dataset.csv"


def main():

    loader = DataLoader()

    df = loader.load_data(DATASET_PATH)

    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print(df.head())


if __name__ == "__main__":
    main()