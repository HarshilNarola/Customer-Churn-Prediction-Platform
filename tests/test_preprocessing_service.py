from utils.data_loader import DataLoader
from services.preprocessing_service import PreprocessingService


def main():

    loader = DataLoader()
    df = loader.load_data("data/raw/customer_churn_dataset.csv")

    service = PreprocessingService()

    X_train, X_test, y_train, y_test = service.preprocess(df)

    print("=" * 60)
    print("Training Features Shape")
    print(X_train.shape)

    print("=" * 60)
    print("Testing Features Shape")
    print(X_test.shape)

    print("=" * 60)
    print("Training Labels Shape")
    print(y_train.shape)

    print("=" * 60)
    print("Testing Labels Shape")
    print(y_test.shape)

    print("=" * 60)
    print("Encoded Training Data")
    print(X_train.head())

    print("=" * 60)
    print("Preprocessing Pipeline Saved Successfully")


if __name__ == "__main__":
    main()