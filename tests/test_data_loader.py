from utils.data_loader import load_dataset

FILE_PATH = "data/raw/customer_churn_dataset.csv"

df = load_dataset(FILE_PATH)

print(df.head())