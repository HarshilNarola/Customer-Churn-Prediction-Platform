import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the customer churn dataset.

    Parameters
    ----------
    file_path : str
        Path of the Excel or CSV file.

    Returns
    -------
    pd.DataFrame
    """

    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)

    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)

    else:
        raise ValueError("Unsupported file format")