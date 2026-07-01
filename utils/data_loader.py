import pandas as pd


class DataLoader:
    """
    Responsible for loading datasets from CSV or Excel files.
    """

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the dataset.

        Parameters
        ----------
        file_path : str
            Path to CSV or Excel file.

        Returns
        -------
        pd.DataFrame
        """

        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)

        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)

        else:
            raise ValueError(
                "Unsupported file format. Only CSV and XLSX files are supported."
            )