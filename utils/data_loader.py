from pathlib import Path

import pandas as pd


class DataLoader:
    """
    Provides utility methods for loading datasets from supported file formats.
    """

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load a dataset from a CSV or Excel file.

        Parameters
        ----------
        file_path : str
            Path to the input dataset.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.

        ValueError
            If the file format is not supported.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        extension = path.suffix.lower()

        if extension == ".csv":
            return pd.read_csv(path)

        if extension in [".xlsx", ".xls"]:
            return pd.read_excel(path)

        raise ValueError(
            "Unsupported file format. Only CSV and Excel (.xls/.xlsx) files are supported."
        )