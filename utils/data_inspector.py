import pandas as pd


class DataInspector:
    """
    Provides utility methods for inspecting a dataset.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the DataInspector with a pandas DataFrame.
        """
        self.dataframe = dataframe

    def get_shape(self) -> tuple:
        """
        Return the shape of the dataset.
        """
        return self.dataframe.shape

    def get_columns(self) -> list:
        """
        Return all column names.
        """
        return list(self.dataframe.columns)

    def get_data_types(self) -> pd.Series:
        """
        Return the data type of each column.
        """
        return self.dataframe.dtypes

    def get_missing_values(self) -> pd.Series:
        """
        Return the number of missing values in each column.
        """
        return self.dataframe.isnull().sum()

    def get_duplicate_rows(self) -> int:
        """
        Return the number of duplicate rows.
        """
        return self.dataframe.duplicated().sum()

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Return summary statistics for the dataset.
        """
        return self.dataframe.describe(include="all")

    def get_target_distribution(self) -> pd.Series:
        """
        Return the distribution of the target variable.
        """
        if "Churn" not in self.dataframe.columns:
            raise KeyError("Target column 'Churn' not found.")

        return self.dataframe["Churn"].value_counts()

    def get_numerical_columns(self) -> list:
        """
        Return all numerical column names.
        """
        return list(
            self.dataframe.select_dtypes(
                include=["number"]
            ).columns
        )

    def get_categorical_columns(self) -> list:
        """
        Return all categorical column names.
        """
        return list(
            self.dataframe.select_dtypes(
                include=["object", "category"]
            ).columns
        )