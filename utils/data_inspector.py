import pandas as pd


class DataInspector:
    """
    Performs basic inspection of a dataset.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def get_shape(self):
        return self.df.shape

    def get_columns(self):
        return list(self.df.columns)

    def get_data_types(self):
        return self.df.dtypes

    def get_missing_values(self):
        return self.df.isnull().sum()

    def get_duplicate_rows(self):
        return self.df.duplicated().sum()

    def get_summary_statistics(self):
        return self.df.describe(include="all")

    def get_target_distribution(self):
        return self.df["Churn"].value_counts()

    def get_numerical_columns(self):
        return list(
            self.df.select_dtypes(include=["number"]).columns
        )

    def get_categorical_columns(self):
        return list(
            self.df.select_dtypes(include=["object", "category"]).columns
        )