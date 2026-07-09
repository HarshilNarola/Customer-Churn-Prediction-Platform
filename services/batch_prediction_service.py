from pathlib import Path

import joblib
import pandas as pd


class BatchPredictionService:
    """
    Handles batch prediction for customer churn using
    the trained preprocessing pipeline and calibrated model.
    """

    MODEL_PATH = Path("models/calibrated_best_model.pkl")

    PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")

    DOWNLOAD_DIRECTORY = Path("downloads")

    REQUIRED_COLUMNS = [

        "Age",

        "Gender",

        "Tenure",

        "Usage Frequency",

        "Support Calls",

        "Payment Delay",

        "Subscription Type",

        "Contract Length",

        "Total Spend",

        "Last Interaction"

    ]

    CATEGORICAL_COLUMNS = [

        "Gender",

        "Subscription Type",

        "Contract Length"

    ]

    NUMERICAL_COLUMNS = [

        "Age",

        "Tenure",

        "Usage Frequency",

        "Support Calls",

        "Payment Delay",

        "Total Spend",

        "Last Interaction"

    ]

    def __init__(self) -> None:
        """
        Initialize the batch prediction service.

        The model and preprocessing pipeline are
        loaded lazily when first required.
        """

        self.model = None

        self.pipeline = None

        self.DOWNLOAD_DIRECTORY.mkdir(

            parents=True,

            exist_ok=True

        )

    # ==========================================================
    # Load Model & Pipeline
    # ==========================================================

    def load_resources(self) -> None:
        """
        Load the trained model and preprocessing
        pipeline if they have not already been loaded.
        """

        if self.model is None:

            if not self.MODEL_PATH.exists():

                raise FileNotFoundError(

                    f"Model not found: {self.MODEL_PATH}"

                )

            self.model = joblib.load(

                self.MODEL_PATH

            )

        if self.pipeline is None:

            if not self.PIPELINE_PATH.exists():

                raise FileNotFoundError(

                    f"Pipeline not found: {self.PIPELINE_PATH}"

                )

            self.pipeline = joblib.load(

                self.PIPELINE_PATH

            )

    # ==========================================================
    # Load CSV
    # ==========================================================

    def load_csv(
        self,
        filepath: str
    ) -> pd.DataFrame:
        """
        Load a CSV file for batch prediction.
        """

        return pd.read_csv(filepath)

    # ==========================================================
    # Validate Required Columns
    # ==========================================================

    def validate_columns(
        self,
        dataframe: pd.DataFrame
    ) -> list[str]:
        """
        Check whether all required columns are present.
        """

        return [

            column

            for column in self.REQUIRED_COLUMNS

            if column not in dataframe.columns

        ]

    # ==========================================================
    # Preprocess Data
    # ==========================================================

    def preprocess(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply the saved preprocessing pipeline.
        """

        self.load_resources()

        dataframe = dataframe.copy()

        label_encoders = self.pipeline["label_encoders"]

        scaler = self.pipeline["scaler"]

        for column in self.CATEGORICAL_COLUMNS:

            try:

                dataframe[column] = label_encoders[column].transform(

                    dataframe[column]

                )

            except ValueError as error:

                raise ValueError(

                    f"Unknown value found in column '{column}'."

                ) from error

        dataframe[self.NUMERICAL_COLUMNS] = scaler.transform(

            dataframe[self.NUMERICAL_COLUMNS]

        )

        return dataframe

    # ==========================================================
    # Predict
    # ==========================================================

    def predict(
        self,
        dataframe: pd.DataFrame
    ) -> tuple:
        """
        Predict customer churn and probability.
        """

        self.load_resources()

        prediction = self.model.predict(

            dataframe

        )

        probability = self.model.predict_proba(

            dataframe

        )[:, 1]

        return prediction, probability

    # ==========================================================
    # Save Results
    # ==========================================================

    def save_results(
        self,
        original_dataframe: pd.DataFrame,
        prediction,
        probability
    ) -> Path:
        """
        Save prediction results to a CSV file.
        """

        output = original_dataframe.copy()

        output["Prediction"] = [

            "Churn"

            if value == 1

            else "Stay"

            for value in prediction

        ]

        output["Probability (%)"] = (

            probability * 100

        ).round(2)

        output_path = (

            self.DOWNLOAD_DIRECTORY /

            "prediction_results.csv"

        )

        output.to_csv(

            output_path,

            index=False

        )

        return output_path