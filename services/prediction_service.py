from pathlib import Path

import joblib
import pandas as pd


class PredictionService:
    """
    Handles preprocessing and prediction for a
    single customer using the trained machine
    learning model.
    """

    MODEL_PATH = Path("models/calibrated_best_model.pkl")
    PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")

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
        Load the trained model and preprocessing pipeline.
        """

        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {self.MODEL_PATH}"
            )

        if not self.PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Pipeline not found: {self.PIPELINE_PATH}"
            )

        self.model = joblib.load(
            self.MODEL_PATH
        )

        self.pipeline = joblib.load(
            self.PIPELINE_PATH
        )

    # ==========================================================
    # Preprocess Input
    # ==========================================================

    def preprocess(
        self,
        form_data: dict
    ) -> pd.DataFrame:
        """
        Apply the saved preprocessing pipeline
        to a single customer.
        """

        dataframe = pd.DataFrame([form_data])

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
        form_data: dict
    ) -> tuple[str, float]:
        """
        Predict customer churn.

        Returns
        -------
        tuple
            Prediction label and calibrated probability.
        """

        dataframe = self.preprocess(
            form_data
        )

        prediction = self.model.predict(
            dataframe
        )[0]

        probability = self.model.predict_proba(
            dataframe
        )[0][1]

        prediction_label = (

            "Churn"

            if prediction == 1

            else "Stay"

        )

        return prediction_label, probability