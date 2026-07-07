import os
import joblib
import pandas as pd


class BatchPredictionService:

    def __init__(self):

        self.pipeline = joblib.load(
            "models/preprocessing_pipeline.pkl"
        )

        self.model = joblib.load(
            "models/best_model.pkl"
        )

    # ==========================================================
    # Load CSV
    # ==========================================================

    def load_csv(self, filepath):

        return pd.read_csv(filepath)

    # ==========================================================
    # Validate Columns
    # ==========================================================

    def validate_columns(self, df):

        required_columns = [

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

        missing = [

            column

            for column in required_columns

            if column not in df.columns

        ]

        return missing

    # ==========================================================
    # Preprocess
    # ==========================================================

    def preprocess(self, df):

        df = df.copy()

        label_encoders = self.pipeline["label_encoders"]

        scaler = self.pipeline["scaler"]

        categorical_columns = [

            "Gender",
            "Subscription Type",
            "Contract Length"

        ]

        for column in categorical_columns:

            df[column] = label_encoders[column].transform(
                df[column]
            )

        numerical_columns = [

            "Age",
            "Tenure",
            "Usage Frequency",
            "Support Calls",
            "Payment Delay",
            "Total Spend",
            "Last Interaction"

        ]

        df[numerical_columns] = scaler.transform(
            df[numerical_columns]
        )

        return df

    # ==========================================================
    # Predict
    # ==========================================================

    def predict(self, df):

        prediction = self.model.predict(df)

        probability = self.model.predict_proba(df)[:, 1]

        return prediction, probability

    # ==========================================================
    # Save Results
    # ==========================================================

    def save_results(

        self,

        original_df,

        prediction,

        probability

    ):

        output = original_df.copy()

        output["Prediction"] = [

            "Churn"

            if p == 1

            else "Stay"

            for p in prediction

        ]

        output["Probability"] = (

            probability * 100

        ).round(2)

        output_path = os.path.join(

            "downloads",

            "prediction_results.csv"

        )

        output.to_csv(

            output_path,

            index=False

        )

        return output_path