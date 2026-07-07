import joblib
import pandas as pd


class PredictionController:

    def __init__(self):

        self.model = joblib.load(
            "models/best_model.pkl"
        )

        self.pipeline = joblib.load(
            "models/preprocessing_pipeline.pkl"
        )

    # ==========================================================
    # Predict Customer Churn
    # ==========================================================

    def predict(self, form_data):

        df = pd.DataFrame([form_data])

        # -----------------------------
        # Label Encoding
        # -----------------------------

        label_encoders = self.pipeline["label_encoders"]

        categorical_columns = [
            "Gender",
            "Subscription Type",
            "Contract Length"
        ]

        for column in categorical_columns:

            encoder = label_encoders[column]

            df[column] = encoder.transform(df[column])

        # -----------------------------
        # Standard Scaling
        # -----------------------------

        scaler = self.pipeline["scaler"]

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

        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = self.model.predict(df)[0]

        probability = self.model.predict_proba(df)[0][1]

        return prediction, probability