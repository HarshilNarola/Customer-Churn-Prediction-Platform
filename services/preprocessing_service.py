import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class PreprocessingService:
    """
    Handles preprocessing of the customer churn dataset.

    This service performs:
    - Removal of unnecessary columns.
    - Encoding of categorical features.
    - Train-test split.
    - Standardization of numerical features.
    - Saving the preprocessing pipeline.
    """

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

    PIPELINE_PATH = "models/preprocessing_pipeline.pkl"

    def preprocess(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Preprocess the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input customer churn dataset.

        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test
        """

        # ==========================================================
        # Create Copy of Dataset
        # ==========================================================

        df = df.copy()

        # ==========================================================
        # Drop CustomerID
        # ==========================================================

        df = df.drop(
            columns=["CustomerID"],
            errors="ignore"
        )

        # ==========================================================
        # Encode Categorical Features
        # ==========================================================

        label_encoders = {}

        for column in self.CATEGORICAL_COLUMNS:

            encoder = LabelEncoder()

            df[column] = encoder.fit_transform(
                df[column]
            )

            label_encoders[column] = encoder

        # ==========================================================
        # Separate Features and Target
        # ==========================================================

        X = df.drop(columns=["Churn"])

        y = df["Churn"]

        # ==========================================================
        # Train-Test Split
        # ==========================================================

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42,

            stratify=y

        )

        # ==========================================================
        # Feature Scaling
        # ==========================================================

        scaler = StandardScaler()

        X_train[self.NUMERICAL_COLUMNS] = scaler.fit_transform(

            X_train[self.NUMERICAL_COLUMNS]

        )

        X_test[self.NUMERICAL_COLUMNS] = scaler.transform(

            X_test[self.NUMERICAL_COLUMNS]

        )

        # ==========================================================
        # Save Preprocessing Pipeline
        # ==========================================================

        preprocessing_pipeline = {

            "label_encoders": label_encoders,

            "scaler": scaler

        }

        joblib.dump(

            preprocessing_pipeline,

            self.PIPELINE_PATH

        )

        # ==========================================================
        # Return Processed Data
        # ==========================================================

        return (

            X_train,

            X_test,

            y_train,

            y_test

        )