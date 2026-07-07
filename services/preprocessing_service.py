import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class PreprocessingService:

    def preprocess(self, df):

        # -----------------------------
        # Drop CustomerID
        # -----------------------------
        df = df.drop(columns=["CustomerID"])

        # -----------------------------
        # Encode categorical columns
        # -----------------------------
        label_encoders = {}

        categorical_columns = [
            "Gender",
            "Subscription Type",
            "Contract Length"
        ]

        for column in categorical_columns:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
            label_encoders[column] = encoder

        # -----------------------------
        # Separate Features and Target
        # -----------------------------
        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        # -----------------------------
        # Train Test Split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # -----------------------------
        # Scale Numerical Columns
        # -----------------------------
        numerical_columns = [
            "Age",
            "Tenure",
            "Usage Frequency",
            "Support Calls",
            "Payment Delay",
            "Total Spend",
            "Last Interaction"
        ]

        scaler = StandardScaler()

        X_train[numerical_columns] = scaler.fit_transform(
            X_train[numerical_columns]
        )

        X_test[numerical_columns] = scaler.transform(
            X_test[numerical_columns]
        )

        # -----------------------------
        # Save preprocessing pipeline
        # -----------------------------
        preprocessing_pipeline = {
            "label_encoders": label_encoders,
            "scaler": scaler
        }

        joblib.dump(
            preprocessing_pipeline,
            "models/preprocessing_pipeline.pkl"
        )

        return X_train, X_test, y_train, y_test