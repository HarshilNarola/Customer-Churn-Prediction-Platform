from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ModelTrainingService:
    """
    Provides methods for training, evaluating and saving
    machine learning classification models.
    """

    RANDOM_STATE = 42
    MODEL_DIRECTORY = Path("models")

    # ==========================================================
    # Private Methods
    # ==========================================================

    def _evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> dict:
        """
        Evaluate a trained classification model.
        """

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        return {

            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "precision": precision_score(
                y_test,
                y_pred
            ),

            "recall": recall_score(
                y_test,
                y_pred
            ),

            "f1_score": f1_score(
                y_test,
                y_pred
            ),

            "roc_auc": roc_auc_score(
                y_test,
                y_prob
            ),

            "confusion_matrix": confusion_matrix(
                y_test,
                y_pred
            ),

            "classification_report": classification_report(
                y_test,
                y_pred
            )

        }

    def _train_model(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ):
        """
        Train and evaluate a model.
        """

        model.fit(
            X_train,
            y_train
        )

        results = self._evaluate_model(
            model,
            X_test,
            y_test
        )

        return model, results

    # ==========================================================
    # Logistic Regression
    # ==========================================================

    def train_logistic_regression(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):
        """
        Train a Logistic Regression classifier.
        """

        model = LogisticRegression(

            random_state=self.RANDOM_STATE,

            max_iter=1000

        )

        return self._train_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    # ==========================================================
    # Decision Tree
    # ==========================================================

    def train_decision_tree(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):
        """
        Train a Decision Tree classifier.
        """

        model = DecisionTreeClassifier(

            random_state=self.RANDOM_STATE

        )

        return self._train_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    # ==========================================================
    # Random Forest
    # ==========================================================

    def train_random_forest(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):
        """
        Train a Random Forest classifier.
        """

        model = RandomForestClassifier(

            n_estimators=100,

            random_state=self.RANDOM_STATE

        )

        return self._train_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    # ==========================================================
    # K-Nearest Neighbors
    # ==========================================================

    def train_knn(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):
        """
        Train a K-Nearest Neighbors classifier.
        """

        model = KNeighborsClassifier(

            n_neighbors=5

        )

        return self._train_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    # ==========================================================
    # Gaussian Naive Bayes
    # ==========================================================

    def train_naive_bayes(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):
        """
        Train a Gaussian Naive Bayes classifier.
        """

        model = GaussianNB()

        return self._train_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    # ==========================================================
    # Support Vector Machine
    # ==========================================================

    def train_svm(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):
        """
        Train a Support Vector Machine classifier.
        """

        model = SVC(

            probability=True,

            kernel="rbf",

            C=1.0,

            random_state=self.RANDOM_STATE

        )

        return self._train_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    # ==========================================================
    # Save Model
    # ==========================================================

    def save_model(
        self,
        model,
        filename: str
    ) -> None:
        """
        Save a trained model.
        """

        self.MODEL_DIRECTORY.mkdir(
            exist_ok=True
        )

        joblib.dump(

            model,

            self.MODEL_DIRECTORY / filename

        )