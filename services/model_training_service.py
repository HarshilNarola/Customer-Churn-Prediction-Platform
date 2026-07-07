import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


class ModelTrainingService:
    """
    Provides utilities for training and evaluating
    machine learning classification models.
    """

    # ==========================================================
    # Private Evaluation Method
    # ==========================================================

    def _evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred)
        }

        return results

    # ==========================================================
    # Logistic Regression
    # ==========================================================

    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

        model.fit(X_train, y_train)

        results = self._evaluate_model(model, X_test, y_test)

        return model, results

    # ==========================================================
    # Decision Tree
    # ==========================================================

    def train_decision_tree(self, X_train, X_test, y_train, y_test):
        model = DecisionTreeClassifier(
            random_state=42
        )

        model.fit(X_train, y_train)

        results = self._evaluate_model(model, X_test, y_test)

        return model, results

    # ==========================================================
    # Random Forest
    # ==========================================================

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        results = self._evaluate_model(model, X_test, y_test)

        return model, results

    # ==========================================================
    # K-Nearest Neighbors
    # ==========================================================

    def train_knn(self, X_train, X_test, y_train, y_test):
        model = KNeighborsClassifier(
            n_neighbors=5
        )

        model.fit(X_train, y_train)

        results = self._evaluate_model(model, X_test, y_test)

        return model, results

    # ==========================================================
    # Gaussian Naive Bayes
    # ==========================================================

    def train_naive_bayes(self, X_train, X_test, y_train, y_test):
        model = GaussianNB()

        model.fit(X_train, y_train)

        results = self._evaluate_model(model, X_test, y_test)

        return model, results

    # ==========================================================
    # Support Vector Machine
    # ==========================================================

    def train_svm(self, X_train, X_test, y_train, y_test):
        model = SVC(
            probability=True,
            kernel="rbf",
            C=1.0,
            random_state=42
        )

        model.fit(X_train, y_train)

        results = self._evaluate_model(model, X_test, y_test)

        return model, results

    # ==========================================================
    # Save Model
    # ==========================================================

    def save_model(self, model, filename):
        joblib.dump(model, f"models/{filename}")