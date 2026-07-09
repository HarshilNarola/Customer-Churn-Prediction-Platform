import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)


class CrossValidationService:
    """
    Performs k-fold cross-validation for classification models
    and returns the evaluation metrics for each fold.
    """

    N_SPLITS = 5
    RANDOM_STATE = 42

    SCORING = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    def evaluate(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Perform stratified k-fold cross-validation.

        Parameters
        ----------
        model
            Machine learning model.

        X : pd.DataFrame
            Feature matrix.

        y : pd.Series
            Target labels.

        Returns
        -------
        pd.DataFrame
            Cross-validation results for each fold.
        """

        cv = StratifiedKFold(

            n_splits=self.N_SPLITS,

            shuffle=True,

            random_state=self.RANDOM_STATE

        )

        scores = cross_validate(

            estimator=model,

            X=X,

            y=y,

            cv=cv,

            scoring=self.SCORING,

            return_train_score=False

        )

        results = pd.DataFrame({

            "Accuracy": scores["test_accuracy"],

            "Precision": scores["test_precision"],

            "Recall": scores["test_recall"],

            "F1 Score": scores["test_f1"],

            "ROC AUC": scores["test_roc_auc"]

        })

        return results