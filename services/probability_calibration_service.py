from pathlib import Path

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV


class ProbabilityCalibrationService:
    """
    Performs probability calibration for classification models
    and provides utilities for saving the calibrated model.
    """

    METHOD = "sigmoid"
    CV = 5

    def calibrate(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> CalibratedClassifierCV:
        """
        Calibrate a trained classification model.

        Parameters
        ----------
        model
            Trained machine learning model.

        X_train : pd.DataFrame
            Training feature matrix.

        y_train : pd.Series
            Training target labels.

        Returns
        -------
        CalibratedClassifierCV
            Calibrated classification model.
        """

        calibrated_model = CalibratedClassifierCV(

            estimator=model,

            method=self.METHOD,

            cv=self.CV

        )

        calibrated_model.fit(

            X_train,

            y_train

        )

        return calibrated_model

    # ==========================================================
    # Save Model
    # ==========================================================

    def save(
        self,
        model,
        filename: str
    ) -> None:
        """
        Save the calibrated model.

        Parameters
        ----------
        model
            Calibrated machine learning model.

        filename : str
            Output file path.
        """

        path = Path(filename)

        path.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        joblib.dump(

            model,

            path

        )