import pandas as pd

from sklearn.model_selection import GridSearchCV


class HyperparameterTuningService:
    """
    Performs hyperparameter tuning using GridSearchCV
    and returns the optimized model along with the
    cross-validation results.
    """

    CV = 5
    SCORING = "accuracy"
    N_JOBS = -1
    VERBOSE = 2

    def tune(
        self,
        model,
        parameter_grid: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> dict:
        """
        Perform hyperparameter tuning.

        Parameters
        ----------
        model
            Machine learning model.

        parameter_grid : dict
            Dictionary containing the hyperparameter search space.

        X_train : pd.DataFrame
            Training feature matrix.

        y_train : pd.Series
            Training target labels.

        Returns
        -------
        dict
            Dictionary containing the best model,
            best parameters, best cross-validation score
            and complete GridSearchCV results.
        """

        grid_search = GridSearchCV(

            estimator=model,

            param_grid=parameter_grid,

            cv=self.CV,

            scoring=self.SCORING,

            n_jobs=self.N_JOBS,

            verbose=self.VERBOSE,

            refit=True

        )

        grid_search.fit(

            X_train,

            y_train

        )

        return {

            "best_model": grid_search.best_estimator_,

            "best_parameters": grid_search.best_params_,

            "best_score": grid_search.best_score_,

            "cv_results": pd.DataFrame(

                grid_search.cv_results_

            )

        }