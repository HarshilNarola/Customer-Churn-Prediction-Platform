"""
Test script for evaluating machine learning models and
generating evaluation reports.
"""

from services.evaluation_service import EvaluationService
from services.model_training_service import ModelTrainingService
from services.preprocessing_service import PreprocessingService
from utils.data_loader import DataLoader


# ==========================================================
# Constants
# ==========================================================

DATASET_PATH = "data/raw/customer_churn_dataset.csv"

MODELS = [

    (
        "Logistic Regression",
        "train_logistic_regression",
        False
    ),

    (
        "Decision Tree",
        "train_decision_tree",
        True
    ),

    (
        "Random Forest",
        "train_random_forest",
        True
    ),

    (
        "KNN",
        "train_knn",
        False
    ),

    (
        "Gaussian Naive Bayes",
        "train_naive_bayes",
        False
    ),

    (
        "Support Vector Machine",
        "train_svm",
        False
    )

]


# ==========================================================
# Helper Function
# ==========================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ==========================================================
# Main
# ==========================================================

def main():

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    loader = DataLoader()

    dataframe = loader.load_data(

        DATASET_PATH

    )

    # ------------------------------------------------------
    # Preprocess Dataset
    # ------------------------------------------------------

    preprocessing_service = PreprocessingService()

    X_train, X_test, y_train, y_test = (

        preprocessing_service.preprocess(

            dataframe

        )

    )

    feature_names = X_train.columns

    # ------------------------------------------------------
    # Initialize Services
    # ------------------------------------------------------

    trainer = ModelTrainingService()

    evaluator = EvaluationService()

    trained_models = {}

    # ------------------------------------------------------
    # Evaluate Models
    # ------------------------------------------------------

    for model_name, train_method, plot_importance in MODELS:

        model, results = getattr(

            trainer,

            train_method

        )(

            X_train,

            X_test,

            y_train,

            y_test

        )

        trained_models[model_name] = model

        evaluator.add_result(

            model_name,

            results

        )

        evaluator.plot_confusion_matrix(

            model,

            X_test,

            y_test,

            model_name

        )

        evaluator.plot_roc_curve(

            model,

            X_test,

            y_test,

            model_name

        )

        if plot_importance:

            evaluator.plot_feature_importance(

                model,

                feature_names,

                model_name

            )

    # ------------------------------------------------------
    # Reports
    # ------------------------------------------------------

    evaluator.print_comparison()

    evaluator.save_csv()

    evaluator.save_markdown()

    # ------------------------------------------------------
    # Best Model
    # ------------------------------------------------------

    best_model_information = evaluator.get_best_model()

    evaluator.save_best_model(

        trained_models[

            best_model_information["Model"]

        ]

    )

    print_section("BEST MODEL")

    print(best_model_information)


if __name__ == "__main__":

    main()