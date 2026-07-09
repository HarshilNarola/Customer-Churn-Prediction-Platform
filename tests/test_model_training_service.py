"""
Test script for training and evaluating machine learning models.
"""

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
        "logistic_regression.pkl",
        "train_logistic_regression"
    ),

    (
        "Decision Tree",
        "decision_tree.pkl",
        "train_decision_tree"
    ),

    (
        "Random Forest",
        "random_forest.pkl",
        "train_random_forest"
    ),

    (
        "K-Nearest Neighbors",
        "knn.pkl",
        "train_knn"
    ),

    (
        "Gaussian Naive Bayes",
        "naive_bayes.pkl",
        "train_naive_bayes"
    ),

    (
        "Support Vector Machine",
        "svm.pkl",
        "train_svm"
    )

]


# ==========================================================
# Helper Functions
# ==========================================================

def print_section(title):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_results(model_name, results):

    print_section(model_name)

    print(f"Accuracy   : {results['accuracy']:.4f}")
    print(f"Precision  : {results['precision']:.4f}")
    print(f"Recall     : {results['recall']:.4f}")
    print(f"F1 Score   : {results['f1_score']:.4f}")
    print(f"ROC AUC    : {results['roc_auc']:.4f}")

    print("\nConfusion Matrix")
    print(results["confusion_matrix"])

    print("\nClassification Report")
    print(results["classification_report"])


# ==========================================================
# Main
# ==========================================================

def main():

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    loader = DataLoader()

    dataframe = loader.load_data(DATASET_PATH)

    # ------------------------------------------------------
    # Preprocess Dataset
    # ------------------------------------------------------

    preprocessing_service = PreprocessingService()

    X_train, X_test, y_train, y_test = (

        preprocessing_service.preprocess(dataframe)

    )

    # ------------------------------------------------------
    # Initialize Trainer
    # ------------------------------------------------------

    trainer = ModelTrainingService()

    # ------------------------------------------------------
    # Train All Models
    # ------------------------------------------------------

    for model_name, filename, train_method in MODELS:

        model, results = getattr(

            trainer,

            train_method

        )(

            X_train,

            X_test,

            y_train,

            y_test

        )

        trainer.save_model(

            model,

            filename

        )

        print_results(

            model_name,

            results

        )


if __name__ == "__main__":

    main()