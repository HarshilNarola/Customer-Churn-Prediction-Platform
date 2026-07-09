"""
Test script for evaluating machine learning models using
5-fold cross validation.
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from services.cross_validation_service import CrossValidationService
from services.preprocessing_service import PreprocessingService
from utils.data_loader import DataLoader


# ==========================================================
# Constants
# ==========================================================

DATASET_PATH = "data/raw/customer_churn_dataset.csv"

MODELS = {

    "Logistic Regression":

        LogisticRegression(

            random_state=42,

            max_iter=1000

        ),

    "Decision Tree":

        DecisionTreeClassifier(

            random_state=42

        ),

    "Random Forest":

        RandomForestClassifier(

            random_state=42

        ),

    "K-Nearest Neighbors":

        KNeighborsClassifier(),

    "Gaussian Naive Bayes":

        GaussianNB(),

    "Support Vector Machine":

        SVC(

            probability=True,

            random_state=42

        )

}


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

    # ------------------------------------------------------
    # Reconstruct Complete Dataset
    # ------------------------------------------------------

    X = pd.concat(

        [X_train, X_test],

        axis=0

    )

    y = pd.concat(

        [y_train, y_test],

        axis=0

    )

    # ------------------------------------------------------
    # Cross Validation
    # ------------------------------------------------------

    cross_validation = CrossValidationService()

    summary = []

    print_section("5-FOLD CROSS VALIDATION")

    for model_name, model in MODELS.items():

        print(f"\n{model_name}")

        print("-" * 60)

        results = cross_validation.evaluate(

            model,

            X,

            y

        )

        print(results)

        summary.append({

            "Model": model_name,

            "Accuracy":

                results["Accuracy"].mean(),

            "Precision":

                results["Precision"].mean(),

            "Recall":

                results["Recall"].mean(),

            "F1 Score":

                results["F1 Score"].mean(),

            "ROC AUC":

                results["ROC AUC"].mean()

        })

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    summary = pd.DataFrame(summary)

    print_section("AVERAGE CROSS VALIDATION RESULTS")

    print(summary)

    # ------------------------------------------------------
    # Best Model
    # ------------------------------------------------------

    best_model = summary.loc[

        summary["Accuracy"].idxmax()

    ]

    print_section("BEST MODEL")

    print(best_model)

    # ------------------------------------------------------
    # Save Report
    # ------------------------------------------------------

    summary.to_csv(

        "reports/cross_validation_results.csv",

        index=False

    )

    print("\nCross validation report saved successfully.")

    print("Location : reports/cross_validation_results.csv")


if __name__ == "__main__":

    main()