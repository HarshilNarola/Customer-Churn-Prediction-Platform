"""
Test script for hyperparameter tuning of machine learning models.
"""

import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from services.hyperparameter_tuning_service import HyperparameterTuningService
from services.preprocessing_service import PreprocessingService
from utils.data_loader import DataLoader


# ==========================================================
# Constants
# ==========================================================

DATASET_PATH = "data/raw/customer_churn_dataset.csv"

REPORTS_DIRECTORY = "reports"

MODELS = {

    "Logistic Regression": (

        LogisticRegression(

            random_state=42,

            max_iter=1000

        ),

        {

            "C": [

                0.01,
                0.1,
                1,
                10,
                100

            ],

            "solver": [

                "lbfgs"

            ]

        }

    ),

    "Decision Tree": (

        DecisionTreeClassifier(

            random_state=42

        ),

        {

            "criterion": [

                "gini",
                "entropy"

            ],

            "max_depth": [

                5,
                10,
                15,
                20,
                None

            ],

            "min_samples_split": [

                2,
                5,
                10

            ],

            "min_samples_leaf": [

                1,
                2,
                4

            ]

        }

    ),

    "Random Forest": (

        RandomForestClassifier(

            random_state=42

        ),

        {

            "n_estimators": [

                100,
                200

            ],

            "max_depth": [

                10,
                20,
                None

            ],

            "min_samples_split": [

                2,
                5

            ],

            "min_samples_leaf": [

                1,
                2

            ]

        }

    ),

    "K-Nearest Neighbors": (

        KNeighborsClassifier(),

        {

            "n_neighbors": [

                3,
                5,
                7,
                9,
                11

            ],

            "weights": [

                "uniform",
                "distance"

            ],

            "metric": [

                "euclidean",
                "manhattan"

            ]

        }

    )

}


# ==========================================================
# Helper Function
# ==========================================================

def print_section(title):

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ==========================================================
# Main
# ==========================================================

def main():

    os.makedirs(

        REPORTS_DIRECTORY,

        exist_ok=True

    )

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
    # Hyperparameter Tuning
    # ------------------------------------------------------

    tuning_service = HyperparameterTuningService()

    summary = []

    print_section("HYPERPARAMETER TUNING")

    for model_name, (model, parameter_grid) in MODELS.items():

        print(f"\n{model_name}")

        print("-" * 70)

        results = tuning_service.tune(

            model,

            parameter_grid,

            X_train,

            y_train

        )

        print("\nBest Parameters")

        print(results["best_parameters"])

        print("\nBest Cross Validation Accuracy")

        print(f"{results['best_score']:.6f}")

        results["cv_results"].to_csv(

            os.path.join(

                REPORTS_DIRECTORY,

                f"{model_name.replace(' ', '_')}_grid_search.csv"

            ),

            index=False

        )

        summary.append({

            "Model": model_name,

            "Best Accuracy": round(

                results["best_score"],

                6

            ),

            "Best Parameters": str(

                results["best_parameters"]

            )

        })

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    summary = pd.DataFrame(summary)

    print_section("FINAL SUMMARY")

    print(summary)

    summary.to_csv(

        os.path.join(

            REPORTS_DIRECTORY,

            "hyperparameter_tuning_summary.csv"

        ),

        index=False

    )

    # ------------------------------------------------------
    # Best Model
    # ------------------------------------------------------

    best_model = summary.loc[

        summary["Best Accuracy"].idxmax()

    ]

    print_section("BEST MODEL")

    print(best_model)


if __name__ == "__main__":

    main()