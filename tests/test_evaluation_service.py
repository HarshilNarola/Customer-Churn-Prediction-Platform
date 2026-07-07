from utils.data_loader import DataLoader

from services.preprocessing_service import PreprocessingService
from services.model_training_service import ModelTrainingService
from services.evaluation_service import EvaluationService


DATASET_PATH = "data/raw/customer_churn_dataset.csv"


def main():

    # ==========================================================
    # Load Dataset
    # ==========================================================

    loader = DataLoader()

    df = loader.load_data(DATASET_PATH)

    # ==========================================================
    # Preprocess Dataset
    # ==========================================================

    preprocessing = PreprocessingService()

    X_train, X_test, y_train, y_test = preprocessing.preprocess(df)

    feature_names = X_train.columns

    # ==========================================================
    # Initialize Services
    # ==========================================================

    trainer = ModelTrainingService()

    evaluator = EvaluationService()

    # ==========================================================
    # Logistic Regression
    # ==========================================================

    logistic_model, logistic_results = trainer.train_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluator.add_result(
        "Logistic Regression",
        logistic_results
    )

    evaluator.plot_confusion_matrix(
        logistic_model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    evaluator.plot_roc_curve(
        logistic_model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    # ==========================================================
    # Decision Tree
    # ==========================================================

    decision_tree_model, decision_tree_results = trainer.train_decision_tree(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluator.add_result(
        "Decision Tree",
        decision_tree_results
    )

    evaluator.plot_confusion_matrix(
        decision_tree_model,
        X_test,
        y_test,
        "Decision Tree"
    )

    evaluator.plot_roc_curve(
        decision_tree_model,
        X_test,
        y_test,
        "Decision Tree"
    )

    evaluator.plot_feature_importance(
        decision_tree_model,
        feature_names,
        "Decision Tree"
    )

    # ==========================================================
    # Random Forest
    # ==========================================================

    random_forest_model, random_forest_results = trainer.train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluator.add_result(
        "Random Forest",
        random_forest_results
    )

    evaluator.plot_confusion_matrix(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest"
    )

    evaluator.plot_roc_curve(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest"
    )

    evaluator.plot_feature_importance(
        random_forest_model,
        feature_names,
        "Random Forest"
    )

    # ==========================================================
    # KNN
    # ==========================================================

    knn_model, knn_results = trainer.train_knn(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluator.add_result(
        "KNN",
        knn_results
    )

    evaluator.plot_confusion_matrix(
        knn_model,
        X_test,
        y_test,
        "KNN"
    )

    evaluator.plot_roc_curve(
        knn_model,
        X_test,
        y_test,
        "KNN"
    )

    # ==========================================================
    # Gaussian Naive Bayes
    # ==========================================================

    naive_model, naive_results = trainer.train_naive_bayes(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluator.add_result(
        "Gaussian Naive Bayes",
        naive_results
    )

    evaluator.plot_confusion_matrix(
        naive_model,
        X_test,
        y_test,
        "Gaussian Naive Bayes"
    )

    evaluator.plot_roc_curve(
        naive_model,
        X_test,
        y_test,
        "Gaussian Naive Bayes"
    )

    # ==========================================================
    # Support Vector Machine
    # ==========================================================

    svm_model, svm_results = trainer.train_svm(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluator.add_result(
        "Support Vector Machine",
        svm_results
    )

    evaluator.plot_confusion_matrix(
        svm_model,
        X_test,
        y_test,
        "Support Vector Machine"
    )

    evaluator.plot_roc_curve(
        svm_model,
        X_test,
        y_test,
        "Support Vector Machine"
    )

    # ==========================================================
    # Print Comparison
    # ==========================================================

    evaluator.print_comparison()

    # ==========================================================
    # Save Reports
    # ==========================================================

    evaluator.save_csv()

    evaluator.save_markdown()

    # ==========================================================
    # Best Model
    # ==========================================================

    best = evaluator.get_best_model()

    model_mapping = {

        "Logistic Regression": logistic_model,

        "Decision Tree": decision_tree_model,

        "Random Forest": random_forest_model,

        "KNN": knn_model,

        "Gaussian Naive Bayes": naive_model,

        "Support Vector Machine": svm_model

    }

    best_model = model_mapping[best["Model"]]

    evaluator.save_best_model(
        best_model
    )

    print()

    print("=" * 80)
    print("BEST MODEL")
    print("=" * 80)

    print(best)


if __name__ == "__main__":
    main()