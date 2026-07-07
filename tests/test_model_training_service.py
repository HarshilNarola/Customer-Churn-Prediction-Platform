from utils.data_loader import DataLoader
from services.preprocessing_service import PreprocessingService
from services.model_training_service import ModelTrainingService


DATASET_PATH = "data/raw/customer_churn_dataset.csv"


def print_results(model_name, results):

    print()
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1_score']:.4f}")
    print(f"ROC AUC   : {results['roc_auc']:.4f}")

    print("\nConfusion Matrix")
    print(results["confusion_matrix"])

    print("\nClassification Report")
    print(results["classification_report"])


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

    # ==========================================================
    # Initialize Trainer
    # ==========================================================

    trainer = ModelTrainingService()

    # ==========================================================
    # Logistic Regression
    # ==========================================================

    logistic_model, logistic_results = (
        trainer.train_logistic_regression(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    trainer.save_model(
        logistic_model,
        "logistic_regression.pkl"
    )

    print_results(
        "Logistic Regression Results",
        logistic_results
    )

    # ==========================================================
    # Decision Tree
    # ==========================================================

    decision_tree_model, decision_tree_results = (
        trainer.train_decision_tree(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    trainer.save_model(
        decision_tree_model,
        "decision_tree.pkl"
    )

    print_results(
        "Decision Tree Results",
        decision_tree_results
    )

    # ==========================================================
    # Random Forest
    # ==========================================================

    random_forest_model, random_forest_results = (
        trainer.train_random_forest(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    trainer.save_model(
        random_forest_model,
        "random_forest.pkl"
    )

    print_results(
        "Random Forest Results",
        random_forest_results
    )

    # ==========================================================
    # KNN
    # ==========================================================

    knn_model, knn_results = (
        trainer.train_knn(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    trainer.save_model(
        knn_model,
        "knn.pkl"
    )

    print_results(
        "KNN Results",
        knn_results
    )

    # ==========================================================
    # Gaussian Naive Bayes
    # ==========================================================

    naive_model, naive_results = (
        trainer.train_naive_bayes(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    trainer.save_model(
        naive_model,
        "naive_bayes.pkl"
    )

    print_results(
        "Gaussian Naive Bayes Results",
        naive_results
    )

    # ==========================================================
    # Support Vector Machine
    # ==========================================================

    svm_model, svm_results = (
        trainer.train_svm(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    trainer.save_model(
        svm_model,
        "svm.pkl"
    )

    print_results(
        "Support Vector Machine Results",
        svm_results
    )


if __name__ == "__main__":
    main()