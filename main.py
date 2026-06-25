from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from src.loader import load_dataset
from src.validator import validate_dataset
from src.preprocessor import (
    split_features_target,
    split_train_test,
    encode_features
)
from src.trainer import train_model
from src.model_saver import save_model
from src.evaluator import (
    evaluate_model,
    generate_classification_report,
    save_classification_report
)
from src.visualizer import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_confusion_matrix
)

def main():
    # ==================================================
    # LOAD DATASET
    # ==================================================

    while True:

        file_name = input("Enter dataset name: ")

        df = load_dataset(f"data/{file_name}")

        if df is not None:
            break

        print("Please enter a valid dataset.")


    # ==================================================
    # VALIDATE DATASET
    # ==================================================

    validation_report = validate_dataset(df)

    print("\n=== DATA VALIDATION REPORT ===")
    print(f"Missing Values   : {validation_report['missing_values']}")
    print(f"Duplicates       : {validation_report['duplicates']}")
    print(f"Valid Dataset    : {validation_report['is_valid']}")
    print(f"Number of Rows   : {validation_report['rows']}")
    print(f"Number of Columns: {validation_report['columns']}")

    if not validation_report["is_valid"]:
        print("\nDataset is not ready for modeling.")
        exit()


    # ==================================================
    # SELECT TARGET COLUMN
    # ==================================================

    print("\nAvailable Columns:")

    for column in df.columns:
        print(f"- {column}")

    while True:

        target_column = input("\nEnter target column: ")

        if target_column in df.columns:
            break

        print("Error: Target column not found.")


    # ==================================================
    # SELECT MACHINE LEARNING TASK
    # ==================================================

    print("\n============================")
    print("SELECT MACHINE LEARNING TASK")
    print("============================")
    print("1. Regression")
    print("2. Classification")

    while True:

        task = input("Enter your choice number: ")

        if task == "1":
            model = LinearRegression()
            model_name = "linear_regression"
            break

        elif task == "2":
            model = LogisticRegression()
            model_name = "logistic_regression"
            break

        print("Error: Enter 1 or 2.")


    # ==================================================
    # PREPARE DATA
    # ==================================================

    X, y = split_features_target(df, target_column)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor, X_train_processed, X_test_processed = encode_features(
        X_train,
        X_test
    )


    # ==================================================
    # TRAIN MODEL
    # ==================================================

    model = train_model(
        model,
        X_train_processed,
        y_train
    )

    print("\nModel trained successfully.")

    save_model(
        model,
        model_name
    )

    y_pred = model.predict(X_test_processed)

    print(f"Training Rows: {len(X_train)}")


    # ==================================================
    # EVALUATE MODEL
    # ==================================================

    evaluation_report = evaluate_model(
        task,
        y_test,
        y_pred
    )


    # ==================================================
    # REGRESSION RESULTS
    # ==================================================

    if task == "1":

        print("\n=== REGRESSION EVALUATION ===")
        print(f"MAE      : {evaluation_report['mae']:.4f}")
        print(f"RMSE     : {evaluation_report['rmse']:.4f}")
        print(f"R² Score : {evaluation_report['r2']:.4f}")

        plot_actual_vs_predicted(
            y_test,
            y_pred
        )

        plot_residuals(
            y_test,
            y_pred
        )


    # ==================================================
    # CLASSIFICATION RESULTS
    # ==================================================

    elif task == "2":

        print("\n=== CLASSIFICATION EVALUATION ===")
        print(f"Accuracy : {evaluation_report['accuracy']:.4f}")
        print(f"Precision: {evaluation_report['precision']:.4f}")
        print(f"Recall   : {evaluation_report['recall']:.4f}")
        print(f"F1 Score : {evaluation_report['f1']:.4f}")

        classification_report = generate_classification_report(
            y_test,
            y_pred
        )

        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report)

        save_classification_report(
            classification_report
        )

        plot_confusion_matrix(
            y_test,
            y_pred
        )
if __name__ == "__main__":
    while True:
        main()

        while True:

            print("\n====================================")
            print("ANALYZE ANOTHER DATASET")
            print("====================================")
            print("1. Analyze another dataset")
            print("2. Exit")

            choice = input("Enter your choice: ")

            if choice == "1":
                break

            elif choice == "2":
                print("\nThank you for using Model Evaluation Dashboard!")
                exit()

            else:
                print("Invalid choice. Please enter 1 or 2.")