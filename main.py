from src.loader import load_dataset
from src.validator import validate_dataset
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.preprocessor import split_features_target, split_train_test, encode_features
from src.visualizer import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_confusion_matrix
)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# =========================
# LOAD DATASET
# =========================

while True:

    file_name = input("Enter dataset name: ")

    df = load_dataset(f"data/{file_name}")

    if df is not None:
        break

    print("Please enter a valid dataset.")


# =========================
# VALIDATE DATASET
# =========================

report = validate_dataset(df)

print("\n=== DATA VALIDATION REPORT ===")
print(f"Missing Values: {report['missing_values']}")
print(f"Duplicates: {report['duplicates']}")
print(f"Valid Dataset: {report['is_valid']}")
print(f"Number of rows: {report['rows']}")
print(f"Number of columns: {report['columns']}")

if not report["is_valid"]:
    print("\nDataset is not ready for modeling.")
    exit()


# =========================
# SELECT TARGET COLUMN
# =========================

print("\nAvailable Columns:")
for column in df.columns:
    print(f"- {column}")

while True:

    target_column = input("Enter target column: ")

    if target_column in df.columns:
        break

    print("Error: Target column not found.")
  
print("============================")
print("SELECT MACHINE LEARNING TASK")
print("============================")
print("1. Regression")
print("2. Classification")

while True:
    task = input("Enter your choice number: ")
    if task == "1":
        model = LinearRegression()
        break
    elif task == "2":
        model = LogisticRegression()
        break
    else:
        print("Error: (Enter 1 or 2) ") 
X, y = split_features_target(df, target_column)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = split_train_test(X, y)
preprocessor, X_train_processed, X_test_processed = encode_features(
    X_train,
    X_test
)
 
# =========================
# TRAIN MODEL
# =========================
model = train_model(
    model,
    X_train_processed,
    y_train
)

y_pred = model.predict(X_test_processed)



print("\nModel trained successfully.")
print(f"\nTraining Rows: {len(X_train)}")

# =========================
# EVALUATE MODEL
# =========================

evaluation_report = evaluate_model(
    task,
    y_test,
    y_pred
)
if task == "1":
    print("=== REGRESSION EVALUATION ===")
    print(f"MAE: {evaluation_report['mae']:.4f}")
    print(f"RMSE: {evaluation_report['rmse']:.4f}")
    print(f"R² Score: {evaluation_report['r2']:.4f}")

    plot_actual_vs_predicted(
    y_test,
    y_pred
    )
    plot_residuals(
        y_test,
        y_pred
    )


elif task == "2":
    print("=== CLASSIFICATION EVALUATION ===")
    print(f"Accuracy : {evaluation_report['accuracy']:.4f}")
    print(f"Precision: {evaluation_report['precision']:.4f}")
    print(f"Recall   : {evaluation_report['recall']:.4f}")
    print(f"F1 Score : {evaluation_report['f1']:.4f}")
    plot_confusion_matrix(
        y_test,
        y_pred
    )
