from src.loader import load_data, validate_data
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.visualizer import (
    plot_actual_vs_predicted,
    plot_residuals
)
from sklearn.model_selection import train_test_split


# =========================
# LOAD DATASET
# =========================

while True:

    file_name = input("Enter dataset name: ")

    df = load_data(f"data/{file_name}")

    if df is not None:
        break

    print("Please enter a valid dataset.")


# =========================
# VALIDATE DATASET
# =========================

report = validate_data(df)

print("\n=== DATA VALIDATION REPORT ===")
print(f"Missing Values: {report['missing_values']}")
print(f"Duplicates: {report['duplicates']}")
print(f"Valid Dataset: {report['is_valid']}")

if not report["is_valid"]:
    print("\nDataset is not ready for modeling.")
    exit()


# =========================
# SELECT TARGET COLUMN
# =========================

feature_count = len(df.columns) - 1
print("\nDataset Information")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

while True:

    target_column = input("Enter target column: ")

    if target_column in df.columns:
        break

    print("Error: Target column not found.")


# =========================
# PREPARE FEATURES/TARGET
# =========================

X = df.drop(columns=[target_column])
y = df[target_column]


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

while True:
    print("\nChoose Model:")
    print("1. Linear Regression")
    print("2. Decision Tree")
    print("3. Random Forest")

    model_choice = input("Enter choice: ")

    if model_choice in ["1", "2", "3"]:
        break

    print("Invalid choice. Please enter 1, 2, or 3.")
# =========================
# TRAIN MODEL
# =========================

model, y_pred, model_name = train_model(
    model_choice,
    X_train,
    X_test,
    y_train
)
print("\nModel trained successfully.")
print(f"\nSelected Model: {model_name}")
print(f"\nTraining Rows: {len(X_train)}")
print(f"Testing Rows: {len(X_test)}")

print("\nFirst 5 Predictions:")
print(y_pred[:5])


# =========================
# EVALUATE MODEL
# =========================

evaluation_report = evaluate_model(
    y_test,
    y_pred
)

print("\n=== MODEL EVALUATION ===")
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