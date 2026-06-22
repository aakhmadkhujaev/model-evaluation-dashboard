from src.loader import load_dataset
from src.validator import validate_dataset
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.preprocessor import split_features_target, split_train_test, encode_features
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
# 
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
