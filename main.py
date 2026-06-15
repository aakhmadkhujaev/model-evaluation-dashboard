

from src.loader import load_data, validate_data

file_name = input("Enter dataset name: ")
df = load_data(f"data/{file_name}")

if df is not None:

    report = validate_data(df)

    print("\n=== DATA VALIDATION REPORT ===")
    print(f"Missing Values: {report['missing_values']}")
    print(f"Duplicates: {report['duplicates']}")
    print(f"Valid Dataset: {report['is_valid']}")