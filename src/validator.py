def validate_dataset(df):
    missing_values = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    is_valid = (
    missing_values == 0
    and duplicates == 0
    )
    rows = len(df)
    columns = len(df.columns)
    validation_report = {
    "rows": rows,
    "columns": columns,
    "missing_values": missing_values,
    "duplicates": duplicates,
    "is_valid": is_valid
    }
    return validation_report