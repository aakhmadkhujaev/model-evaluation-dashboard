import pandas as pd
from pathlib import Path

def load_data(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"Error: File not found -> {path}")
        return None

    try:
        df = pd.read_csv(path)
        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def validate_data(df):
    missing_values = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    is_valid = (
    missing_values == 0
    and duplicates == 0
    )
    validation_report = {
        "missing_values" : missing_values,
        "duplicates" : duplicates,
        "is_valid" : is_valid  
    }
    return validation_report