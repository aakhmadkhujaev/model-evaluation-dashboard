import pandas as pd
from pathlib import Path

def load_dataset(file_path):
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


