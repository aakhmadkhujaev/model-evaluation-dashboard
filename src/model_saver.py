from pathlib import Path
import joblib

OUTPUTS_DIR = Path("outputs")
MODELS_DIR = OUTPUTS_DIR / "models"
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def save_model(model, model_name):
    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved: {model_path}")

def load_model(model_name):

    model_path = MODELS_DIR / f"{model_name}.pkl"

    model = joblib.load(model_path)
    print(f"Loaded: {model_path}")

    return model