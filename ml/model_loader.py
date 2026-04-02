import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

models = {}

def load_models():
    global models

    models["crop_classifier"] = joblib.load(os.path.join(MODEL_DIR, "crop_classifier.pkl"))
    models["yield_regressor"] = joblib.load(os.path.join(MODEL_DIR, "yield_regressor.pkl"))
    models["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    models["soil_encoder"] = joblib.load(os.path.join(MODEL_DIR, "soil_encoder.pkl"))
    models["crop_encoder"] = joblib.load(os.path.join(MODEL_DIR, "crop_encoder.pkl"))
    models["feature_columns"] = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

    print("All ML models loaded")

    return models
