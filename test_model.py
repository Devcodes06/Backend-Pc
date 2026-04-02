import joblib
import numpy as np

# Load models
scaler = joblib.load("scaler.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_classifier = joblib.load("crop_classifier.pkl")
yield_regressor = joblib.load("yield_regressor.pkl")
feature_columns = joblib.load("feature_columns.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")

# Example sensor values
sensor_data = {
    "nitrogen": 50,
    "phosphorus": 40,
    "potassium": 30,
    "temperature": 25,
    "moisture": 60,
    "ph": 6.5
}

def test_soil(soil_type):
    soil_encoded = soil_encoder.transform([soil_type])[0]

    features = [
        sensor_data["nitrogen"],
        sensor_data["phosphorus"],
        sensor_data["potassium"],
        sensor_data["temperature"],
        sensor_data["moisture"],
        sensor_data["ph"],
        soil_encoded
    ]

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    crop_probs = crop_classifier.predict_proba(X_scaled)[0]
    top_index = crop_probs.argmax()

    predicted_crop = crop_encoder.inverse_transform([top_index])[0]
    predicted_yield = yield_regressor.predict(X_scaled)[0]

    print("Soil:", soil_type)
    print("Predicted Crop:", predicted_crop)
    print("Yield:", predicted_yield)
    print("------")

# Test different soils
test_soil("Loamy")
test_soil("Clay")
test_soil("Sandy")
