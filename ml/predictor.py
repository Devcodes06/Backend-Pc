import numpy as np

def run_prediction(sensor_data, soil_type, models):
    try:
        # ===================== 1. CLEAN INPUT =====================
        soil_type = soil_type.strip().lower()

        # ===================== 2. EXTRACT RAW VALUES =====================
        features = {
            "nitrogen": sensor_data["npk"]["nitrogen"],
            "phosphorus": sensor_data["npk"]["phosphorus"],
            "potassium": sensor_data["npk"]["potassium"],
            "temperature": sensor_data["soil"]["temperature"],
            "moisture": sensor_data["soil"]["moisture"],
            "ph": sensor_data["soil"]["ph"],
            "soil_type": 0  # dummy value
        }

        # ===================== 3. ORDER FEATURES =====================
        ordered_features = [features[col] for col in models["feature_columns"]]

        # ===================== 4. SCALE =====================
        X = models["scaler"].transform([ordered_features])

        # ===================== 5. CROP PREDICTION =====================
        crop_probs = models["crop_classifier"].predict_proba(X)[0]
        top_indices = np.argsort(crop_probs)[::-1][:3]

        crops = models["crop_encoder"].inverse_transform(top_indices)

        crop_results = [
            {
                "crop": crops[i],
                # ✅ percentage + 3 decimal places
                "probability": round(float(crop_probs[top_indices[i]] * 100), 3)
            }
            for i in range(len(top_indices))
        ]

        # ===================== 6. YIELD PREDICTION =====================
        yield_value = float(models["yield_regressor"].predict(X)[0])

        # ✅ limit to 3 decimal places
        yield_value = round(yield_value, 3)

        return {
            "top_crops": crop_results,
            "expected_yield": yield_value
        }

    except Exception as e:
        return {"error": str(e)}