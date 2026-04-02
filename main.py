from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import time

from services.pi_client import fetch_sensor_data
from ml.model_loader import load_models
from ml.predictor import run_prediction

app = FastAPI()

# ===================== CORS =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== SHARED STATE =====================
latest_pi_data = {
    "data": None,
    "last_updated": None,
    "status": "starting"
}

models = {}

# ===================== POLL PI LOOP =====================
def poll_pi_loop():
    global latest_pi_data

    while True:
        try:
            data = fetch_sensor_data()
            latest_pi_data["data"] = data
            latest_pi_data["last_updated"] = time.time()
            latest_pi_data["status"] = "ok"
            print("Pulled data from Raspberry Pi")

        except Exception as e:
            latest_pi_data["status"] = "error"
            print("Failed to fetch from Pi:", e)

        time.sleep(5)

# ===================== STARTUP =====================
@app.on_event("startup")
def startup():
    global models
    models = load_models()
    threading.Thread(target=poll_pi_loop, daemon=True).start()
    print("Backend started")

# ===================== PREDICT API =====================
@app.get("/api/predict")
def predict(soil_type: str):
    try:
        # clean input
        soil_type = soil_type.strip().lower()

        if latest_pi_data["data"] is None:
            return {"error": "Sensor data not ready"}

        prediction = run_prediction(
            latest_pi_data["data"],
            soil_type,
            models
        )

        return {
            "sensor_data": latest_pi_data["data"],
            "prediction": prediction
        }

    except Exception as e:
        return {"error": str(e)}