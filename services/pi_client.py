import requests

PI_BASE_URL = "http://10.177.163.28:8000/api/sensor-data"

def fetch_sensor_data():
    response = requests.get(PI_BASE_URL, timeout=3)
    response.raise_for_status()
    return response.json()
