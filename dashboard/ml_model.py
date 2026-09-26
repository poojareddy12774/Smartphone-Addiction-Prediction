import joblib
import os
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, "model/stacking_model.pkl")
SCALER_PATH = os.path.join(settings.BASE_DIR, "model/scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class_map = {
    0: "High",
    1: "Low",
    2: "Medium"
}