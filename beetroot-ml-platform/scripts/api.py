# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as rt

app = FastAPI(title="Beetroot Root Yield Prediction API")

# ---- 1. Load ONNX model at startup ----
ONNX_PATH = "root_yield_model.onnx"
sess = rt.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# ---- 2. Define feature order exactly as in training ----
FEATURE_ORDER = [
    "dps",
    "soil_moisture_30",
    "soil_moisture_60",
    "soil_moisture_90",
    "PAW_30",
    "PAW_60",
    "PAW_90",
    "LAI",
    "min_temp",
    "max_temp",
    "av_temp",
    "precipitation",
    "glob_radiation",
    "ET_grass",
    "irrigation_binary",
]

# ---- 3. Define request body schema ----
class RootYieldRequest(BaseModel):
    dps: float
    irrigation: str  # "yes" or "no"
    soil_moisture_30: float
    soil_moisture_60: float
    soil_moisture_90: float
    PAW_30: float
    PAW_60: float
    PAW_90: float
    LAI: float
    min_temp: float
    max_temp: float
    av_temp: float
    precipitation: float
    glob_radiation: float
    ET_grass: float

class RootYieldResponse(BaseModel):
    root_yield_prediction: float

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Beetroot root_yield ONNX API running"}

@app.post("/predict", response_model=RootYieldResponse)
def predict_root_yield(req: RootYieldRequest):
    # ---- 4. Convert irrigation -> binary ----
    irr = req.irrigation.strip().lower()
    if irr not in ("yes", "no"):
        raise ValueError("irrigation must be 'yes' or 'no'")
    irrigation_binary = 1.0 if irr == "yes" else 0.0

    # ---- 5. Build feature vector in correct order ----
    feature_values = [
        req.dps,
        req.soil_moisture_30,
        req.soil_moisture_60,
        req.soil_moisture_90,
        req.PAW_30,
        req.PAW_60,
        req.PAW_90,
        req.LAI,
        req.min_temp,
        req.max_temp,
        req.av_temp,
        req.precipitation,
        req.glob_radiation,
        req.ET_grass,
        irrigation_binary,
    ]

    arr = np.array([feature_values], dtype=np.float32)  # shape (1, n_features)

    # ---- 6. Run ONNX inference ----
    pred = sess.run([output_name], {input_name: arr})[0][0]

    return RootYieldResponse(root_yield_prediction=float(pred))
