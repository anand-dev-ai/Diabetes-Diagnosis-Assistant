from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

# Load model
model = joblib.load("models/best_model.pkl")

app = FastAPI(title="Diabetes Diagnosis Assistant")

# Enable CORS (must include OPTIONS preflight)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://diabetes-diagnosis-assistant.onrender.com",  # frontend domain
        "http://localhost:5500"  # optional: for local testing
    ],
    allow_credentials=True,
    allow_methods=["*"],   # includes OPTIONS
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

@app.post("/predict")
def predict_diabetes(data: dict):
    input_data = np.array([[
        data["Pregnancies"],
        data["Glucose"],
        data["BloodPressure"],
        data["SkinThickness"],
        data["Insulin"],
        data["BMI"],
        data["DiabetesPedigreeFunction"],
        data["Age"]
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 3)
    }
