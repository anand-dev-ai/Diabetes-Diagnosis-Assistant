from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

# Load model
model = joblib.load("models/best_model.pkl")

app = FastAPI(title="Diabetes Diagnosis Assistant")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://diabetes-diagnosis-assistant.onrender.com",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)

# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "Diabetes Assistant API is running"
    }

# Home page
@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

# Prediction
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
