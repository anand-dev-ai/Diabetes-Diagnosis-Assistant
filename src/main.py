from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

# Load model
model = joblib.load("models/best_model.pkl")

app = FastAPI(title="Diabetes Diagnosis Assistant")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://diabetes-diagnosis-assistant.onrender.com"],  # your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve index.html
@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

# Prediction endpoint
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
        "prediction": int(prediction),       # ✅ convert to Python int
        "probability": round(float(probability), 3)  # ✅ convert to Python float
    }

