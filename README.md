# 🩺 Diabetes Diagnosis Assistant

A machine learning project that predicts the likelihood of diabetes using patient health data.  
Built with **Logistic Regression, Random Forest, and XGBoost**, deployed via **FastAPI** and a custom **HTML/CSS/JS frontend**.

---

## 📂 Project Structure

data/diabetes.csv          # Dataset
notebooks/eda.ipynb        # Exploratory Data Analysis
src/train_model.py         # Model training & evaluation
src/main.py                # FastAPI backend + frontend serving
models/best_model.pkl      # Best model saved for deployment
frontend/index.html        # Web UI
frontend/style.css         # Styling
frontend/script.js         # JS logic
requirements.txt           # Dependencies
README.md                  # Documentation

Code

---

## 🚀 Features
- Cleaned dataset with EDA.
- Baseline (Logistic Regression) + advanced models (Random Forest, XGBoost).
- Automatic model comparison and best model selection.
- FastAPI backend serving predictions.
- Professional HTML/CSS/JS frontend with validation and probability bar.

---

## ⚙️ Installation

```bash
git clone https://github.com/anand-dev-ai/Diabetes-Diagnosis-Assistant.git
cd Diabetes-Diagnosis-Assistant
pip install -r requirements.txt
📊 Training Models
bash
python src/train_model.py
This will:

Train models and evaluate metrics.

Save all models in models/.

Auto‑select best model (best_model.pkl).

🌐 Run FastAPI + Frontend
bash
uvicorn src.main:app --reload
Open in browser:

Code
http://127.0.0.1:8000/
Fill out patient details in the form → get prediction + probability bar.

📦 Requirements
Python 3.9+

pandas, numpy, scikit-learn, xgboost

fastapi, uvicorn

joblib