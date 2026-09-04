🩺 Diabetes Diagnosis Assistant

A machine learning web application that predicts the likelihood of diabetes from patient health measurements.

The project compares multiple machine learning algorithms, automatically selects the best-performing model, and serves predictions through a FastAPI backend with a responsive HTML/CSS/JavaScript frontend.

⚠️ Disclaimer: This project is intended for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice.

🌐 Live Demo

Try the application:
https://diabetes-assistant.onrender.com

✨ Features
🧹 Data preprocessing and exploratory data analysis
📊 Exploratory Data Analysis (EDA)
🤖 Multiple machine learning models:
Logistic Regression
Random Forest
XGBoost
🏆 Automatic model comparison and best-model selection
💾 Best-performing model saved using Joblib
⚡ FastAPI REST API for real-time predictions
🎨 Custom HTML/CSS/JavaScript frontend
📈 Diabetes probability visualization
✅ Client-side form validation
📱 Responsive interface for desktop and mobile
☁️ Deployed on Render
🧠 Machine Learning Workflow

The project follows this workflow:

Patient Health Data
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Train Multiple Models
        ↓
Evaluate Model Performance
        ↓
Select Best Model
        ↓
Save best_model.pkl
        ↓
FastAPI Prediction API
        ↓
Web Frontend
        ↓
Prediction + Probability

📂 Project Structure
Diabetes-Diagnosis-Assistant/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── models/
│   └── best_model.pkl
│
├── src/
│   ├── train_model.py
│   └── main.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── requirements.txt
└── README.md

🛠️ Tech Stack
Machine Learning
Python
Pandas
NumPy
Scikit-learn
XGBoost
Joblib
Backend
FastAPI
Uvicorn
Frontend
HTML5
CSS3
JavaScript
Deployment
Render
📊 Models

Three classification algorithms are evaluated:

Model	Purpose
Logistic Regression	Baseline classification model
Random Forest	Ensemble learning approach
XGBoost	Gradient boosting approach

The training pipeline evaluates the models and saves the selected best-performing model as:

models/best_model.pkl

⚙️ Installation
1. Clone the repository
git clone https://github.com/anand-dev-ai/Diabetes-Diagnosis-Assistant.git

2. Enter the project directory
cd Diabetes-Diagnosis-Assistant

3. Install dependencies
pip install -r requirements.txt

🧪 Train the Models

Run:

python src/train_model.py


The training process:

Loads the diabetes dataset.
Preprocesses the data.
Trains multiple classification models.
Evaluates model performance.
Compares the models.
Selects the best-performing model.
Saves the model to:
models/best_model.pkl

🚀 Run the Application Locally

Start the FastAPI server:

uvicorn src.main:app --reload


Open the application in your browser:

http://127.0.0.1:8000/


Enter the patient's health information and click Predict to receive a prediction and probability score.

🔌 API
Prediction Endpoint
POST /predict

Input

The API expects JSON containing:

{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 79,
  "BMI": 25.0,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 35
}

Response
{
  "prediction": 0,
  "probability": 0.123
}


Where:

prediction: 0 → model predicts a negative class
prediction: 1 → model predicts a positive class
probability → model's estimated probability for the positive class
❤️ Health Check

The backend provides a health-check endpoint:

GET /health


Example response:

{
  "status": "healthy",
  "message": "Diabetes Assistant API is running"
}

🔒 Important Limitations

Machine learning predictions depend on the quality and characteristics of the training data.

This application:

Does not replace a doctor or healthcare professional.
Should not be used to make medical decisions.
May produce incorrect predictions.
Provides model-based estimates rather than confirmed diagnoses.
🚧 Future Improvements

Possible future improvements include:

Add authentication and API rate limiting
Improve input validation
Add model performance visualizations
Add confusion matrix and ROC-AUC analysis
Add automated testing
Add CI/CD
Improve accessibility
Add prediction history
Improve API documentation
Add Docker support
Monitor model performance after deployment
👨‍💻 Author

Anand Samadhiya

GitHub:
https://github.com/anand-dev-ai

⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.