import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import joblib

# Step 2: Load dataset
df = pd.read_csv("data/diabetes.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# Step 3: Split dataset into train/test sets
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Step 4: Train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

print("Model training complete.")

# Step 5: Evaluate the Model
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Step 6: Save the trained model
joblib.dump(log_reg, "models/model.pkl")
print("Model saved successfully as models/model.pkl")


rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

print("Random Forest training complete.")

# Evaluate Random Forest
y_pred_rf = rf_clf.predict(X_test)
y_prob_rf = rf_clf.predict_proba(X_test)[:,1]

print("\n--- Random Forest Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

joblib.dump(rf_clf, "models/random_forest.pkl")
print("Random Forest model saved successfully as models/random_forest.pkl")

#  Train XGBoost model
xgb_clf = XGBClassifier(
    n_estimator=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_clf.fit(X_train, y_train)

print("XGBoost training complete.")

# Evaluate XGBoost
y_pred_xgb = xgb_clf.predict(X_test)
y_prob_xgb = xgb_clf.predict_proba(X_test)[:,1]

print("\n--- XGBoost Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

joblib.dump(xgb_clf, "models/xgboost.pkl")
print("XGBoost model saved successfully as models/xgboost.pkl")

# Compare all models side by side

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],
    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_xgb)
    ],
    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_xgb)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_prob),
        roc_auc_score(y_test, y_prob_rf),
        roc_auc_score(y_test, y_prob_xgb)
    ]
})

print("\n--- Model Comparison ---")
print(results)

import matplotlib.pyplot as plt

# Plot comparison
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

plt.figure(figsize=(12,6))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    plt.bar(results["Model"], results[metric], color=["skyblue", "lightgreen", "salmon"])
    plt.title(metric)
    plt.ylabel("Score")
    plt.ylim(0,1)

plt.tight_layout()
plt.show()

# Automatically select the best model based on Recall (priority in healthcare)
best_metric = "Recall"   # best_metric = "ROC-AUC"  /if you want ROC-AUC instead,just change

# Find the model with highest Recall
best_index = results[best_metric].idxmax()
best_model_name = results.loc[best_index, "Model"]

print(f"\nBest model based on {best_metric}: {best_model_name}")

# Map model names to actual objects
model_map = {
    "Logistic Regression": log_reg,
    "Random Forest": rf_clf,
    "XGBoost": xgb_clf
}

# Save only the best model for deployment
best_model = model_map[best_model_name]
joblib.dump(best_model, "models/best_model.pkl")

print("Best model saved successfully as models/best_model.pkl")
