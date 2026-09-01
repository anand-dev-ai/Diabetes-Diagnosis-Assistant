import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# --- Exploration ---
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print("Missing values:\n", df.isnull().sum())
print(df.describe())

# --- Visualization before cleaning ---
plt.figure(figsize=(10,6))
sns.histplot(df['Glucose'], bins=30, kde=True)
plt.title("Glucose Level Distribution (Raw)")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# --- Cleaning ---
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].median())

print(df[cols_with_zeros].describe())

# --- Visualization after cleaning ---
plt.figure(figsize=(10,6))
sns.histplot(df['Glucose'], bins=30, kde=True)
plt.title("Glucose Level Distribution (Cleaned)")
plt.show()

sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Outcome Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Outcome', y='Glucose', data=df)
plt.title("Glucose Levels vs Diabetes Outcome")
plt.show()

# --- Correlation ---
corr = df.corr()['Outcome'].sort_values(ascending=False)
print("Correlation with Outcome:\n", corr)

sns.pairplot(df, hue="Outcome")
plt.show()
