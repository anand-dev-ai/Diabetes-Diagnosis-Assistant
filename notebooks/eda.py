import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Basic exploration
print("Dataset shape:", df.shape)
print("Column names:", df.columns)
print(df.head())

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Summary statistics
print(df.describe())

# Visualize glucose distribution
plt.figure(figsize=(10,6))
sns.histplot(df['Glucose'], bins=30, kde=True)
plt.title("Glucose Level Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(method="pearson"), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for col in cols_with_zeros:
    df[col] = df[col].replace(0,df[col].median())

print(df[cols_with_zeros].describe())

plt.figure(figsize=(10,6))
sns.histplot(df['Glucose'], bins=30, kde=True)
plt.title("Cleaned Glucose Distribution")
plt.show()

sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Outcome Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Outcome', y='Glucose', data=df)
plt.title("Glucose Levels vs Diabetes Outcome")
plt.show()

corr = df.corr()['Outcome'].sort_values(ascending=False)
print(corr)

sns.pairplot(df, hue="Outcome")
plt.show()
