import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path):
    df = pd.read_csv(path)
    df.drop(['customerID'], axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

def preprocess(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns

# Churn Count Plot
def plot_churn_distribution(df):
    sns.countplot(data=df, x='Churn')
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Number of Customers")
    plt.show()

# Monthly Charges vs Churn
def plot_monthly_charges_vs_churn(df):
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
    plt.title("Monthly Charges vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.show()

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred)}")

    joblib.dump(model, "churn_model.pkl")
    print("✅ Model saved as churn_model.pkl")

def main():
    data_path = "data/telco_churn.csv"
    if not os.path.exists(data_path):
        print("❌ Dataset not found.")
        return

    df = load_data(data_path)
    X, y, scaler, feature_names = preprocess(df)

    plot_churn_distribution(df)
    plot_monthly_charges_vs_churn(df)

    train_model(X, y)
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_names.tolist(), "features.pkl")
    print("✅ Scaler and feature names saved.")

    print("🔎 Expected feature order:")
    print(feature_names.tolist())

if __name__ == "__main__":
    main()
