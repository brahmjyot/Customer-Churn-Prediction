import joblib
import numpy as np
import argparse
import os

def predict_user(input_features):
    if not os.path.exists("churn_model.pkl") or not os.path.exists("scaler.pkl"):
        print("❌ Model or scaler not found. Please run main.py first to train.")
        return

    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("features.pkl")

    print(f"🧾 Using feature order: {feature_names}")

    if len(input_features) != len(feature_names):
        print(f"⚠️ Please provide {len(feature_names)} features. You gave {len(input_features)}.")
        return

    input_array = np.array([input_features])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    print("🔮 Churn Prediction:", "Yes" if prediction[0] == 1 else "No")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', nargs='+', type=float, help="List of user features for prediction")

    args = parser.parse_args()

    if args.features:
        predict_user(args.features)
    else:
        print("⚠️ Please provide user features using --features")

if __name__ == "__main__":
    main()
