from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("iris_model.pkl")

FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "iris-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: float(data[feature])
            for feature in FEATURES
        }])
    except (TypeError, ValueError):
        return jsonify({
            "error": "All feature values must be numbers"
        }), 400

    if not MODEL_PATH.exists():
        return jsonify({
            "error": "Model file not found. Train the model first."
        }), 500

    model = joblib.load(MODEL_PATH)
    prediction = str(model.predict(sample)[0])

    return jsonify({
        "prediction": prediction
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
