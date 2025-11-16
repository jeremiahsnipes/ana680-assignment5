from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Wine Quality Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Expecting a list of 11 feature values
    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)[0]

    return jsonify({"quality_prediction": float(prediction)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
