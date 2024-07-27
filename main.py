from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('predictive_maintenance_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Predictive Maintenance API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the request
    data = request.get_json()

    # Ensure all required fields are in the request
    required_fields = ['Usage Hours', 'Maintenance Count', 'Environmental Temperature', 'Humidity', 'Asset Age Days', 'Days Since Last Maintenance']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    # Convert the input data to a numpy array
    features = np.array([[
        data['Usage Hours'],
        data['Maintenance Count'],
        data['Environmental Temperature'],
        data['Humidity'],
        data['Asset Age Days'],
        data['Days Since Last Maintenance']
    ]])

    # Make prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)[:, 1]

    # Return the prediction and probability
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(prediction_proba[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
