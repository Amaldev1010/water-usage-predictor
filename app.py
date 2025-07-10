from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Example model (replace with your trained model)
model = LinearRegression()
# Dummy data for demonstration (replace with your training data)
X = np.array([[1000, 20.0, 50000], [2000, 30.0, 100000]])
y = np.array([5000, 10000])
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    industry_name = data.get('industryName')
    location = data.get('location')
    count_employees = data.get('countEmployees', 0)
    average_daily_usage = data.get('averageDailyUsage', 0.0)
    total_capacity = data.get('totalCapacity', 0)

    # Prepare input for prediction
    input_data = np.array([[count_employees, average_daily_usage, total_capacity]])
    predicted_consumption = model.predict(input_data)[0]

    return jsonify({"predictedDailyConsumption": float(predicted_consumption)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
