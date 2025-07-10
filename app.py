import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy import stats
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load and preprocess data (done once at startup)
df = pd.read_csv('Daily.csv')
df['Average_Daily_Usage'] = df['Average_Daily_Usage'].fillna(
    (df['Daily_Consumption'] / df['Count_Employees'].replace(0, 1))
).round(2)

df.dropna(inplace=True)
numeric_cols = ['Daily_Consumption', 'Count_Employees', 'Average_Daily_Usage', 'total_capacity']
df = df[(np.abs(stats.zscore(df[numeric_cols])) < 3).all(axis=1)]

X = df.drop(columns=['Daily_Consumption'])
y = df['Daily_Consumption']

categorical_features = ['Industry_Name', 'Location']
numeric_features = ['Count_Employees', 'Average_Daily_Usage', 'total_capacity']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])

# Train the model (done once at startup)
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([{
            'Industry_Name': data['industryName'],
            'Location': data['location'],
            'Count_Employees': float(data['countEmployees']),
            'Average_Daily_Usage': float(data['averageDailyUsage']),
            'total_capacity': float(data['totalCapacity'])
        }])
        predicted_value = model.predict(input_df)[0]
        return jsonify({'predictedDailyConsumption': round(predicted_value, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)