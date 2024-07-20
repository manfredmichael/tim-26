from flask import Flask, request, jsonify
import json
import sqlite3
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

# Define the path to the SQLite database file
db_file_path = 'sensor_data.db'

# Create SQLite database if it doesn't exist
def create_db():
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL,
            humidity REAL,
            soil_moisture INTEGER,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

create_db()

@app.route('/', methods=["GET"])
def index():
    return "Hello!"

@app.route('/sensor/data', methods=["POST"])
def sensor():
    data = request.get_json()
    temperature = data["temperature"]
    humidity = data["humidity"]
    soil_moisture = data["soil_moisture"]
    timestamp = data["timestamp"]

    # Save data to SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sensor_data (temperature, humidity, soil_moisture, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (temperature, humidity, soil_moisture, timestamp))
    conn.commit()
    conn.close()

    response_data = {
        'message': 'Data saved successfully'
    }
    return jsonify(response_data), 200

@app.route('/train', methods=['GET'])
def train():
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute('SELECT soil_moisture FROM sensor_data')
    soil_moisture_data = cursor.fetchall()
    conn.close()

    # Flatten the list of tuples
    soil_moisture_data = [x[0] for x in soil_moisture_data]

    X, y = create_windowed_dataset(soil_moisture_data)
    model = train_model(X, y)

    # Save the model using joblib
    joblib.dump(model, 'soil_moisture_model.pkl')

    return jsonify({'message': 'Model trained successfully'}), 200

def create_windowed_dataset(data, window_size=48):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

@app.route('/predict', methods=['GET'])
def predict():
    model = joblib.load('soil_moisture_model.pkl')
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute('SELECT soil_moisture FROM sensor_data ORDER BY id DESC LIMIT 48')
    last_data = cursor.fetchall()
    conn.close()

    # Flatten the list of tuples and create a single prediction
    last_data = [x[0] for x in last_data]
    last_data = np.array(last_data).reshape(1, -1)
    predicted_soil_moisture = model.predict(last_data)

    return jsonify({'predicted_soil_moisture': predicted_soil_moisture[0]}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)