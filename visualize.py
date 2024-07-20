import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

db_file_path = 'sensor_data.db'

def load_data():
    conn = sqlite3.connect(db_file_path)
    df = pd.read_sql_query("SELECT * FROM sensor_data", conn)
    conn.close()
    return df

def create_windowed_dataset(data, feature_cols, target_col, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data.loc[i:i+window_size-1, feature_cols].to_numpy().flatten())
        y.append(data.loc[i+window_size, target_col])
    return np.array(X), np.array(y)

df = load_data()

# Window size is based on past observations; you can adjust this value
window_size = 10  
feature_columns = ['temperature', 'humidity', 'soil_moisture']
target_column = 'soil_moisture'

X, y = create_windowed_dataset(df, feature_columns, target_column, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction loop for 8 hours into the future
predictions = []
input_features = X[-1].reshape(1, -1)

for _ in range(480):  # 480 minutes into the future
    next_prediction = model.predict(input_features)[0]
    predictions.append(next_prediction)
    # Update input features with new prediction
    input_features = np.roll(input_features, -3)  # shift left by 3 positions
    input_features[0, -3] = df['temperature'].iloc[-1]  # static future temperature
    input_features[0, -2] = df['humidity'].iloc[-1]  # static future humidity
    input_features[0, -1] = next_prediction  # updated soil moisture

st.title('Sensor Data History')
st.write('This is the historical data of temperature, humidity, and soil moisture.')

if not df.empty:
    fig = px.line(df, x='timestamp', y='temperature', title='Temperature Over Time')
    st.plotly_chart(fig)

    fig = px.line(df, x='timestamp', y='humidity', title='Humidity Over Time')
    st.plotly_chart(fig)

    # Combine historical and predicted soil moisture for plotting
    future_timestamps = pd.date_range(df['timestamp'].iloc[-1], periods=481, freq='T')[1:]  # 1 minute freq, skip the first (existing last point)
    predicted_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'soil_moisture': predictions,
        'Type': 'Predicted'  # new column to differentiate the data
    })
    historical_df = df[['timestamp', 'soil_moisture']].copy()
    historical_df['Type'] = 'Historical'
    combined_df = pd.concat([historical_df, predicted_df])

    fig = px.line(combined_df, x='timestamp', y='soil_moisture', color='Type', title='Soil Moisture Over Time and Predictions')
    st.plotly_chart(fig)
else:
    st.write("No data available.")
