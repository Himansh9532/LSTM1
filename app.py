import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Configure Streamlit to use utf-8 encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Streamlit code to create a text input for the stock ticker
user_input = st.text_input("Enter stock name", "AAPL")

# Fetch data for the ticker symbol entered by the user
df = yf.download(user_input, start='2020-01-01', end='2024-07-09')

# Display a subheader and the data description in Streamlit
st.subheader("Data from 2020-2024")
st.write(df.describe())

# Plot the Close Price vs Time chart
st.subheader("Close Price vs Time chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
st.pyplot(fig)

# Plot the Close Price vs Time chart with 100MA
st.subheader("Close Price vs Time chart with 100MA")
MA100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.plot(MA100, label='100MA')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
st.pyplot(fig)

# Plot the Close Price vs Time chart with 100MA and 200MA
st.subheader("Close Price vs Time chart with 100MA and 200MA")
MA200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.plot(MA100, label='100MA')
plt.plot(MA200, label='200MA')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df)*0.70):])

st.subheader("Training Data")
st.write(data_training.describe())

st.subheader("Testing Data")
st.write(data_testing.describe())

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load the pre-trained model
model = load_model("keras_model.h5")

# Prepare testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

# Creating x_test and y_test
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

# Convert lists to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Predict the prices
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot the predictions vs original prices
st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, "b", label="Original Price")
plt.plot(y_predicted, "r", label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
