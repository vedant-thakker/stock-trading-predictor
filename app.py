import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

st.title("📈 Dummy Stock Trading App")
st.sidebar.header("Select Stock & Date Range")

# Select stock symbol
stock_symbol = st.sidebar.selectbox("Choose a stock", ["AAPL", "GOOG", "MSFT", "AMZN", "Nifty 50 (NSEI)"])
stock_ticker_map = {
    "AAPL": "AAPL",
    "GOOG": "GOOG",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "Nifty 50 (NSEI)": "^NSEI"  # Correct ticker for NSEI in Yahoo Finance
}
selected_ticker = stock_ticker_map[stock_symbol]

# Select date range
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Fetch stock data
try:
    data = yf.download(selected_ticker, start=start_date, end=end_date)
    
    if data.empty:
        st.warning("No data available for the selected range. Try a different date range or stock.")
    else:
        # Display stock chart
        st.subheader(f"Stock Price of {stock_symbol}")
        st.line_chart(data['Close'])

        # Preprocess Data for LSTM
        df = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0,1))
        df_scaled = scaler.fit_transform(df)

        # Define lookback period (past 60 days to predict next)
        lookback = 60
        X, y = [], []
        for i in range(lookback, len(df_scaled)):
            X.append(df_scaled[i-lookback:i, 0])
            y.append(df_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM

        # Load or Train LSTM Model
        model_path = f"{stock_symbol}_lstm_model.h5"  # Save separate model for each stock
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(lookback,1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(X, y, epochs=10, batch_size=16, verbose=0)
            model.save(model_path)

        # Predict next day price
        last_60_days = df_scaled[-lookback:].reshape(1, lookback, 1)
        predicted_price = model.predict(last_60_days)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        # Convert to float
        last_price = float(df['Close'].iloc[-1])
        predicted_price = float(predicted_price)  # Convert numpy.float32 to Python float

        # Display Prediction
        st.subheader("Predicted Next Close Price")
        st.metric(label="Predicted Price", 
                  value=float(round(predicted_price, 2)), 
                  delta=float(round(predicted_price - last_price, 2)))

        # Simulated Buy/Sell decision
        st.subheader("Trading Decision")
        if predicted_price > last_price:
            st.success("✅ Buy Signal")
        else:
            st.error("❌ Sell Signal")

except Exception as e:
    st.error(f"Error fetching stock data: {str(e)}")

# Footer
st.markdown("---")
st.caption("Dummy Trading App for University Project")