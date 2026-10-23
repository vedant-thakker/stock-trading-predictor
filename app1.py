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

# ---------------- Login/Logout System ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔒 Login to Access Stock Predictor")

    username = st.text_input("Enter Username")
    password = st.text_input("Enter Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if username and password:
            st.session_state.logged_in = True
            st.success("Login successful! Please click anywhere to continue.")
        else:
            st.error("Please enter both username and password to login.")
    st.stop()

# ---------------- Logout Button ----------------
st.sidebar.button("🔓 Logout", on_click=lambda: st.session_state.update(logged_in=False))


# ---------------- Streamlit Title ----------------
st.title("\U0001F4C8 Dummy Stock Trading App")

# ---------------- Sidebar Header ----------------
st.sidebar.header("Select Stock & Date Range")

# ---------------- Stock Selection ----------------
stock_symbol = st.sidebar.selectbox("Choose a stock", [
    "AAPL", 
    "GOOG", 
    "MSFT", 
    "AMZN", 
    "Nifty 50 (NSEI)",
    "BlackRock, Inc. (BLK)"
])

stock_ticker_map = {
    "AAPL": "AAPL",
    "GOOG": "GOOG",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "Nifty 50 (NSEI)": "^NSEI",
    "Nippon India Ultra Short Duration Fund (0P0001BB3Y.BO)": "0P0001BB3Y.BO",
    "TATASTEEL.NS": "TATASTEEL.NS",
    "LT.NS": "LT.NS",
    "BlackRock, Inc. (BLK)": "BLK"
}

selected_ticker = stock_ticker_map[stock_symbol]

# ---------------- Date Range Selection ----------------
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# ---------------- Fetching Data ----------------
st.subheader("Fetching stock data...")
try:
    data = yf.download(selected_ticker, start=start_date, end=end_date)

    if data.empty:
        st.warning("No data available for the selected range. Try a different date range or stock.")
    else:
        st.success("Data successfully fetched!")

        # ---------------- Display Data Table ----------------
        with st.expander("Raw Stock Data"):
            st.dataframe(data)

        # ---------------- Plotting Stock Chart ----------------
        st.subheader(f"Stock Price of {stock_symbol}")
        st.line_chart(data['Close'])

        # ---------------- LSTM Preprocessing ----------------
        st.subheader("Preparing data for LSTM model...")
        df = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0,1))
        df_scaled = scaler.fit_transform(df)

        lookback = 60
        X, y = [], []
        for i in range(lookback, len(df_scaled)):
            X.append(df_scaled[i-lookback:i, 0])
            y.append(df_scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        st.write(f"Number of training samples: {X.shape[0]}")

        # ---------------- Load/Train Model ----------------
        model_path = f"{stock_symbol}_lstm_model.h5"
        if os.path.exists(model_path):
            st.info("Loading saved model...")
            model = load_model(model_path)
        else:
            st.warning("Training new model. This may take a few seconds...")
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
            st.success("Model trained and saved successfully!")

        # ---------------- Prediction ----------------
        st.subheader("Predicting next day's closing price...")
        last_60_days = df_scaled[-lookback:].reshape(1, lookback, 1)
        predicted_price = model.predict(last_60_days)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        last_price = float(df['Close'].iloc[-1])
        predicted_price = float(predicted_price)

        # ---------------- Display Prediction ----------------
        st.metric(label="Predicted Price", 
                  value=float(round(predicted_price, 2)), 
                  delta=float(round(predicted_price - last_price, 2)))

        # ---------------- Trading Signal ----------------
        st.subheader("Trading Decision")
        if predicted_price > last_price:
            st.success("✅ Buy Signal")
        else:
            st.error("❌ Sell Signal")

        # ---------------- Extra Info ----------------
        st.subheader("Model Information")
        st.markdown("- Model Type: LSTM")
        st.markdown("- Optimizer: Adam")
        st.markdown("- Loss Function: Mean Squared Error")
        st.markdown("- Lookback Window: 60 days")
        st.markdown("- Epochs Trained: 10")

        st.subheader("Data Summary")
        st.write(df.describe())

        st.subheader("Closing Price Trend")
        fig, ax = plt.subplots()
        ax.plot(df.index[-100:], df['Close'][-100:], label="Close Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title("Last 100 Days of Stock Price")
        ax.legend()
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error fetching stock data: {str(e)}")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Lime® App for University Project")