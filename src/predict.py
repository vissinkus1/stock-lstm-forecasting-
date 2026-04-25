"""
predict.py — Load Model & Generate Predictions
Provides functions for single-stock prediction and future forecasting.

Project: Stock Price Forecasting using LSTM Neural Network
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from src.model import build_lstm_model

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_trained_model(model_path='models/lstm_weights.weights.h5', scaler_path='models/scaler.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or Scaler not found. Run train.py first.")
    
    model = build_lstm_model()
    model.load_weights(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_stock(ticker='RELIANCE.NS', period='1y', window_size=60,
                  model_path='models/lstm_weights.weights.h5',
                  scaler_path='models/scaler.pkl'):
    """
    Load trained model and generate predictions on recent data.

    Parameters:
        ticker      (str): Stock ticker symbol
        period      (str): Historical period to fetch
        window_size (int): Input sequence length
        model_path  (str): Path to trained model
        scaler_path (str): Path to fitted scaler

    Returns:
        dates       (pd.DatetimeIndex): Dates for predictions
        actual      (np.array):         Actual closing prices
        predicted   (np.array):         Predicted closing prices
    """
    # Load model and scaler
    model, scaler = load_trained_model(model_path, scaler_path)

    # Download fresh data
    df = yf.download(ticker, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df[['Close']].values
    scaled = scaler.transform(close)

    # Create sequences
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)

    # Predict
    pred_scaled = model.predict(X, verbose=0)
    predicted = scaler.inverse_transform(pred_scaled).flatten()
    actual = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    dates = df.index[window_size:]

    return dates, actual, predicted


def forecast_future(ticker='RELIANCE.NS', days_ahead=30, window_size=60,
                    model_path='models/lstm_weights.weights.h5',
                    scaler_path='models/scaler.pkl'):
    """
    Generate future price forecasts beyond available data.

    Uses autoregressive approach: each prediction becomes input
    for the next prediction.

    Parameters:
        ticker      (str): Stock ticker symbol
        days_ahead  (int): Number of future days to forecast
        window_size (int): Input sequence length

    Returns:
        future_prices (np.array): Forecasted closing prices
    """
    # Load model and scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Get latest data
    df = yf.download(ticker, period='1y')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df[['Close']].values
    scaled = scaler.transform(close)

    # Take last `window_size` values as initial input
    input_seq = scaled[-window_size:].reshape(1, window_size, 1)

    future_predictions = []
    for _ in range(days_ahead):
        # Predict next value
        pred = model.predict(input_seq, verbose=0)
        future_predictions.append(pred[0, 0])

        # Shift window: drop first, append prediction
        new_input = np.append(input_seq[0, 1:, 0], pred[0, 0])
        input_seq = new_input.reshape(1, window_size, 1)

    # Inverse transform to original scale
    future_prices = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).flatten()

    print(f'\n[INFO] {days_ahead}-day forecast for {ticker}:')
    for i, price in enumerate(future_prices, 1):
        print(f'  Day {i:3d}: INR {price:,.2f}')

    return future_prices


if __name__ == '__main__':
    # Test: predict on recent data
    dates, actual, predicted = predict_stock('RELIANCE.NS', period='1y')
    print(f'Predictions generated for {len(dates)} days')
    print(f'Last actual:    INR {actual[-1]:,.2f}')
    print(f'Last predicted: INR {predicted[-1]:,.2f}')

    # Test: forecast next 10 days
    future = forecast_future('RELIANCE.NS', days_ahead=10)
