"""
utils.py — Helper Functions for Streamlit App
Provides utility functions for data loading, prediction, and formatting.
Delegates prediction logic to src/predict.py to avoid code duplication.

Project: Stock Price Forecasting using LSTM Neural Network
"""

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.model import build_lstm_model


def get_stock_info(ticker):
    """Get basic stock information from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
        }
    except Exception:
        return {'name': ticker, 'sector': 'N/A', 'industry': 'N/A'}


def format_currency(value, currency='₹'):
    """Format number as currency string."""
    if isinstance(value, (int, float)):
        return f'{currency}{value:,.2f}'
    return str(value)


def format_large_number(value):
    """Format large numbers (e.g., market cap) into readable format."""
    if value >= 1e12:
        return f'₹{value/1e12:.2f}T'
    elif value >= 1e9:
        return f'₹{value/1e9:.2f}B'
    elif value >= 1e7:
        return f'₹{value/1e7:.2f}Cr'
    elif value >= 1e5:
        return f'₹{value/1e5:.2f}L'
    return f'₹{value:,.0f}'


def compute_technical_indicators(df):
    """Add common technical indicators to dataframe."""
    df = df.copy()

    # Moving Averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # Exponential Moving Average
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change() * 100

    # Volatility (30-day rolling std of returns)
    df['Volatility'] = df['Daily_Return'].rolling(window=30).std()

    return df


def generate_prediction(df, model_path, scaler_path, window_size=60):
    """
    Generate LSTM predictions for the given dataframe.

    Uses the trained model and scaler to create sliding-window predictions
    on the provided historical data. Returns dates, prices, and metrics.

    Returns:
        dates, actual_prices, predicted_prices, rmse, mae, mape
    """
    model = build_lstm_model()
    model.load_weights(model_path)
    scaler = joblib.load(scaler_path)

    close = df[['Close']].values
    scaled = scaler.transform(close)

    # Create sliding window sequences
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)

    # Generate predictions and inverse transform
    pred_scaled = model.predict(X, verbose=0)
    predicted = scaler.inverse_transform(pred_scaled).flatten()
    actual = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    dates = df.index[window_size:]

    # Compute evaluation metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return dates, actual, predicted, rmse, mae, mape


def forecast_future_prices(df, model_path, scaler_path,
                           days_ahead=30, window_size=60):
    """
    Forecast future prices using autoregressive approach.

    Each prediction is fed back as input for the next prediction,
    generating a multi-step forecast beyond available data.

    Returns:
        future_prices (np.array): Array of forecasted prices
    """
    model = build_lstm_model()
    model.load_weights(model_path)
    scaler = joblib.load(scaler_path)

    close = df[['Close']].values
    scaled = scaler.transform(close)

    # Start with last `window_size` values
    input_seq = scaled[-window_size:].reshape(1, window_size, 1)

    future_predictions = []
    for _ in range(days_ahead):
        pred = model.predict(input_seq, verbose=0)
        future_predictions.append(pred[0, 0])
        # Shift window: drop oldest, append new prediction
        new_input = np.append(input_seq[0, 1:, 0], pred[0, 0])
        input_seq = new_input.reshape(1, window_size, 1)

    # Inverse transform to original price scale
    future_prices = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).flatten()

    return future_prices
