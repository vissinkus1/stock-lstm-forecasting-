"""
feature_engineering.py — Data Preprocessing & Sequence Creation
Converts raw OHLCV data into LSTM-ready sliding window sequences.

Project: Stock Price Forecasting using LSTM Neural Network
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def preprocess(df, feature_col='Close', window_size=60, test_split=0.2):
    """
    Full preprocessing pipeline for LSTM model.

    Steps:
        1. Extract feature column (Close price)
        2. Normalize with MinMaxScaler to [0, 1]
        3. Create sliding window sequences of length window_size
        4. Split into train/test sets (preserving temporal order)

    Parameters:
        df          (pd.DataFrame): Raw OHLCV dataframe
        feature_col (str):          Column to use for prediction (default: 'Close')
        window_size (int):          Number of past days as input (default: 60)
        test_split  (float):        Fraction of data for testing (default: 0.2)

    Returns:
        X_train (np.array): Training sequences — shape (samples, window_size, 1)
        y_train (np.array): Training labels — shape (samples,)
        X_test  (np.array): Test sequences — shape (samples, window_size, 1)
        y_test  (np.array): Test labels — shape (samples,)
        scaler  (MinMaxScaler): Fitted scaler for inverse transformation
    """
    # Step 1: Extract feature column
    data = df[[feature_col]].values  # shape: (N, 1)
    print(f'[INFO] Using feature: {feature_col} | Total samples: {len(data)}')

    # Step 2: Normalize to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)  # shape: (N, 1)

    # Save scaler for inference/deployment
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print('[INFO] Scaler saved to models/scaler.pkl')

    # Step 3: Create sliding window sequences
    # X = [day_{t-60}, day_{t-59}, ..., day_{t-1}]  →  y = day_{t}
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i, 0])  # 60 past values
        y.append(scaled[i, 0])                   # next value

    X, y = np.array(X), np.array(y)

    # Reshape X for LSTM: (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    print(f'[INFO] Created {len(X)} sequences with window_size={window_size}')

    # Step 4: Train/Test split (temporal — NO shuffle!)
    split = int(len(X) * (1 - test_split))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print(f'[INFO] Train: {X_train.shape} | Test: {X_test.shape}')

    # Save processed arrays for quick re-loading
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_test.npy', y_test)
    print('[INFO] Processed arrays saved to data/processed/')

    return X_train, y_train, X_test, y_test, scaler


def load_processed_data():
    """
    Load previously processed train/test arrays from disk.

    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    scaler = joblib.load('models/scaler.pkl')
    print(f'[INFO] Loaded processed data - Train: {X_train.shape}, Test: {X_test.shape}')
    return X_train, y_train, X_test, y_test, scaler


if __name__ == '__main__':
    # Quick test
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.data_loader import load_stock_data
    df = load_stock_data('RELIANCE.NS')
    X_train, y_train, X_test, y_test, scaler = preprocess(df, window_size=60)
    print(f'\nX_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape:  {X_test.shape}')
    print(f'y_test shape:  {y_test.shape}')
