"""
data_loader.py — Stock Data Download Module
Downloads OHLCV data from Yahoo Finance using yfinance library.

Project: Stock Price Forecasting using LSTM Neural Network
"""

import yfinance as yf
import pandas as pd
import os


def download_stock_data(ticker='RELIANCE.NS', period='5y', interval='1d'):
    """
    Download OHLCV data from Yahoo Finance.

    Parameters:
        ticker  (str): Stock ticker symbol (e.g., 'RELIANCE.NS' for NSE)
        period  (str): Historical period — '1y', '2y', '5y', '10y', 'max'
        interval(str): Data interval — '1d' (daily), '1wk', '1mo'

    Returns:
        pd.DataFrame: OHLCV dataframe with Date index
    """
    print(f'[INFO] Downloading {ticker} data for period={period}, interval={interval}...')
    df = yf.download(ticker, period=period, interval=interval)

    # Flatten MultiIndex columns if present (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Drop any rows with NaN values
    df.dropna(inplace=True)

    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    filename = f'data/raw/{ticker.replace(".", "_")}.csv'
    df.to_csv(filename)
    print(f'[INFO] Saved {len(df)} rows to {filename}')

    return df


def load_stock_data(ticker='RELIANCE.NS'):
    """
    Load previously downloaded stock data from CSV.

    Parameters:
        ticker (str): Stock ticker symbol used during download

    Returns:
        pd.DataFrame: OHLCV dataframe with Date index
    """
    filename = f'data/raw/{ticker.replace(".", "_")}.csv'
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f'{filename} not found. Run download_stock_data() first.'
        )
    df = pd.read_csv(filename, index_col='Date', parse_dates=True)
    print(f'[INFO] Loaded {len(df)} rows from {filename}')
    return df


if __name__ == '__main__':
    # Quick test — download RELIANCE 5-year daily data
    df = download_stock_data('RELIANCE.NS', period='5y')
    print(df.tail())
    print(f'\nShape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print(f'Date Range: {df.index.min()} -> {df.index.max()}')
