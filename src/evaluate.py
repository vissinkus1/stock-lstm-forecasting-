"""
evaluate.py — Model Evaluation & Visualization
Loads trained model, generates predictions, computes metrics, plots results.

Project: Stock Price Forecasting using LSTM Neural Network

Usage:
    cd stock-lstm-forecasting
    python src/evaluate.py
"""

import sys
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def evaluate(X_test, y_test, model_path='models/lstm_model.keras',
             scaler_path='models/scaler.pkl'):
    """
    Evaluate trained LSTM model on test data.

    Parameters:
        X_test     (np.array): Test sequences — shape (samples, window_size, 1)
        y_test     (np.array): Test labels — shape (samples,)
        model_path (str):      Path to saved .h5 model
        scaler_path(str):      Path to saved MinMaxScaler

    Returns:
        rmse  (float): Root Mean Square Error
        mae   (float): Mean Absolute Error
        mape  (float): Mean Absolute Percentage Error
    """
    # Load model and scaler
    print('[INFO] Loading model and scaler...')
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Generate predictions
    print('[INFO] Generating predictions on test set...')
    pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform to original price scale (INR)
    pred_price = scaler.inverse_transform(pred_scaled)
    actual_price = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ── Compute Metrics ──────────────────────────────────────
    rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
    mae = mean_absolute_error(actual_price, pred_price)
    mape = np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100

    print('\n' + '=' * 50)
    print('  MODEL EVALUATION RESULTS')
    print('=' * 50)
    print(f'  RMSE  : {rmse:.2f} INR')
    print(f'  MAE   : {mae:.2f} INR')
    print(f'  MAPE  : {mape:.2f}%')
    print('=' * 50 + '\n')

    # ── Plot: Actual vs Predicted ────────────────────────────
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(actual_price, label='Actual Price', color='steelblue', linewidth=1.8)
    plt.plot(pred_price, label='Predicted Price', color='tomato', linewidth=1.8,
             linestyle='--', alpha=0.85)
    plt.title(f'LSTM Stock Price Prediction - RMSE: {rmse:.2f} INR',
              fontsize=14, fontweight='bold')
    plt.xlabel('Test Sample Index', fontsize=12)
    plt.ylabel('Price (INR)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/prediction_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.savefig('data/processed/prediction_vs_actual.png', dpi=150, bbox_inches='tight')
    print('[INFO] Plot saved to plots/prediction_vs_actual.png')
    plt.close()

    # ── Plot: Error Distribution ─────────────────────────────
    errors = (actual_price - pred_price).flatten()
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    plt.axvline(x=0, color='tomato', linestyle='--', linewidth=2)
    plt.title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Error (INR)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/error_distribution.png', dpi=150, bbox_inches='tight')
    print('[INFO] Error distribution saved to plots/error_distribution.png')
    plt.close()

    return rmse, mae, mape


def main():
    """Load test data and evaluate model."""
    from src.feature_engineering import load_processed_data

    print('[INFO] Loading processed test data...')
    _, _, X_test, y_test, _ = load_processed_data()

    rmse, mae, mape = evaluate(X_test, y_test)

    # Save metrics to file
    with open('models/evaluation_metrics.txt', 'w') as f:
        f.write(f'RMSE : {rmse:.2f}\n')
        f.write(f'MAE  : {mae:.2f}\n')
        f.write(f'MAPE : {mape:.2f}%\n')
    print('[INFO] Metrics saved to models/evaluation_metrics.txt')


if __name__ == '__main__':
    main()
