"""
train.py — LSTM Model Training Pipeline
Downloads data, preprocesses, builds model, trains with callbacks.

Project: Stock Price Forecasting using LSTM Neural Network

Usage:
    cd stock-lstm-forecasting
    python src/train.py
"""

import sys
import os
import json
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import download_stock_data
from src.feature_engineering import preprocess
from src.model import build_lstm_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — Modify these hyperparameters as needed
# ══════════════════════════════════════════════════════════════
TICKER      = 'RELIANCE.NS'  # NSE stock ticker
PERIOD      = '5y'           # Historical data period
WINDOW_SIZE = 60             # 60 trading days (~3 months) lookback
EPOCHS      = 100            # Max epochs (will early-stop before)
BATCH_SIZE  = 32             # Mini-batch size
LSTM_UNITS  = 128            # Units in first LSTM layer
DROPOUT     = 0.2            # Dropout rate
LR          = 0.001          # Initial learning rate
VAL_SPLIT   = 0.1            # Validation split from training data
TEST_SPLIT  = 0.2            # Test split from total data


def plot_training_history(history):
    """Plot training & validation loss curves and save to disk."""
    os.makedirs('plots', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(history.history['loss'], label='Training Loss', color='steelblue', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='tomato', linewidth=2)
    ax1.set_title('Model Loss (MSE)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MAE curve
    ax2.plot(history.history['mae'], label='Training MAE', color='steelblue', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='tomato', linewidth=2)
    ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/training_history.png', dpi=150, bbox_inches='tight')
    print('[INFO] Training history plot saved to plots/training_history.png')
    plt.close(fig)


def main():
    """Main training pipeline."""
    print('\n' + '=' * 60)
    print('  STOCK PRICE FORECASTING - LSTM TRAINING PIPELINE')
    print(f'  Ticker: {TICKER} | Window: {WINDOW_SIZE} | Epochs: {EPOCHS}')
    print('=' * 60 + '\n')

    # -- Step 1: Download Data --------------------------------
    print('[PHASE 1] Downloading stock data...')
    df = download_stock_data(TICKER, period=PERIOD)
    print(f'  -> Data shape: {df.shape}')
    print(f'  -> Date range: {df.index.min()} to {df.index.max()}\n')

    # -- Step 2: Preprocess & Create Sequences ----------------
    print('[PHASE 2] Preprocessing & feature engineering...')
    X_train, y_train, X_test, y_test, scaler = preprocess(
        df, feature_col='Close', window_size=WINDOW_SIZE, test_split=TEST_SPLIT
    )
    print(f'  -> Train shape: {X_train.shape}')
    print(f'  -> Test shape:  {X_test.shape}\n')

    # -- Step 3: Build Model ──────────────────────────────────
    print('[PHASE 3] Building LSTM model...')
    model = build_lstm_model(
        window_size=WINDOW_SIZE,
        units=LSTM_UNITS,
        dropout=DROPOUT,
        learning_rate=LR
    )

    # -- Step 4: Define Callbacks ─────────────────────────────
    print('\n[PHASE 4] Setting up training callbacks...')
    os.makedirs('models', exist_ok=True)

    callbacks = [
        # Stop training when val_loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model based on val_loss
        ModelCheckpoint(
            'models/lstm_model.keras',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        # Reduce learning rate when val_loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Log training metrics to CSV
        CSVLogger('models/training_log.csv', separator=',', append=False)
    ]

    # -- Step 5: Train Model ──────────────────────────────────
    print('\n[PHASE 5] Training LSTM model...\n')
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    # -- Step 6: Save Results ─────────────────────────────────
    print('\n[PHASE 6] Saving results...')

    # Save training config
    config = {
        'ticker': TICKER,
        'period': PERIOD,
        'window_size': WINDOW_SIZE,
        'epochs_trained': len(history.history['loss']),
        'batch_size': BATCH_SIZE,
        'lstm_units': LSTM_UNITS,
        'dropout': DROPOUT,
        'initial_lr': LR,
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_loss': float(min(history.history['val_loss'])),
    }
    with open('models/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print('  -> Training config saved to models/training_config.json')

    # Plot training curves
    plot_training_history(history)

    print('\n' + '=' * 60)
    print('  TRAINING COMPLETE!')
    print(f'  Best model saved to: models/lstm_model.keras')
    print(f'  Scaler saved to:     models/scaler.pkl')
    print(f'  Epochs trained:      {len(history.history["loss"])}')
    print(f'  Best val_loss:       {min(history.history["val_loss"]):.6f}')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
