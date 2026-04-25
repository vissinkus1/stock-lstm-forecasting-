"""
model.py — LSTM Model Architecture Definition
Defines a 3-layer Stacked LSTM with Dropout regularization.


Project: Stock Price Forecasting using LSTM Neural Network
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_lstm_model(window_size=60, units=128, dropout=0.2, learning_rate=0.001):
    """
    Build a 3-layer Stacked LSTM model for stock price prediction.

    Architecture:
        Input  → (batch, window_size, 1)
        LSTM(128, return_sequences=True)  + Dropout(0.2)
        LSTM(64,  return_sequences=True)  + Dropout(0.2)
        LSTM(32,  return_sequences=False) + Dropout(0.2)
        Dense(32, activation='relu')
        Dense(1)  → Output: next day closing price (scaled)

    Parameters:
        window_size    (int):   Number of past timesteps (default: 60)
        units          (int):   LSTM units in first layer (default: 128)
        dropout        (float): Dropout rate between layers (default: 0.2)
        learning_rate  (float): Adam optimizer learning rate (default: 0.001)

    Returns:
        tf.keras.Model: Compiled Keras Sequential model
    """
    model = Sequential([
        # Input layer
        Input(shape=(window_size, 1)),

        # LSTM Layer 1 — captures short-term patterns
        LSTM(units, return_sequences=True, name='lstm_1'),
        Dropout(dropout, name='dropout_1'),

        # LSTM Layer 2 — captures medium-term patterns
        LSTM(units // 2, return_sequences=True, name='lstm_2'),
        Dropout(dropout, name='dropout_2'),

        # LSTM Layer 3 — captures long-term dependencies
        LSTM(units // 4, return_sequences=False, name='lstm_3'),
        Dropout(dropout, name='dropout_3'),

        # Fully connected layers
        Dense(32, activation='relu', name='fc_1'),
        Dense(1, name='output')  # Single output: predicted price
    ])

    # Compile with Adam optimizer and MSE loss
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    # Print model summary
    print('\n' + '=' * 60)
    print('LSTM MODEL ARCHITECTURE')
    print('=' * 60)
    model.summary()

    return model


if __name__ == '__main__':
    # Quick test — build and verify model
    model = build_lstm_model(window_size=60)
    print(f'\nTotal parameters: {model.count_params():,}')
