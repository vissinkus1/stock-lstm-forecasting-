# 📈 Stock Price Forecasting using LSTM Neural Network

> End-to-end deep learning project that forecasts stock closing prices using a Stacked LSTM model trained on Yahoo Finance data.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎓 Project Details

| Field | Detail |
|-------|--------|
| **Author** | Vishal Singh Kushwaha |
| **Enrollment** | CS-23411025 |
| **Institution** | IILM University, Greater Noida |
| **Program** | B.Tech CSE (AI & ML), Year 3 |
| **Category** | Deep Learning — Sequence Modeling |

## 🚀 Live Demo

🔗 *Deploy to Render.com using the included `render.yaml` — update this link after deployment*

## 🏗️ Architecture

```
Input (60-day window) → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2)
→ LSTM(32) → Dropout(0.2) → Dense(32, ReLU) → Dense(1)
```

- **Model**: 3-Layer Stacked LSTM with Dropout Regularization
- **Input**: 60-day sliding window of normalized closing prices
- **Normalization**: MinMaxScaler [0, 1]
- **Optimizer**: Adam (lr=0.001) with ReduceLROnPlateau
- **Loss**: Mean Squared Error (MSE)
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

## 📊 Results

| Metric | Value |
|--------|-------|
| RMSE   | 26.52 INR |
| MAE    | 20.88 INR |
| MAPE   | 1.45% |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Collection | yfinance, pandas |
| Deep Learning | TensorFlow 2.x / Keras |
| Visualization | Matplotlib, Plotly |
| Web App | Streamlit |
| Deployment | Render.com |

## 📁 Project Structure

```
stock-lstm-forecasting/
├── data/
│   ├── raw/                    # Downloaded CSV from yfinance
│   └── processed/              # Scaled arrays & plots
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   └── 02_model_training.ipynb # Model experimentation
├── src/
│   ├── __init__.py             # Package initializer
│   ├── data_loader.py          # yfinance download + preprocessing
│   ├── feature_engineering.py  # Window/sequence creation
│   ├── model.py                # LSTM architecture definition
│   ├── train.py                # Training pipeline + callbacks
│   ├── evaluate.py             # RMSE, MAE, MAPE computation
│   └── predict.py              # Load model + generate predictions
├── models/
│   ├── lstm_model.keras        # Saved trained model
│   └── scaler.pkl              # Saved MinMaxScaler
├── app/
│   ├── __init__.py             # Package initializer
│   ├── streamlit_app.py        # Main Streamlit dashboard
│   └── utils.py                # Helper functions
├── .streamlit/
│   └── config.toml             # Streamlit theme & server config
├── plots/                      # Training & evaluation plots
├── requirements.txt
├── render.yaml                 # Render.com deployment config
├── LICENSE                     # MIT License
├── README.md
└── .gitignore
```

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/vissinkus1/stock-lstm-forecasting.git
cd stock-lstm-forecasting
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data & Train Model
```bash
python src/train.py
```

### 5. Evaluate Model
```bash
python src/evaluate.py
```

### 6. Launch Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

## 📸 Screenshots

### Dashboard
*Interactive Streamlit dashboard with candlestick charts, technical indicators, and LSTM predictions*

### Prediction Results
*Actual vs Predicted closing prices with RMSE, MAE, and MAPE metrics*

## 🔑 Key Features

- ✅ **Live Data**: Fetches real-time stock data from Yahoo Finance
- ✅ **Technical Indicators**: MA, EMA, Bollinger Bands, RSI
- ✅ **Interactive Charts**: Plotly candlestick charts with hover tooltips
- ✅ **LSTM Prediction**: Compare actual vs predicted prices
- ✅ **Future Forecasting**: Autoregressive forecasting for N days ahead
- ✅ **Multiple Stocks**: Support for 10+ NSE-listed stocks
- ✅ **Error Analysis**: Error distribution and time-series error plots
- ✅ **Export**: View and export forecast data tables

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock price predictions from this model should **NOT** be used for actual investment decisions. Financial markets are influenced by countless factors beyond historical price data.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

**Vishal Singh Kushwaha** | CS-23411025 | B.Tech AI & ML | IILM University, Greater Noida

[![GitHub](https://img.shields.io/badge/GitHub-vissinkus1-181717?logo=github)](https://github.com/vissinkus1)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Vishal_Singh_Kushwaha-0A66C2?logo=linkedin)](https://linkedin.com/in/vishal-singh-kushwaha27)
