"""
streamlit_app.py — Stock Price Forecasting Dashboard
Interactive web app with LSTM predictions, technical indicators, and charts.

Project: Stock Price Forecasting using LSTM Neural Network
Usage:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
import sys
from datetime import date, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.utils import (
    get_stock_info,
    format_currency,
    format_large_number,
    compute_technical_indicators,
    generate_prediction,
    forecast_future_prices
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title='Stock LSTM Forecast',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium Dark Theme Styling
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0.5rem 0 0;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-card .label {
        color: #8892b0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        color: #ccd6f6;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .metric-card .value.green { color: #64ffda; }
    .metric-card .value.red   { color: #ff6b6b; }
    .metric-card .value.blue  { color: #667eea; }

    /* Section Headers */
    .section-header {
        color: #ccd6f6;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }

    /* Info Box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #8892b0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #8892b0;
        font-size: 0.85rem;
        border-top: 1px solid rgba(102, 126, 234, 0.15);
        margin-top: 3rem;
    }
    .footer a {
        color: #667eea;
        text-decoration: none;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>📈 Stock Price Forecasting — LSTM Model</h1>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR — Configuration Panel
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('### ⚙️ Configuration')
    st.markdown('---')

    # Stock Selection
    ticker = st.selectbox(
        '🏢 Select Stock',
        ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
         'ICICIBANK.NS', 'WIPRO.NS', 'SBIN.NS', 'BHARTIARTL.NS',
         'TATAMOTORS.NS', 'LT.NS'],
        index=0,
        help='Select an NSE-listed stock ticker'
    )

    period = st.selectbox(
        '📅 Historical Period',
        ['1y', '2y', '5y', '10y'],
        index=2,
        help='Amount of historical data to fetch'
    )

    st.markdown('---')
    st.markdown('### 🧠 Model Settings')

    window_size = st.slider(
        'Lookback Window (days)', 30, 120, 60,
        help='Number of past trading days used as input to LSTM'
    )

    forecast_days = st.slider(
        'Forecast Horizon (days)', 5, 60, 30,
        help='Number of future days to forecast'
    )

    st.markdown('---')
    st.markdown('### 📊 Display Options')
    show_volume = st.checkbox('Show Volume', value=True)
    show_ma = st.checkbox('Show Moving Averages', value=True)
    show_bollinger = st.checkbox('Show Bollinger Bands', value=False)
    show_rsi = st.checkbox('Show RSI', value=False)

    st.markdown('---')
    st.markdown("""
    <div style="text-align:center; color:#8892b0; font-size:0.8rem;">
        <strong>Built with</strong><br>
        TensorFlow • Streamlit • Plotly<br><br>
        <strong>Model</strong><br>
        3-Layer Stacked LSTM<br>
        with Dropout Regularization
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, period):
    """Download and cache stock data."""
    df = yf.download(ticker, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


with st.spinner(f'Fetching {ticker} data...'):
    df = load_data(ticker, period)

if df.empty:
    st.error(f'❌ No data found for {ticker}. Please try a different stock.')
    st.stop()

# Add technical indicators
df_tech = compute_technical_indicators(df)

# ══════════════════════════════════════════════════════════════
# KEY METRICS ROW
# ══════════════════════════════════════════════════════════════
latest_close = df['Close'].iloc[-1]
prev_close = df['Close'].iloc[-2]
price_change = latest_close - prev_close
pct_change = (price_change / prev_close) * 100
high_52w = df['Close'].tail(252).max() if len(df) >= 252 else df['Close'].max()
low_52w = df['Close'].tail(252).min() if len(df) >= 252 else df['Close'].min()
avg_volume = df['Volume'].tail(30).mean()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    change_color = 'green' if price_change >= 0 else 'red'
    change_sign = '+' if price_change >= 0 else ''
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Current Price</div>
        <div class="value">{format_currency(latest_close)}</div>
        <div class="value {change_color}" style="font-size:1rem;">
            {change_sign}{format_currency(price_change)} ({change_sign}{pct_change:.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">52W High</div>
        <div class="value green">{format_currency(high_52w)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">52W Low</div>
        <div class="value red">{format_currency(low_52w)}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Avg Volume (30d)</div>
        <div class="value blue">{avg_volume:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Data Points</div>
        <div class="value blue">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SECTION 1: INTERACTIVE PRICE CHART
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Historical Price Chart</div>',
            unsafe_allow_html=True)

# Build the main chart
rows = 1 + (1 if show_volume else 0) + (1 if show_rsi else 0)
row_heights = [0.6]
if show_volume:
    row_heights.append(0.2)
if show_rsi:
    row_heights.append(0.2)

subplot_titles = ['Price']
if show_volume:
    subplot_titles.append('Volume')
if show_rsi:
    subplot_titles.append('RSI (14)')

fig_main = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=row_heights,
    subplot_titles=subplot_titles
)

# Candlestick chart
fig_main.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    name='OHLC',
    increasing_line_color='#64ffda',
    decreasing_line_color='#ff6b6b'
), row=1, col=1)

# Moving Averages
if show_ma:
    if 'MA_20' in df_tech.columns:
        fig_main.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['MA_20'],
            name='MA 20', line=dict(color='#ffd93d', width=1)
        ), row=1, col=1)
    if 'MA_50' in df_tech.columns:
        fig_main.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['MA_50'],
            name='MA 50', line=dict(color='#ff6b6b', width=1)
        ), row=1, col=1)
    if 'MA_200' in df_tech.columns:
        fig_main.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['MA_200'],
            name='MA 200', line=dict(color='#667eea', width=1.5)
        ), row=1, col=1)

# Bollinger Bands
if show_bollinger:
    fig_main.add_trace(go.Scatter(
        x=df_tech.index, y=df_tech['BB_Upper'],
        name='BB Upper', line=dict(color='rgba(102,126,234,0.4)', width=1)
    ), row=1, col=1)
    fig_main.add_trace(go.Scatter(
        x=df_tech.index, y=df_tech['BB_Lower'],
        name='BB Lower', line=dict(color='rgba(102,126,234,0.4)', width=1),
        fill='tonexty', fillcolor='rgba(102,126,234,0.08)'
    ), row=1, col=1)

# Volume
current_row = 2
if show_volume:
    colors = ['#64ffda' if c >= o else '#ff6b6b'
              for c, o in zip(df['Close'], df['Open'])]
    fig_main.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume', marker_color=colors, opacity=0.6
    ), row=current_row, col=1)
    current_row += 1

# RSI
if show_rsi and 'RSI' in df_tech.columns:
    fig_main.add_trace(go.Scatter(
        x=df_tech.index, y=df_tech['RSI'],
        name='RSI', line=dict(color='#667eea', width=1.5)
    ), row=current_row, col=1)
    fig_main.add_hline(y=70, line_dash='dash', line_color='#ff6b6b',
                       opacity=0.5, row=current_row, col=1)
    fig_main.add_hline(y=30, line_dash='dash', line_color='#64ffda',
                       opacity=0.5, row=current_row, col=1)

# Layout
fig_main.update_layout(
    template='plotly_dark',
    height=500 + (rows - 1) * 150,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=50, r=20, t=40, b=30),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(26,26,46,0.8)',
)
fig_main.update_xaxes(gridcolor='rgba(102,126,234,0.1)')
fig_main.update_yaxes(gridcolor='rgba(102,126,234,0.1)')

st.plotly_chart(fig_main, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# SECTION 2: LSTM PREDICTION
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🧠 LSTM Model Prediction</div>',
            unsafe_allow_html=True)

model_path = 'models/lstm_model.keras'
scaler_path = 'models/scaler.pkl'

model_exists = os.path.exists(model_path) and os.path.exists(scaler_path)

if not model_exists:
    st.markdown("""
    <div class="info-box">
        <strong>⚠️ Model not found!</strong><br>
        Train the LSTM model first by running:<br>
        <code>python src/train.py</code><br><br>
        This will create <code>models/lstm_model.keras</code> and <code>models/scaler.pkl</code>
    </div>
    """, unsafe_allow_html=True)

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    run_prediction = st.button('🔮 Run LSTM Prediction', type='primary',
                               disabled=not model_exists,
                               use_container_width=True)

with col_btn2:
    run_forecast = st.button(f'🔭 Forecast Next {forecast_days} Days',
                             disabled=not model_exists,
                             use_container_width=True)

# ── Run Prediction ───────────────────────────────────────────
if run_prediction and model_exists:
    with st.spinner('🧠 Running LSTM model on historical data...'):
        try:
            dates, actual, predicted, rmse, mae, mape = generate_prediction(
                df, model_path, scaler_path, window_size
            )

            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">RMSE</div>
                    <div class="value blue">{format_currency(rmse)}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">MAE</div>
                    <div class="value blue">{format_currency(mae)}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">MAPE</div>
                    <div class="value blue">{mape:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Model</div>
                    <div class="value blue" style="font-size:1.2rem;">Stacked LSTM</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)

            # Prediction Chart
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=dates, y=actual,
                name='Actual Price',
                line=dict(color='#64ffda', width=2)
            ))
            fig_pred.add_trace(go.Scatter(
                x=dates, y=predicted,
                name='LSTM Predicted',
                line=dict(color='#ff6b6b', width=2, dash='dash')
            ))
            fig_pred.update_layout(
                title=f'Actual vs LSTM Predicted — {ticker} (RMSE: {format_currency(rmse)})',
                template='plotly_dark',
                height=500,
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1),
                margin=dict(l=50, r=20, t=60, b=30),
            )
            fig_pred.update_xaxes(gridcolor='rgba(102,126,234,0.1)')
            fig_pred.update_yaxes(gridcolor='rgba(102,126,234,0.1)',
                                  title_text='Price (INR)')
            st.plotly_chart(fig_pred, use_container_width=True)

            # Error Analysis
            st.markdown('<div class="section-header">📉 Prediction Error Analysis</div>',
                        unsafe_allow_html=True)

            errors = actual - predicted
            fig_err = make_subplots(rows=1, cols=2,
                                    subplot_titles=['Error Over Time',
                                                    'Error Distribution'])

            fig_err.add_trace(go.Scatter(
                x=dates, y=errors,
                name='Error', line=dict(color='#667eea', width=1),
                fill='tozeroy', fillcolor='rgba(102,126,234,0.1)'
            ), row=1, col=1)
            fig_err.add_hline(y=0, line_dash='dash', line_color='white',
                              opacity=0.3, row=1, col=1)

            fig_err.add_trace(go.Histogram(
                x=errors, nbinsx=50, name='Error Dist',
                marker_color='#667eea', opacity=0.8
            ), row=1, col=2)

            fig_err.update_layout(
                template='plotly_dark', height=350, showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                margin=dict(l=50, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_err, use_container_width=True)

            st.success('✅ Prediction complete!')

        except Exception as e:
            st.error(f'❌ Error during prediction: {str(e)}')
            st.info('Make sure the model was trained with the same window size.')

# ── Run Forecast ─────────────────────────────────────────────
if run_forecast and model_exists:
    with st.spinner(f'🔭 Forecasting next {forecast_days} days...'):
        try:
            future_prices = forecast_future_prices(
                df, model_path, scaler_path, forecast_days, window_size
            )

            # Create future dates (skip weekends)
            last_date = df.index[-1]
            future_dates = pd.bdate_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days
            )

            # Forecast Chart
            fig_forecast = go.Figure()

            # Historical (last 90 days)
            hist_days = min(90, len(df))
            fig_forecast.add_trace(go.Scatter(
                x=df.index[-hist_days:],
                y=df['Close'].iloc[-hist_days:],
                name='Historical',
                line=dict(color='#64ffda', width=2)
            ))

            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=future_dates,
                y=future_prices,
                name=f'{forecast_days}-Day Forecast',
                line=dict(color='#ffd93d', width=2.5, dash='dot'),
                mode='lines+markers',
                marker=dict(size=4)
            ))

            # Confidence band (simple ±2% band)
            upper = future_prices * 1.02
            lower = future_prices * 0.98
            fig_forecast.add_trace(go.Scatter(
                x=future_dates, y=upper,
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_dates, y=lower,
                line=dict(width=0), showlegend=False, hoverinfo='skip',
                fill='tonexty', fillcolor='rgba(255,217,61,0.1)'
            ))

            # Divider line
            fig_forecast.add_vline(
                x=last_date, line_dash='dash',
                line_color='white', opacity=0.4
            )

            fig_forecast.update_layout(
                title=f'{forecast_days}-Day Price Forecast — {ticker}',
                template='plotly_dark',
                height=500,
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1),
                margin=dict(l=50, r=20, t=60, b=30),
            )
            fig_forecast.update_xaxes(gridcolor='rgba(102,126,234,0.1)')
            fig_forecast.update_yaxes(gridcolor='rgba(102,126,234,0.1)',
                                      title_text='Price (INR)')

            st.plotly_chart(fig_forecast, use_container_width=True)

            # Forecast Summary
            st.markdown(f"""
            <div class="info-box">
                <strong>Forecast Summary</strong><br>
                Last Close: {format_currency(df['Close'].iloc[-1])} •
                Day 1 Forecast: {format_currency(future_prices[0])} •
                Day {forecast_days} Forecast: {format_currency(future_prices[-1])} •
                Range: {format_currency(future_prices.min())} — {format_currency(future_prices.max())}
            </div>
            """, unsafe_allow_html=True)

            # Forecast Table
            with st.expander('📋 View Detailed Forecast Table'):
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecasted Price (₹)': [f'₹{p:,.2f}' for p in future_prices],
                    'Change from Today (₹)': [f'{p - latest_close:+,.2f}'
                                              for p in future_prices],
                    'Change (%)': [f'{((p / latest_close) - 1) * 100:+.2f}%'
                                   for p in future_prices]
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            st.success(f'✅ {forecast_days}-day forecast complete!')

        except Exception as e:
            st.error(f'❌ Error during forecasting: {str(e)}')

# ══════════════════════════════════════════════════════════════
# SECTION 3: DATA TABLE
# ══════════════════════════════════════════════════════════════
with st.expander('📋 View Raw Data Table'):
    st.dataframe(
        df.tail(100).style.format({
            'Open': '₹{:,.2f}', 'High': '₹{:,.2f}',
            'Low': '₹{:,.2f}', 'Close': '₹{:,.2f}',
            'Volume': '{:,.0f}'
        }),
        use_container_width=True
    )



# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">

    Built with ❤️ using TensorFlow, Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
