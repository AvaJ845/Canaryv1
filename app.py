#
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Stock & ETF Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Stock & ETF Analysis Dashboard")
st.markdown("""
This app tracks and analyzes stocks and ETFs for notable changes, significant movements, and emerging trends.
Data is sourced from Yahoo Finance API.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Default tickers
default_stocks = ['COST', 'NVDA', 'PFE', 'NFLX', 'AMD', 'SHOP', 'NOV', 'TSM', 'LLY']
default_etfs = ['SDIV', 'PSEC', 'CLM', 'VOO', 'BRK-B', 'FXAIX', 'JEPI']
default_dividend_stocks = ['HD', 'MAIN', 'MO', 'MMM', 'DUK', 'ABBV', 'O', 'PBA', 'HON', 'XOM', 'SBUX', 'SOLV']

# User input for custom tickers
st.sidebar.subheader("Add Custom Tickers")
custom_stocks = st.sidebar.text_input("Add custom stocks (comma-separated)", "", key="custom_stocks_input")
custom_etfs = st.sidebar.text_input("Add custom ETFs (comma-separated)", "", key="custom_etfs_input")
custom_dividend_stocks = st.sidebar.text_input("Add custom dividend stocks (comma-separated)", "", key="custom_dividend_stocks_input")

# Process user inputs
if custom_stocks:
    stocks_list = default_stocks + [ticker.strip().upper() for ticker in custom_stocks.split(',')]
else:
    stocks_list = default_stocks

if custom_etfs:
    etfs_list = default_etfs + [ticker.strip().upper() for ticker in custom_etfs.split(',')]
else:
    etfs_list = default_etfs

if custom_dividend_stocks:
    dividend_stocks_list = default_dividend_stocks + [ticker.strip().upper() for ticker in custom_dividend_stocks.split(',')]
else:
    dividend_stocks_list = default_dividend_stocks

# Remove duplicates
stocks_list = list(dict.fromkeys(stocks_list))
etfs_list = list(dict.fromkeys(etfs_list))
dividend_stocks_list = list(dict.fromkeys(dividend_stocks_list))

# Combine all tickers for data fetching
tickers = stocks_list + etfs_list + dividend_stocks_list

# Time period selection
time_periods = {
    "1 Day": "1d",
    "5 Days": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}
selected_period = st.sidebar.selectbox("Select Time Period", list(time_periods.keys()), key="time_period_select")

# Data interval selection
intervals = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h",
    "1 Day": "1d",
    "1 Week": "1wk",
    "1 Month": "1mo"
}
selected_interval = st.sidebar.selectbox("Select Data Interval", list(intervals.keys()), key="interval_select")

# Prediction model settings
st.sidebar.subheader("Price Prediction Settings")
prediction_days = st.sidebar.slider("Prediction Horizon (Days)", 1, 30, 7, key="prediction_days_slider")
model_type = st.sidebar.selectbox(
    "Prediction Model Type",
    ["Linear Regression", "Random Forest"],
    key="model_type_select"
)

# Significance threshold for highlighting changes
significance_threshold = st.sidebar.slider("Significance Threshold (%)", 1.0, 10.0, 3.0, 0.1, key="significance_threshold_slider")

# Feature importance flag (for Random Forest only)
show_feature_importance = False
if model_type == "Random Forest":
    show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True, key="feature_importance_checkbox")

# Fetch more data for prediction models
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_prediction_data(ticker):
    try:
        # Get 2 years of data for better model training
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", interval="1d")
        if not hist.empty:
            return hist
    except Exception as e:
        st.warning(f"Error fetching prediction data for {ticker}: {e}")
    return None

# Function to create prediction model features
def create_features(data):
    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()
    
    # Technical indicators as features
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['20STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['20MA'] + (df['20STD'] * 2)
    df['Lower_Band'] = df['20MA'] - (df['20STD'] * 2)
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['20MA']
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    # Average Directional Index (ADX)
    # Simplified calculation
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Volume metrics
    df['Volume_ROC'] = df['Volume'].pct_change(periods=1) * 100
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    # Price change features
    df['Price_Change'] = df['Close'].pct_change() * 100
    df['Price_Change_5D'] = df['Close'].pct_change(periods=5) * 100
    
    # Day of week (0=Monday, 6=Sunday)
    df['Day_of_Week'] = pd.to_datetime(df.index).dayofweek
    
    # Target: Next day's closing price
    df['Target'] = df['Close'].shift(-1)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

# Function to build and train price prediction model
def build_price_prediction_model(ticker, prediction_days=7):
    # Fetch data
    data = fetch_prediction_data(ticker)
    if data is None or data.empty:
        return None, None, None
    
    # Create features
    features_df = create_features(data)
    if len(features_df) < 100:  # Not enough data
        return None, None, None
    
    # Prepare features and target
    feature_columns = [
        'MA5', 'MA10', 'MA20', 'MACD', 'MACD_Signal', 
        'BB_Width', 'RSI', 'ROC', 'ATR', 
        'Volume_ROC', 'Volume_Ratio', 
        'Price_Change', 'Price_Change_5D', 'Day_of_Week'
    ]
    
    X = features_df[feature_columns].values
    y = features_df['Target'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:  # Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    # Get latest feature values for prediction
    latest_features = features_df[feature_columns].iloc[-1].values.reshape(1, -1)
    latest_features_scaled = scaler.transform(latest_features)
    
    # Generate predictions for future days
    predictions = []
    latest_price = features_df['Close'].iloc[-1]
    current_features = latest_features_scaled.copy()
    
    # For visualization, include the last known price
    predictions.append(latest_price)
    
    for i in range(prediction_days):
        # Predict next price
        next_price = model.predict(current_features)[0]
        predictions.append(next_price)
        
        # Update features for next prediction (simplified approach)
        # In a real application, you would need to update all features
        # This is a simplified version that just updates price-related features
        price_change = ((next_price / latest_price) - 1) * 100
        current_features[0][feature_columns.index('Price_Change')] = price_change
        
        # Update latest price for next iteration
        latest_price = next_price
    
    # Feature importance (for Random Forest)
    feature_importance = None
    if model_type == "Random Forest":
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
    
    # Calculate accuracy metrics
    y_pred_test = model.predict(X_test_scaled)
    mse = np.mean((y_pred_test - y_test) ** 2)
    accuracy = 1 - np.mean(np.abs((y_test - y_pred_test) / y_test))
    
    # Prepare dates for visualization
    last_date = features_df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(prediction_days + 1)]
    
    return predictions, future_dates, {
        'mse': mse,
        'accuracy': accuracy,
        'feature_importance': feature_importance
    }
    # Calculate accuracy metrics
    y_pred_test = model.predict(X_test_scaled)
    mse = np.mean((y_pred_test - y_test) ** 2)
    accuracy = 1 - np.mean(np.abs((y_test - y_pred_test) / y_test))
    
    # Prepare dates for visualization
    last_date = features_df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(prediction_days + 1)]
    
    return predictions, future_dates, {
        'mse': mse,
        'accuracy': accuracy,
        'feature_importance': feature_importance
    }

# Data interval selection
intervals = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h",
    "1 Day": "1d",
    "1 Week": "1wk",
    "1 Month": "1mo"
}
selected_interval = st.sidebar.selectbox("Select Data Interval", list(intervals.keys()))

# Significance threshold for highlighting changes
significance_threshold = st.sidebar.slider("Significance Threshold (%)", 1.0, 10.0, 3.0, 0.1)

# Feature importance flag (for Random Forest only)
show_feature_importance = False
if model_type == "Random Forest":
    show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True)

# Function to fetch stock data
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_stock_data(tickers, period, interval):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            if not hist.empty:
                data[ticker] = hist
        except Exception as e:
            st.warning(f"Error fetching data for {ticker}: {e}")
    return data

# Fetch more data for prediction models
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_prediction_data(ticker):
    try:
        # Get 2 years of data for better model training
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", interval="1d")
        if not hist.empty:
            return hist
    except Exception as e:
        st.warning(f"Error fetching prediction data for {ticker}: {e}")
    return None

# Function to create prediction model features
def create_features(data):
    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()
    
    # Technical indicators as features
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['20STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['20MA'] + (df['20STD'] * 2)
    df['Lower_Band'] = df['20MA'] - (df['20STD'] * 2)
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['20MA']
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    # Average Directional Index (ADX)
    # Simplified calculation
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Volume metrics
    df['Volume_ROC'] = df['Volume'].pct_change(periods=1) * 100
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    # Price change features
    df['Price_Change'] = df['Close'].pct_change() * 100
    df['Price_Change_5D'] = df['Close'].pct_change(periods=5) * 100
    
    # Day of week (0=Monday, 6=Sunday)
    df['Day_of_Week'] = pd.to_datetime(df.index).dayofweek
    
    # Target: Next day's closing price
    df['Target'] = df['Close'].shift(-1)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

# Function to build and train price prediction model
def build_price_prediction_model(ticker, prediction_days=7):
    # Fetch data
    data = fetch_prediction_data(ticker)
    if data is None or data.empty:
        return None, None, None
    
    # Create features
    features_df = create_features(data)
    if len(features_df) < 100:  # Not enough data
        return None, None, None
    
    # Prepare features and target
    feature_columns = [
        'MA5', 'MA10', 'MA20', 'MACD', 'MACD_Signal', 
        'BB_Width', 'RSI', 'ROC', 'ATR', 
        'Volume_ROC', 'Volume_Ratio', 
        'Price_Change', 'Price_Change_5D', 'Day_of_Week'
    ]
    
    X = features_df[feature_columns].values
    y = features_df['Target'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:  # Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    # Get latest feature values for prediction
    latest_features = features_df[feature_columns].iloc[-1].values.reshape(1, -1)
    latest_features_scaled = scaler.transform(latest_features)
    
    # Generate predictions for future days
    predictions = []
    latest_price = features_df['Close'].iloc[-1]
    current_features = latest_features_scaled.copy()
    
    # For visualization, include the last known price
    predictions.append(latest_price)
    
    for i in range(prediction_days):
        # Predict next price
        next_price = model.predict(current_features)[0]
        predictions.append(next_price)
        
        # Update features for next prediction (simplified approach)
        # In a real application, you would need to update all features
        # This is a simplified version that just updates price-related features
        price_change = ((next_price / latest_price) - 1) * 100
        current_features[0][feature_columns.index('Price_Change')] = price_change
        
        # Update latest price for next iteration
        latest_price = next_price
    
    # Feature importance (for Random Forest)
    feature_importance = None
    if model_type == "Random Forest":
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
    
    # Calculate accuracy metrics
    y_pred_test = model.predict(X_test_scaled)
    mse = np.mean((y_pred_test - y_test) ** 2)
    accuracy = 1 - np.mean(np.abs((y_test - y_pred_test) / y_test))
    
    # Prepare dates for visualization
    last_date = features_df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(prediction_days + 1)]
    
    return predictions, future_dates, {
        'mse': mse,
        'accuracy': accuracy,
        'feature_importance': feature_importance
    }

# Function to render price prediction
def render_price_prediction(ticker, ticker_tab):
    with ticker_tab:
        st.subheader(f"Price Prediction ({prediction_days} days)")
        
        # Build prediction model
        with st.spinner(f"Building {model_type} model for {ticker}..."):
            predictions, future_dates, model_metrics = build_price_prediction_model(ticker, prediction_days)
        
        if predictions is None:
            st.warning(f"Insufficient data to build prediction model for {ticker}")
            return
        
        # Display model metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", f"{model_metrics['accuracy']*100:.2f}%", key=f"accuracy_metric_{ticker}")
        with col2:
            st.metric("Mean Squared Error", f"{model_metrics['mse']:.4f}", key=f"mse_metric_{ticker}")
        
        # Display feature importance if available
        if show_feature_importance and model_metrics['feature_importance'] is not None:
            st.subheader("Feature Importance")
            
            # Sort feature importance
            sorted_importance = sorted(
                model_metrics['feature_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Plot feature importance
            fig = px.bar(
                x=[item[1] for item in sorted_importance],
                y=[item[0] for item in sorted_importance],
                orientation='h',
                labels={'x': 'Importance', 'y': 'Feature'},
                title="Feature Importance"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Plot predictions
        st.subheader("Price Prediction Chart")
        
        # Get historical data for comparison
        hist_data = fetch_prediction_data(ticker)
        if hist_data is not None:
            # Show last 30 days and predictions
            recent_data = hist_data.iloc[-30:]['Close']
            
            # Create plot
            fig = go.Figure()
            
            # Add historical prices
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data.values,
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Add predictions
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='red', dash='dash')
            ))
            
            # Add confidence interval (simple approximation)
            upper_bound = [price * (1 + model_metrics['mse'] * 0.5) for price in predictions]
            lower_bound = [price * (1 - model_metrics['mse'] * 0.5) for price in predictions]
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=lower_bound,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                name='Confidence Interval'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{ticker} Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction data in table format
            prediction_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': [f"${price:.2f}" for price in predictions],
                'Lower Bound': [f"${price:.2f}" for price in lower_bound],
                'Upper Bound': [f"${price:.2f}" for price in upper_bound]
            })
            
            st.dataframe(prediction_df, key=f"prediction_df_{ticker}")

# Function to calculate key metrics and statistics
def calculate_metrics(stock_data, ticker_type=None):
    metrics = {}
    for ticker, data in stock_data.items():
        if data.empty:
            continue
            
        # Calculate returns
        data['Daily_Return'] = data['Close'].pct_change() * 100
        
        # Basic metrics
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2] if len(data) > 1 else None
        daily_change = data['Daily_Return'].iloc[-1] if len(data) > 1 else None
        
        # Calculate volatility (standard deviation of returns)
        volatility = data['Daily_Return'].std()
        
        # Calculate volume change
        volume_change = ((data['Volume'].iloc[-1] / data['Volume'].iloc[-2]) - 1) * 100 if len(data) > 1 else None
        
        # Calculate moving averages
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Identify trend based on moving averages
        if len(data) > 50:
            if data['MA5'].iloc[-1] > data['MA20'].iloc[-1] > data['MA50'].iloc[-1]:
                trend = "Strong Uptrend"
            elif data['MA5'].iloc[-1] > data['MA20'].iloc[-1]:
                trend = "Uptrend"
            elif data['MA5'].iloc[-1] < data['MA20'].iloc[-1] < data['MA50'].iloc[-1]:
                trend = "Strong Downtrend"
            elif data['MA5'].iloc[-1] < data['MA20'].iloc[-1]:
                trend = "Downtrend"
            else:
                trend = "Sideways"
        else:
            trend = "Insufficient Data"
        
        # Calculate RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        current_rsi = data['RSI'].iloc[-1] if not pd.isna(data['RSI'].iloc[-1]) else None
        
        # Identify support and resistance levels
        recent_highs = data['High'].tail(20)
        recent_lows = data['Low'].tail(20)
        resistance = recent_highs.max()
        support = recent_lows.min()
        
        # For dividend stocks, fetch dividend info
        dividend_yield = None
        dividend_payout = None
        dividend_history = None
        
        if ticker_type == 'dividend' or ticker in dividend_stocks_list:
            try:
                stock_info = yf.Ticker(ticker).info
                if 'dividendYield' in stock_info and stock_info['dividendYield'] is not None:
                    dividend_yield = stock_info['dividendYield'] * 100  # Convert to percentage
                if 'payoutRatio' in stock_info and stock_info['payoutRatio'] is not None:
                    dividend_payout = stock_info['payoutRatio'] * 100  # Convert to percentage
                
                # Get dividend history
                dividend_history = yf.Ticker(ticker).dividends
                if not dividend_history.empty:
                    dividend_history = dividend_history.reset_index()
                    dividend_history.columns = ['Date', 'Dividend']
                    dividend_history = dividend_history.sort_values('Date', ascending=False)
                    dividend_history = dividend_history.head(5)  # Last 5 dividends
            except Exception as e:
                st.warning(f"Error fetching dividend data for {ticker}: {e}")
        
        # Store metrics
        metrics[ticker] = {
            'data': data,
            'current_price': current_price,
            'previous_price': previous_price,
            'daily_change': daily_change,
            'volatility': volatility,
            'volume_change': volume_change,
            'trend': trend,
            'rsi': current_rsi,
            'support': support,
            'resistance': resistance,
            'dividend_yield': dividend_yield,
            'dividend_payout': dividend_payout,
            'dividend_history': dividend_history
        }
    
    return metrics

# Function to identify notable events
def identify_notable_events(metrics, threshold):
    notable_events = []
    
    for ticker, metric in metrics.items():
        # Check for significant price changes
        if metric['daily_change'] and abs(metric['daily_change']) > threshold:
            direction = "up" if metric['daily_change'] > 0 else "down"
            notable_events.append({
                'ticker': ticker,
                'event': f"Significant price movement {direction}",
                'value': f"{metric['daily_change']:.2f}%",
                'importance': abs(metric['daily_change']) / threshold
            })
        
        # Check for unusual volume
        if metric['volume_change'] and abs(metric['volume_change']) > threshold * 2:
            direction = "increase" if metric['volume_change'] > 0 else "decrease"
            notable_events.append({
                'ticker': ticker,
                'event': f"Unusual volume {direction}",
                'value': f"{metric['volume_change']:.2f}%",
                'importance': abs(metric['volume_change']) / (threshold * 2)
            })
        
        # Check for overbought/oversold conditions
        if metric['rsi'] is not None:
            if metric['rsi'] > 70:
                notable_events.append({
                    'ticker': ticker,
                    'event': "Potentially overbought",
                    'value': f"RSI: {metric['rsi']:.2f}",
                    'importance': (metric['rsi'] - 70) / 10
                })
            elif metric['rsi'] < 30:
                notable_events.append({
                    'ticker': ticker,
                    'event': "Potentially oversold",
                    'value': f"RSI: {metric['rsi']:.2f}",
                    'importance': (30 - metric['rsi']) / 10
                })
        
        # Check for trend changes
        if metric['trend'] in ["Strong Uptrend", "Strong Downtrend"]:
            notable_events.append({
                'ticker': ticker,
                'event': f"Strong {metric['trend'].lower()}",
                'value': metric['trend'],
                'importance': 2.0
            })
    
    # Sort by importance
    return sorted(notable_events, key=lambda x: x['importance'], reverse=True)

# Main dashboard components
def render_dashboard(stock_data, metrics, notable_events):
    # Create main tabs for Stocks, ETFs, and Dividends
    stock_tab, etf_tab, dividend_tab = st.tabs(["Stocks", "ETFs", "Dividend Stocks"])
    
    # Filter metrics by ticker type
    stock_metrics = {ticker: metric for ticker, metric in metrics.items() if ticker in stocks_list}
    etf_metrics = {ticker: metric for ticker, metric in metrics.items() if ticker in etfs_list}
    dividend_metrics = {ticker: metric for ticker, metric in metrics.items() if ticker in dividend_stocks_list}
    
    # Filter notable events by ticker type
    stock_events = [event for event in notable_events if event['ticker'] in stocks_list]
    etf_events = [event for event in notable_events if event['ticker'] in etfs_list]
    dividend_events = [event for event in notable_events if event['ticker'] in dividend_stocks_list]
    
    # === STOCKS TAB ===
    with stock_tab:
        # Overview metrics
        st.header("Stocks Overview")
        
        # Display metrics in columns (limit to 5 per row to avoid squeezing)
        if stock_metrics:
            for i in range(0, len(stock_metrics), 5):
                cols = st.columns(min(5, len(stock_metrics) - i))
                for j, ticker in enumerate(list(stock_metrics.keys())[i:i+5]):
                    metric = stock_metrics[ticker]
                    with cols[j]:
                        st.metric(
                            label=ticker,
                            value=f"${metric['current_price']:.2f}",
                            delta=f"{metric['daily_change']:.2f}%" if metric['daily_change'] else "N/A"
                        )
        
            # Notable events section
            st.header("Notable Stock Events")
            
            if stock_events:
                events_df = pd.DataFrame(stock_events)
                
                # Format the dataframe for display
                events_df = events_df[['ticker', 'event', 'value', 'importance']]
                events_df.columns = ['Ticker', 'Event', 'Value', 'Significance']
                
                # Apply styling
                st.dataframe(events_df.style.background_gradient(subset=['Significance'], cmap='YlOrRd'))
            else:
                st.info("No notable stock events detected based on current significance threshold.")
            
            # Individual stock analysis
            st.header("Individual Stock Analysis")
            
            # Create tabs for each stock
            stock_ticker_tabs = st.tabs(list(stock_metrics.keys()))
            
            for i, (ticker, ticker_tab) in enumerate(zip(stock_metrics.keys(), stock_ticker_tabs)):
                render_ticker_analysis(ticker, stock_metrics[ticker], ticker_tab)
        else:
            st.info("No stock data available. Please add stocks in the sidebar.")
    
    # === ETFs TAB ===
    with etf_tab:
        # Overview metrics
        st.header("ETFs Overview")
        
        # Display metrics in columns (limit to 5 per row to avoid squeezing)
        if etf_metrics:
            for i in range(0, len(etf_metrics), 5):
                cols = st.columns(min(5, len(etf_metrics) - i))
                for j, ticker in enumerate(list(etf_metrics.keys())[i:i+5]):
                    metric = etf_metrics[ticker]
                    with cols[j]:
                        st.metric(
                            label=ticker,
                            value=f"${metric['current_price']:.2f}",
                            delta=f"{metric['daily_change']:.2f}%" if metric['daily_change'] else "N/A"
                        )
        
            # Notable events section
            st.header("Notable ETF Events")
            
            if etf_events:
                events_df = pd.DataFrame(etf_events)
                
                # Format the dataframe for display
                events_df = events_df[['ticker', 'event', 'value', 'importance']]
                events_df.columns = ['Ticker', 'Event', 'Value', 'Significance']
                
                # Apply styling
                st.dataframe(events_df.style.background_gradient(subset=['Significance'], cmap='YlOrRd'))
            else:
                st.info("No notable ETF events detected based on current significance threshold.")
            
            # Individual ETF analysis
            st.header("Individual ETF Analysis")
            
            # Create tabs for each ETF
            etf_ticker_tabs = st.tabs(list(etf_metrics.keys()))
            
            for i, (ticker, ticker_tab) in enumerate(zip(etf_metrics.keys(), etf_ticker_tabs)):
                render_ticker_analysis(ticker, etf_metrics[ticker], ticker_tab)
        else:
            st.info("No ETF data available. Please add ETFs in the sidebar.")
    
    # === DIVIDEND STOCKS TAB ===
    with dividend_tab:
        # Overview metrics
        st.header("Dividend Stocks Overview")
        
        # Display metrics in columns (limit to 5 per row to avoid squeezing)
        if dividend_metrics:
            for i in range(0, len(dividend_metrics), 5):
                cols = st.columns(min(5, len(dividend_metrics) - i))
                for j, ticker in enumerate(list(dividend_metrics.keys())[i:i+5]):
                    metric = dividend_metrics[ticker]
                    with cols[j]:
                        st.metric(
                            label=ticker,
                            value=f"${metric['current_price']:.2f}",
                            delta=f"{metric['daily_change']:.2f}%" if metric['daily_change'] else "N/A"
                        )
        
            # Notable events section
            st.header("Notable Dividend Stock Events")
            
            if dividend_events:
                events_df = pd.DataFrame(dividend_events)
                
                # Format the dataframe for display
                events_df = events_df[['ticker', 'event', 'value', 'importance']]
                events_df.columns = ['Ticker', 'Event', 'Value', 'Significance']
                
                # Apply styling
                st.dataframe(events_df.style.background_gradient(subset=['Significance'], cmap='YlOrRd'))
            else:
                st.info("No notable dividend stock events detected based on current significance threshold.")
            
            # Individual dividend stock analysis
            st.header("Individual Dividend Stock Analysis")
            
            # Create tabs for each dividend stock
            dividend_ticker_tabs = st.tabs(list(dividend_metrics.keys()))
            
            for i, (ticker, ticker_tab) in enumerate(zip(dividend_metrics.keys(), dividend_ticker_tabs)):
                render_ticker_analysis(ticker, dividend_metrics[ticker], ticker_tab, show_dividend=True)
        else:
            st.info("No dividend stock data available. Please add dividend stocks in the sidebar.")

# Function to render individual ticker analysis
def render_ticker_analysis(ticker, metric, ticker_tab, show_dividend=False):
    with ticker_tab:
        # Create two columns for chart and metrics
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Price chart with moving averages
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=metric['data'].index,
                open=metric['data']['Open'],
                high=metric['data']['High'],
                low=metric['data']['Low'],
                close=metric['data']['Close'],
                name=ticker
            ))
            
            # Add moving averages
            if 'MA5' in metric['data'].columns and not metric['data']['MA5'].isna().all():
                fig.add_trace(go.Scatter(x=metric['data'].index, y=metric['data']['MA5'], mode='lines', name='MA5', line=dict(color='blue')))
            
            if 'MA20' in metric['data'].columns and not metric['data']['MA20'].isna().all():
                fig.add_trace(go.Scatter(x=metric['data'].index, y=metric['data']['MA20'], mode='lines', name='MA20', line=dict(color='orange')))
            
            if 'MA50' in metric['data'].columns and not metric['data']['MA50'].isna().all():
                fig.add_trace(go.Scatter(x=metric['data'].index, y=metric['data']['MA50'], mode='lines', name='MA50', line=dict(color='green')))
            
            # Update layout
            fig.update_layout(
                title=f'{ticker} Price Chart',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            volume_fig = px.bar(
                metric['data'],
                x=metric['data'].index,
                y='Volume',
                title=f'{ticker} Volume'
            )
            st.plotly_chart(volume_fig, use_container_width=True)
        
        with col2:
            # Key metrics
            st.subheader("Key Metrics")
            
            metrics_data = {
                "Current Price": f"${metric['current_price']:.2f}",
                "Daily Change": f"{metric['daily_change']:.2f}%" if metric['daily_change'] else "N/A",
                "Volatility": f"{metric['volatility']:.2f}%",
                "Volume Change": f"{metric['volume_change']:.2f}%" if metric['volume_change'] else "N/A",
                "Current Trend": metric['trend'],
                "RSI": f"{metric['rsi']:.2f}" if metric['rsi'] else "N/A",
                "Support Level": f"${metric['support']:.2f}",
                "Resistance Level": f"${metric['resistance']:.2f}"
            }
            
            for k, v in metrics_data.items():
                st.text(f"{k}: {v}")
            
            # Add dividend metrics if this is a dividend stock
            if show_dividend and metric['dividend_yield'] is not None:
                st.subheader("Dividend Information")
                
                dividend_data = {
                    "Dividend Yield": f"{metric['dividend_yield']:.2f}%" if metric['dividend_yield'] else "N/A",
                    "Payout Ratio": f"{metric['dividend_payout']:.2f}%" if metric['dividend_payout'] else "N/A"
                }
                
                for k, v in dividend_data.items():
                    st.text(f"{k}: {v}")
                
                # Display recent dividend history if available
                if metric['dividend_history'] is not None and not metric['dividend_history'].empty:
                    st.subheader("Recent Dividends")
                    st.dataframe(metric['dividend_history'])
            
            # Trend analysis
            st.subheader("Trend Analysis")
            
            # Determine background color based on trend
            trend_colors = {
                "Strong Uptrend": "lightgreen",
                "Uptrend": "lightgreen",
                "Strong Downtrend": "lightcoral",
                "Downtrend": "lightcoral",
                "Sideways": "lightyellow",
                "Insufficient Data": "lightgray"
            }
            
            trend_color = trend_colors.get(metric['trend'], "lightgray")
            
            st.markdown(
                f"""
                <div style="background-color: {trend_color}; padding: 10px; border-radius: 5px;">
                    <h4 style="margin: 0;">{metric['trend']}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # RSI gauge chart if available
            if metric['rsi']:
                st.subheader("RSI Indicator")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=metric['rsi'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "RSI"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "gray"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': metric['rsi']
                        }
                    }
                ))
                
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
        
        # Add price prediction 
        render_price_prediction(ticker, ticker_tab)

# Correlation analysis
def correlation_analysis(stock_data):
    st.header("Correlation Analysis")
    
    # Prepare data for correlation matrix
    close_prices = pd.DataFrame()
    
    for ticker, data in stock_data.items():
        if not data.empty:
            close_prices[ticker] = data['Close']
    
    if not close_prices.empty and len(close_prices.columns) > 1:
        # Calculate correlation matrix
        correlation_matrix = close_prices.corr()
        
        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Price Correlation Matrix"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Identify highly correlated and inversely correlated pairs
        st.subheader("Notable Correlations")
        
        # Get upper triangle of correlation matrix (excluding diagonal)
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        
        # Find high positive correlations (> 0.8)
        high_pos_corr = [(i, j, upper_tri.loc[i, j]) 
                         for i in upper_tri.index 
                         for j in upper_tri.columns 
                         if upper_tri.loc[i, j] > 0.8]
        
        # Find high negative correlations (< -0.5)
        high_neg_corr = [(i, j, upper_tri.loc[i, j]) 
                         for i in upper_tri.index 
                         for j in upper_tri.columns 
                         if upper_tri.loc[i, j] < -0.5]
        
        if high_pos_corr:
            st.subheader("Highly Correlated Pairs")
            for i, j, corr in sorted(high_pos_corr, key=lambda x: x[2], reverse=True):
                st.write(f"{i} and {j}: {corr:.2f}")
        
        if high_neg_corr:
            st.subheader("Inversely Correlated Pairs")
            for i, j, corr in sorted(high_neg_corr, key=lambda x: x[2]):
                st.write(f"{i} and {j}: {corr:.2f}")
    else:
        st.info("Insufficient data for correlation analysis. Please add more tickers or extend the time period.")

None
    
    # Calculate metrics
    metrics = calculate_metrics(stock_data)
    
    # Identify notable events
    notable_events = identify_notable_events(metrics, significance_threshold)
    
    # Render dashboard with tabs
    render_dashboard(stock_data, metrics, notable_events)
    
    # Add additional analysis for dividend stocks
    dividend_metrics = {ticker: metric for ticker, metric in metrics.items() if ticker in dividend_stocks_list}
    if dividend_metrics:
        analyze_dividend_stocks(metrics)
    
    # Correlation analysis
    correlation_analysis(stock_data)

if __name__ == "__main__":
    main()