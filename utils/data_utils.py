#data_utils.py 
"""
Utilities for fetching and processing stock data.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_stock_data(tickers, period, interval):
    """
    Fetch stock data for the given tickers, period, and interval.
    
    Args:
        tickers (list): List of ticker symbols
        period (str): Time period (e.g., '1d', '1mo', '1y')
        interval (str): Data interval (e.g., '1m', '1h', '1d')
        
    Returns:
        dict: Dictionary of ticker symbols and their historical data
    """
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

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_prediction_data(ticker):
    """
    Fetch 2 years of daily data for prediction models.
    
    Args:
        ticker (str): Ticker symbol
        
    Returns:
        DataFrame: Historical data for the ticker
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", interval="1d")
        if not hist.empty:
            return hist
    except Exception as e:
        st.warning(f"Error fetching prediction data for {ticker}: {e}")
    return None

def create_features(data):
    """
    Create technical indicator features for prediction models.
    
    Args:
        data (DataFrame): Historical price data
        
    Returns:
        DataFrame: Data with added technical indicators
    """
    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()
    
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
    
    # Average Directional Index (ADX) - Simplified calculation
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
