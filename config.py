#config.py
"""
Configuration settings for the Stock & ETF Analysis Dashboard.
"""

# Default stock tickers
DEFAULT_STOCKS = ['COST', 'NVDA', 'PFE', 'NFLX', 'AMD', 'SHOP', 'NOV', 'TSM', 'LLY']

# Default ETF tickers
DEFAULT_ETFS = ['SDIV', 'PSEC', 'CLM', 'VOO', 'BRK-B', 'FXAIX', 'JEPI']

# Default dividend stock tickers
DEFAULT_DIVIDEND_STOCKS = ['HD', 'MAIN', 'MO', 'MMM', 'DUK', 'ABBV', 'O', 'PBA', 'HON', 'XOM', 'SBUX', 'SOLV']

# Time period options for data fetching
TIME_PERIODS = {
    "1 Day": "1d",
    "5 Days": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}

# Interval options for data fetching
INTERVALS = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h",
    "1 Day": "1d",
    "1 Week": "1wk",
    "1 Month": "1mo"
}

# Feature columns for prediction model
FEATURE_COLUMNS = [
    'MA5', 'MA10', 'MA20', 'MACD', 'MACD_Signal', 
    'BB_Width', 'RSI', 'ROC', 'ATR', 
    'Volume_ROC', 'Volume_Ratio', 
    'Price_Change', 'Price_Change_5D', 'Day_of_Week'
]

# Default settings for Streamlit page
PAGE_CONFIG = {
    "page_title": "Stock & ETF Analysis Dashboard",
    "page_icon": "📈",
    "layout": "wide"
}

# App title and description
APP_TITLE = "Stock & ETF Analysis Dashboard"
APP_DESCRIPTION = """
This app tracks and analyzes stocks and ETFs for notable changes, significant movements, and emerging trends.
Data is sourced from Yahoo Finance API.
"""
