"""
Main application file for the Stock & ETF Analysis Dashboard.
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config

# Import utilities
from utils.data_utils import fetch_stock_data
from utils.metrics_utils import calculate_metrics, identify_notable_events

# Import components
from components.dashboard import render_dashboard
from components.dividend_analysis import analyze_dividend_stocks
from components.correlation import correlation_analysis

# Set page configuration
st.set_page_config(**config.PAGE_CONFIG)

# App title and description
st.title(config.APP_TITLE)
st.markdown(config.APP_DESCRIPTION)

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# User input for custom tickers
st.sidebar.subheader("Add Custom Tickers")
custom_stocks = st.sidebar.text_input("Add custom stocks (comma-separated)", "", key="custom_stocks_input")
custom_etfs = st.sidebar.text_input("Add custom ETFs (comma-separated)", "", key="custom_etfs_input")
custom_dividend_stocks = st.sidebar.text_input("Add custom dividend stocks (comma-separated)", "", key="custom_dividend_stocks_input")

# Process user inputs
if custom_stocks:
    stocks_list = config.DEFAULT_STOCKS + [ticker.strip().upper() for ticker in custom_stocks.split(',')]
else:
    stocks_list = config.DEFAULT_STOCKS

if custom_etfs:
    etfs_list = config.DEFAULT_ETFS + [ticker.strip().upper() for ticker in custom_etfs.split(',')]
else:
    etfs_list = config.DEFAULT_ETFS

if custom_dividend_stocks:
    dividend_stocks_list = config.DEFAULT_DIVIDEND_STOCKS + [ticker.strip().upper() for ticker in custom_dividend_stocks.split(',')]
else:
    dividend_stocks_list = config.DEFAULT_DIVIDEND_STOCKS

# Remove duplicates
stocks_list = list(dict.fromkeys(stocks_list))
etfs_list = list(dict.fromkeys(etfs_list))
dividend_stocks_list = list(dict.fromkeys(dividend_stocks_list))

# Combine all tickers for data fetching
tickers = stocks_list + etfs_list + dividend_stocks_list

# Time period selection
selected_period = st.sidebar.selectbox("Select Time Period", list(config.TIME_PERIODS.keys()), key="time_period_select")

# Data interval selection
selected_interval = st.sidebar.selectbox("Select Data Interval", list(config.INTERVALS.keys()), key="interval_select")

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

def main():
    """Main function to run the application."""
    # Get the selected period and interval values
    period = config.TIME_PERIODS[selected_period]
    interval = config.INTERVALS[selected_interval]
    
    # Fetch data
    with st.spinner('Fetching stock data...'):
        stock_data = fetch_stock_data(tickers, period, interval)
    
    # Calculate metrics
    metrics = calculate_metrics(stock_data, dividend_stocks_list)
    
    # Identify notable events
    notable_events = identify_notable_events(metrics, significance_threshold)
    
    # Render dashboard with tabs
    render_dashboard(stock_data, metrics, notable_events, stocks_list, etfs_list, dividend_stocks_list)
    
    # Add additional analysis for dividend stocks
    dividend_metrics = {ticker: metric for ticker, metric in metrics.items() if ticker in dividend_stocks_list}
    if dividend_metrics:
        analyze_dividend_stocks(dividend_metrics)
    
    # Correlation analysis
    correlation_analysis(stock_data)

if __name__ == "__main__":
    main()