"""
Main application file for the Stock & ETF Analysis Dashboard.
"""

import streamlit as st
import warnings
import traceback
warnings.filterwarnings('ignore')

# Set page title and icon
st.set_page_config(
    page_title="Stock & ETF Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

try:
    # Import configuration (replace with direct imports if using a flat structure)
    try:
        import config
        # Use configuration values from config.py
        DEFAULT_STOCKS = config.DEFAULT_STOCKS
        DEFAULT_ETFS = config.DEFAULT_ETFS
        DEFAULT_DIVIDEND_STOCKS = config.DEFAULT_DIVIDEND_STOCKS
        TIME_PERIODS = config.TIME_PERIODS
        INTERVALS = config.INTERVALS
    except ImportError:
        # Fallback if config.py is not found
        st.warning("Config module not found, using default values")
        # Default tickers
        DEFAULT_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        DEFAULT_ETFS = ['SPY', 'QQQ', 'VTI', 'GLD', 'TLT']
        DEFAULT_DIVIDEND_STOCKS = ['KO', 'JNJ', 'PG', 'VZ', 'T']
        
        # Time period and interval options
        TIME_PERIODS = {
            "1 Day": "1d",
            "5 Days": "5d",
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
        }
        INTERVALS = {
            "1 Day": "1d",
            "1 Hour": "1h",
            "5 Minutes": "5m",
        }
    
    # Import utilities (adapt imports based on your structure)
    try:
        from utils.data_utils import fetch_stock_data
        from utils.metrics_utils import calculate_metrics, identify_notable_events
    except ImportError:
        try:
            from data_utils import fetch_stock_data
            from metrics_utils import calculate_metrics, identify_notable_events
        except ImportError:
            st.error("Cannot import data utilities. Check file structure and imports.")
            st.stop()
    
    # Import components (adapt imports based on your structure)
    try:
        from components.dashboard import render_dashboard
        from components.dividend_analysis import analyze_dividend_stocks
        from components.correlation import correlation_analysis
    except ImportError:
        try:
            from dashboard_components import render_dashboard
            from dividend_analysis import analyze_dividend_stocks
            from correlation_analysis import correlation_analysis
        except ImportError:
            st.error("Cannot import component modules. Check file structure and imports.")
            st.stop()

    # App title and description
    st.title("Stock & ETF Analysis Dashboard")
    st.markdown("""
    This app tracks and analyzes stocks and ETFs for notable changes, significant movements, and emerging trends.
    Data is sourced from Yahoo Finance API.
    """)
    
    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")
    
    # User input for custom tickers
    st.sidebar.subheader("Add Custom Tickers")
    custom_stocks = st.sidebar.text_input("Add custom stocks (comma-separated)", "", key="custom_stocks_input")
    custom_etfs = st.sidebar.text_input("Add custom ETFs (comma-separated)", "", key="custom_etfs_input")
    custom_dividend_stocks = st.sidebar.text_input("Add custom dividend stocks (comma-separated)", "", key="custom_dividend_stocks_input")
    
    # Process user inputs
    if custom_stocks:
        stocks_list = DEFAULT_STOCKS + [ticker.strip().upper() for ticker in custom_stocks.split(',')]
    else:
        stocks_list = DEFAULT_STOCKS
    
    if custom_etfs:
        etfs_list = DEFAULT_ETFS + [ticker.strip().upper() for ticker in custom_etfs.split(',')]
    else:
        etfs_list = DEFAULT_ETFS
    
    if custom_dividend_stocks:
        dividend_stocks_list = DEFAULT_DIVIDEND_STOCKS + [ticker.strip().upper() for ticker in custom_dividend_stocks.split(',')]
    else:
        dividend_stocks_list = DEFAULT_DIVIDEND_STOCKS
    
    # Remove duplicates
    stocks_list = list(dict.fromkeys(stocks_list))
    etfs_list = list(dict.fromkeys(etfs_list))
    dividend_stocks_list = list(dict.fromkeys(dividend_stocks_list))
    
    # Combine all tickers for data fetching
    tickers = stocks_list + etfs_list + dividend_stocks_list
    
    # Time period selection
    selected_period = st.sidebar.selectbox("Select Time Period", list(TIME_PERIODS.keys()), key="time_period_select")
    
    # Data interval selection
    selected_interval = st.sidebar.selectbox("Select Data Interval", list(INTERVALS.keys()), key="interval_select")
    
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
        try:
            # Get the selected period and interval values
            period = TIME_PERIODS[selected_period]
            interval = INTERVALS[selected_interval]
            
            # Fetch data with progress indicator
            progress_bar = st.progress(0)
            st.info('Fetching stock data... This may take a moment.')
            
            stock_data = {}
            for i, ticker in enumerate(tickers):
                try:
                    progress_bar.progress((i + 1) / len(tickers))
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=period, interval=interval)
                    if not hist.empty:
                        stock_data[ticker] = hist
                except Exception as e:
                    st.warning(f"Error fetching data for {ticker}: {str(e)}")
            
            progress_bar.progress(100)
            
            if not stock_data:
                st.error("No data was fetched. Please check your internet connection or try different tickers.")
                return
                
            st.success(f"Successfully fetched data for {len(stock_data)} tickers")
            
            # Calculate metrics
            metrics = calculate_metrics(stock_data, dividend_stocks_list)
            
            # Identify notable events
            notable_events = identify_notable_events(metrics, significance_threshold)
            
            # Render dashboard with tabs
            render_dashboard(stock_data, metrics, notable_events, stocks_list, etfs_list, dividend_stocks_list)
            
            # Add additional analysis for dividend stocks
            dividend_metrics = {ticker: metric for ticker, metric in metrics.items() if ticker in dividend_stocks_list}
            
            try:
                if dividend_metrics:
                    analyze_dividend_stocks(dividend_metrics)
            except Exception as e:
                st.error(f"Error in dividend analysis: {str(e)}")
                st.code(traceback.format_exc())
            
            # Correlation analysis
            try:
                correlation_analysis(stock_data)
            except Exception as e:
                st.error(f"Error in correlation analysis: {str(e)}")
                st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"An error occurred in the main function: {str(e)}")
            st.code(traceback.format_exc())

    # Import yfinance here to make it available in the main function
    import yfinance as yf
    
    if __name__ == "__main__":
        main()

except Exception as e:
    st.error(f"An error occurred during application startup: {str(e)}")
    st.code(traceback.format_exc())