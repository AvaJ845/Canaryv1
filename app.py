import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

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
default_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
default_etfs = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
default_dividend_stocks = ['JNJ', 'PG', 'KO', 'XOM', 'VZ']

# User input for custom tickers
st.sidebar.subheader("Add Custom Tickers")
custom_stocks = st.sidebar.text_input("Add custom stocks (comma-separated)", "")
custom_etfs = st.sidebar.text_input("Add custom ETFs (comma-separated)", "")
custom_dividend_stocks = st.sidebar.text_input("Add custom dividend stocks (comma-separated)", "")

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
selected_period = st.sidebar.selectbox("Select Time Period", list(time_periods.keys()))

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

# Function to analyze dividend stocks
def analyze_dividend_stocks(metrics):
    st.header("Dividend Analysis")
    
    # Filter out only dividend stocks with valid dividend data
    dividend_data = {}
    for ticker, metric in metrics.items():
        if ticker in dividend_stocks_list and metric['dividend_yield'] is not None:
            dividend_data[ticker] = {
                'Yield': metric['dividend_yield'],
                'Payout Ratio': metric['dividend_payout'] if metric['dividend_payout'] else 0,
                'Price': metric['current_price'],
                'Trend': metric['trend']
            }
    
    if dividend_data:
        # Create dataframe
        df = pd.DataFrame(dividend_data).T
        df = df.sort_values('Yield', ascending=False)
        
        # Plot dividend yields
        st.subheader("Dividend Yields Comparison")
        fig = px.bar(
            df.reset_index(), 
            x='index', 
            y='Yield',
            color='Yield',
            labels={'index': 'Ticker', 'Yield': 'Dividend Yield (%)'},
            title="Dividend Yields by Stock",
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display dividend data table
        st.subheader("Dividend Stocks Comparison")
        st.dataframe(df.style.background_gradient(subset=['Yield'], cmap='YlGn'))
    else:
        st.info("No dividend information available for the selected stocks.")

# Main app logic
def main():
    # Fetch stock data
    with st.spinner("Fetching data from Yahoo Finance..."):
        stock_data = fetch_stock_data(tickers, time_periods[selected_period], intervals[selected_interval])
    
    # Check if we have data
    if not stock_data:
        st.error("No data available for the selected tickers and time period. Please try different selections.")
        return
    
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