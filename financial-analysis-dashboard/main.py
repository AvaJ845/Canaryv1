# main.py
# This is the main entry point for our Streamlit application
# It imports modules from other files and sets up the main UI structure

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Import our custom modules
from stock_analysis import get_stock_data, analyze_stock, forecast_stock
from crypto_analysis import get_crypto_data, analyze_crypto, forecast_crypto
from dividend_analysis import get_dividend_data, analyze_dividend, rate_dividend
from etf_analysis import get_etf_data, analyze_etf
from portfolio import load_portfolio_data, calculate_portfolio_metrics
from ui_components import sidebar_filters, header_section, display_metrics

# Set page configuration
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
    .green-text {
        color: #4CAF50;
    }
    .red-text {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
header_section("Financial Analysis Dashboard", 
               "Track and analyze your investments in stocks, ETFs, dividend stocks, and cryptocurrencies")

# Create tabs for different sections
tabs = st.tabs(["Stocks", "Dividends", "ETFs", "Crypto", "Portfolio"])

# Sidebar for global filters and settings
with st.sidebar:
    st.header("Settings & Filters")
    
    # Date range selector for historical data
    st.subheader("Historical Data Range")
    start_date = st.date_input("Start Date", 
                              value=datetime.now() - timedelta(days=365), 
                              max_value=datetime.now() - timedelta(days=1))
    end_date = st.date_input("End Date", 
                            value=datetime.now(), 
                            max_value=datetime.now())
                            
    # Convert date inputs to datetime objects without timezone info
    start_date = pd.Timestamp(start_date).to_pydatetime()
    end_date = pd.Timestamp(end_date).to_pydatetime()
    
    # Forecast settings
    st.subheader("Forecast Settings")
    forecast_days = st.slider("Forecast Days", 1, 90, 30)
    forecast_method = st.selectbox("Forecast Method", 
                                 ["ARIMA", "Prophet", "Linear Regression", "LSTM"])

    # Apply button for refreshing data
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

# Stocks Tab
with tabs[0]:
    st.header("Stock Analysis")
    
    # User input for stock ticker
    stock_ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT)", value="AAPL")
    
    if stock_ticker:
        # Get and analyze stock data
        try:
            stock_data = get_stock_data(stock_ticker, start_date, end_date)
            
            # Display basic stock information
            stock_info_col1, stock_info_col2 = st.columns(2)
            
            with stock_info_col1:
                current_price = stock_data['Close'].iloc[-1]
                previous_close = stock_data['Close'].iloc[-2]
                price_change = current_price - previous_close
                price_change_pct = (price_change / previous_close) * 100
                
                st.metric(
                    label=f"{stock_ticker} Price", 
                    value=f"${current_price:.2f}", 
                    delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
                )
                
                # Display metrics
                metrics = analyze_stock(stock_data)
                for metric_name, metric_value in metrics.items():
                    st.metric(label=metric_name, value=metric_value)
            
            with stock_info_col2:
                # Plot historical prices
                st.subheader("Historical Price")
                st.line_chart(stock_data['Close'])
            
            # Show forecast
            st.subheader(f"{forecast_days}-Day Price Forecast")
            forecast_data = forecast_stock(stock_data, forecast_days, method=forecast_method)
            st.line_chart(forecast_data['Forecast'])
            
            # Technical Analysis
            st.subheader("Technical Analysis")
            with st.expander("View Technical Indicators"):
                # Display moving averages
                st.write("Moving Averages")
                st.line_chart({
                    'Close': stock_data['Close'],
                    'MA50': stock_data['Close'].rolling(window=50).mean(),
                    'MA200': stock_data['Close'].rolling(window=200).mean(),
                })
                
                # Additional technical indicators would be added here
        
        except Exception as e:
            st.error(f"Error retrieving data for {stock_ticker}: {str(e)}")

# Dividends Tab
with tabs[1]:
    st.header("Dividend Stock Analysis")
    
    # Create two columns - one for user input and one for pre-defined list
    div_col1, div_col2 = st.columns(2)
    
    with div_col1:
        st.subheader("Analyze Your Dividend Stock")
        div_ticker = st.text_input("Enter Dividend Stock Ticker Symbol (e.g., KO, JNJ)", value="KO")
        
        if div_ticker:
            try:
                # Convert date inputs to pandas Timestamp for yfinance compatibility
                pd_start_date = pd.Timestamp(start_date)
                pd_end_date = pd.Timestamp(end_date)
                
                # Get dividend data and analyze
                div_data = get_dividend_data(div_ticker, pd_start_date, pd_end_date)
                
                # Display basic dividend information
                current_price = div_data['price']
                div_yield = div_data['dividend_yield']
                payout_ratio = div_data['payout_ratio']
                
                st.metric(label=f"{div_ticker} Price", value=f"${current_price:.2f}")
                st.metric(label="Dividend Yield", value=f"{div_yield:.2f}%")
                st.metric(label="Payout Ratio", value=f"{payout_ratio:.2f}%")
                
                # Dividend analysis
                analysis_results = analyze_dividend(div_data)
                
                # Dividend rating
                rating, rating_details = rate_dividend(div_data)
                
                # Display rating with explanation
                st.subheader("Dividend Rating")
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Rating: {rating}/5</h3>
                    <p>{rating_details['summary']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Dividend growth history
                st.subheader("Dividend Growth History")
                st.line_chart(div_data['dividend_history'])
                
                # Show detailed rating breakdown
                with st.expander("View Rating Details"):
                    for category, score in rating_details['categories'].items():
                        st.markdown(f"**{category}**: {score}/5")
                        st.write(rating_details['explanations'][category])
            
            except Exception as e:
                st.error(f"Error retrieving dividend data for {div_ticker}: {str(e)}")
    
    with div_col2:
        st.subheader("Our Dividend Picks")
        
        # Load the portfolio data focusing on dividend stocks
        portfolio_data = load_portfolio_data()
        div_stocks = portfolio_data[portfolio_data['Amount Per Share'] > 0].sort_values(by='Dist. Yield', ascending=False)
        
        # Display the dividend stocks
        for _, row in div_stocks.iterrows():
            symbol = row['Symbol']
            desc = row['Description']
            price = row['Last Price']
            yield_val = row['Dist. Yield']
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{symbol} - {desc}</h3>
                <p>Price: ${price:.2f} | Yield: {yield_val:.2f}%</p>
                <p>Ex-Date: {row['Ex-Date']} | Pay Date: {row['Pay-Date']}</p>
                <p>Annual Income: ${row['Est. Annual Income']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

# ETFs Tab
with tabs[2]:
    st.header("ETF Analysis")
    
    # Create two columns - one for user input and one for pre-defined list
    etf_col1, etf_col2 = st.columns(2)
    
    with etf_col1:
        st.subheader("Analyze Your ETF")
        etf_ticker = st.text_input("Enter ETF Ticker Symbol (e.g., VOO, SPY)", value="VOO")
        
        if etf_ticker:
            try:
                # Convert date inputs to pandas Timestamp for yfinance compatibility
                pd_start_date = pd.Timestamp(start_date)
                pd_end_date = pd.Timestamp(end_date)
                
                # Get ETF data and analyze
                etf_data = get_etf_data(etf_ticker, pd_start_date, pd_end_date)
                
                # Display basic ETF information
                current_price = etf_data['Close'].iloc[-1]
                previous_close = etf_data['Close'].iloc[-2]
                price_change = current_price - previous_close
                price_change_pct = (price_change / previous_close) * 100
                
                st.metric(
                    label=f"{etf_ticker} Price", 
                    value=f"${current_price:.2f}", 
                    delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
                )
                
                # ETF performance
                st.subheader("Performance")
                performance = analyze_etf(etf_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("YTD Return", f"{performance['ytd_return']:.2f}%")
                    st.metric("1 Year Return", f"{performance['1y_return']:.2f}%")
                
                with col2:
                    st.metric("3 Year Return", f"{performance['3y_return']:.2f}%")
                    st.metric("5 Year Return", f"{performance['5y_return']:.2f}%")
                
                # Display price chart
                st.subheader("Price History")
                st.line_chart(etf_data['Close'])
                
                # Show holdings breakdown if available
                if 'holdings' in etf_data:
                    st.subheader("Top Holdings")
                    st.dataframe(etf_data['holdings'])
            
            except Exception as e:
                st.error(f"Error retrieving ETF data for {etf_ticker}: {str(e)}")
    
    with etf_col2:
        st.subheader("Our ETF Picks")
        
        # Load the portfolio data focusing on ETFs
        portfolio_data = load_portfolio_data()
        etf_data = portfolio_data[portfolio_data['Symbol'].isin(['VOO', 'SDIV', 'JEPI'])]
        
        # Display the ETFs
        for _, row in etf_data.iterrows():
            symbol = row['Symbol']
            desc = row['Description']
            price = row['Last Price']
            value = row['Current Value']
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{symbol} - {desc}</h3>
                <p>Price: ${price:.2f} | Value: ${value:.2f}</p>
                <p>Yield: {row.get('Dist. Yield', 'N/A')}%</p>
                <p>Annual Income: ${row.get('Est. Annual Income', 0):.2f}</p>
            </div>
            """, unsafe_allow_html=True)

# Crypto Tab
with tabs[3]:
    st.header("Cryptocurrency Analysis")
    
    # User input for crypto ticker
    crypto_ticker = st.text_input("Enter Cryptocurrency Symbol (e.g., BTC, ETH)", value="BTC")
    
    if crypto_ticker:
        try:
            # Convert date inputs to pandas Timestamp for yfinance compatibility
            pd_start_date = pd.Timestamp(start_date)
            pd_end_date = pd.Timestamp(end_date)
            
            # Get and analyze crypto data
            crypto_data = get_crypto_data(crypto_ticker, pd_start_date, pd_end_date)
            
            # Display basic crypto information
            crypto_info_col1, crypto_info_col2 = st.columns(2)
            
            with crypto_info_col1:
                current_price = crypto_data['Close'].iloc[-1]
                previous_close = crypto_data['Close'].iloc[-2]
                price_change = current_price - previous_close
                price_change_pct = (price_change / previous_close) * 100
                
                st.metric(
                    label=f"{crypto_ticker} Price", 
                    value=f"${current_price:.2f}", 
                    delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
                )
                
                # Display metrics
                metrics = analyze_crypto(crypto_data)
                for metric_name, metric_value in metrics.items():
                    st.metric(label=metric_name, value=metric_value)
            
            with crypto_info_col2:
                # Plot historical prices
                st.subheader("Historical Price")
                st.line_chart(crypto_data['Close'])
            
            # Show forecast
            st.subheader(f"{forecast_days}-Day Price Forecast")
            forecast_data = forecast_crypto(crypto_data, forecast_days, method=forecast_method)
            st.line_chart(forecast_data['Forecast'])
            
            # Market sentiment analysis
            st.subheader("Market Sentiment")
            sentiment_score = metrics.get('Sentiment Score', 'Neutral')
            st.write(f"Current Market Sentiment: {sentiment_score}")
            
            # Volume analysis
            st.subheader("Volume Analysis")
            st.bar_chart(crypto_data['Volume'])
        
        except Exception as e:
            st.error(f"Error retrieving cryptocurrency data for {crypto_ticker}: {str(e)}")

# Portfolio Tab
with tabs[4]:
    st.header("Portfolio Overview")
    
    # Load portfolio data
    portfolio_data = load_portfolio_data()
    
    # Calculate portfolio metrics
    portfolio_metrics = calculate_portfolio_metrics(portfolio_data)
    
    # Display portfolio summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Value", f"${portfolio_metrics['total_value']:.2f}")
    
    with col2:
        st.metric("Annual Income", f"${portfolio_metrics['annual_income']:.2f}")
    
    with col3:
        st.metric("Avg Dividend Yield", f"{portfolio_metrics['avg_yield']:.2f}%")
    
    # Portfolio allocation pie chart
    st.subheader("Portfolio Allocation")
    allocation_df = portfolio_data.groupby('Symbol')['Current Value'].sum().reset_index()
    allocation_df = allocation_df.sort_values('Current Value', ascending=False)
    
    # Using Plotly for interactive pie chart
    st.write("Asset Allocation by Value")
    st.bar_chart(allocation_df.set_index('Symbol'))
    
    # Display full portfolio table
    st.subheader("Complete Portfolio")
    st.dataframe(portfolio_data)
    
    # Monthly income projection
    st.subheader("Monthly Income Projection")
    
    try:
        # Create a dataframe with monthly income projections
        monthly_income = pd.DataFrame({
            'Month': pd.date_range(start=datetime.now(), periods=12, freq='M').strftime('%b %Y'),
            'Projected Income': [portfolio_metrics['monthly_income']] * 12
        })
        
        # Display monthly income as a bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_income['Month'],
            y=monthly_income['Projected Income'],
            marker_color='green'
        ))
        
        fig.update_layout(
            title="Monthly Income Projection",
            xaxis_title="Month",
            yaxis_title="Income ($)",
            yaxis_tickprefix="$"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display annual summary
        total_annual = monthly_income['Projected Income'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Projected Annual Income", f"${total_annual:,.2f}")
        
        with col2:
            st.metric("Average Monthly Income", f"${portfolio_metrics['monthly_income']:,.2f}")
    
    except Exception as e:
        st.error(f"Error displaying income projection: {str(e)}")
        
        # Simple fallback display
        st.write(f"Estimated Monthly Income: ${portfolio_metrics['monthly_income']:,.2f}")
        st.write(f"Estimated Annual Income: ${portfolio_metrics['annual_income']:,.2f}")
    
    # Display upcoming dividends
    st.subheader("Upcoming Dividend Payments")
    try:
        # Make a copy of the portfolio data for manipulation
        upcoming_div = portfolio_data[portfolio_data['Amount Per Share'] > 0].copy()
        
        # Debug information
        st.write(f"Total dividend-paying positions: {len(upcoming_div)}")
        
        # Convert 'Pay-Date' to datetime, handling potential errors
        upcoming_div['Pay Date'] = pd.to_datetime(upcoming_div['Pay-Date'], errors='coerce')
        
        # Filter for upcoming payments and handle NaT values
        current_date = pd.Timestamp(datetime.now().date())
        upcoming_div = upcoming_div[upcoming_div['Pay Date'].notna()]
        upcoming_div = upcoming_div[upcoming_div['Pay Date'] >= current_date]
        upcoming_div = upcoming_div.sort_values('Pay Date')
        
        # Debug information
        st.write(f"Upcoming dividend payments: {len(upcoming_div)}")
        
        # Display upcoming dividends
        if not upcoming_div.empty:
            for _, row in upcoming_div.iterrows():
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{row['Symbol']} - {row['Description']}</h3>
                    <p>Pay Date: {row['Pay Date'].strftime('%b %d, %Y')}</p>
                    <p>Amount: ${row['Amount Per Share']:.2f} per share</p>
                    <p>Total Payment: ${row['Amount Per Share'] * row['Quantity']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            # If no upcoming dividends, show all dividend payments for reference
            st.write("No upcoming dividend payments found. Here are all dividend-paying positions:")
            
            all_div = portfolio_data[portfolio_data['Amount Per Share'] > 0].copy()
            
            for _, row in all_div.iterrows():
                pay_date = row['Pay-Date'] if isinstance(row['Pay-Date'], str) else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{row['Symbol']} - {row['Description']}</h3>
                    <p>Pay Date: {pay_date}</p>
                    <p>Amount: ${row['Amount Per Share']:.2f} per share</p>
                    <p>Annual Income: ${row['Est. Annual Income']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying upcoming dividends: {str(e)}")
        st.write("Here are all dividend stocks in your portfolio:")
        
        # Display all dividend stocks as a fallback
        div_stocks = portfolio_data[portfolio_data['Amount Per Share'] > 0]
        st.dataframe(div_stocks[['Symbol', 'Description', 'Quantity', 'Amount Per Share', 'Est. Annual Income']])

# Footer
st.markdown("""
---
### About This Dashboard
This financial analysis dashboard allows you to track stocks, ETFs, dividend stocks, and cryptocurrencies.
For dividend stocks, it provides a rating system based on yield, payout ratio, growth history, and more.
You can forecast prices up to 90 days ahead and analyze your portfolio performance.

Developed with ❤️ AvaResearchLLC
""")