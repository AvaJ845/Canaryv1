"""
Components for rendering the main dashboard with tabs.
"""

import streamlit as st
import pandas as pd
from components.ticker_analysis import render_ticker_analysis
import config

def render_dashboard(stock_data, metrics, notable_events, stocks_list, etfs_list, dividend_stocks_list):
    """
    Render the main dashboard with tabs for stocks, ETFs, and dividend stocks.
    
    Args:
        stock_data (dict): Dictionary of ticker symbols and their historical data
        metrics (dict): Dictionary of ticker symbols and their metrics
        notable_events (list): List of notable events
        stocks_list (list): List of stock tickers
        etfs_list (list): List of ETF tickers
        dividend_stocks_list (list): List of dividend stock tickers
    """
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
        render_ticker_tab("Stocks", stock_metrics, stock_events, False)
    
    # === ETFs TAB ===
    with etf_tab:
        render_ticker_tab("ETFs", etf_metrics, etf_events, False)
    
    # === DIVIDEND STOCKS TAB ===
    with dividend_tab:
        render_ticker_tab("Dividend Stocks", dividend_metrics, dividend_events, True)

def render_ticker_tab(tab_title, ticker_metrics, ticker_events, show_dividend):
    """
    Render a tab for a specific type of ticker (stocks, ETFs, or dividend stocks).
    
    Args:
        tab_title (str): Title of the tab
        ticker_metrics (dict): Dictionary of ticker symbols and their metrics
        ticker_events (list): List of notable events for these tickers
        show_dividend (bool): Whether to show dividend information
    """
    # Overview metrics
    st.header(f"{tab_title} Overview")
    
    # Display metrics in columns (limit to 5 per row to avoid squeezing)
    if ticker_metrics:
        for i in range(0, len(ticker_metrics), 5):
            cols = st.columns(min(5, len(ticker_metrics) - i))
            for j, ticker in enumerate(list(ticker_metrics.keys())[i:i+5]):
                metric = ticker_metrics[ticker]
                with cols[j]:
                    st.metric(
                        label=ticker,
                        value=f"${metric['current_price']:.2f}",
                        delta=f"{metric['daily_change']:.2f}%" if metric['daily_change'] else "N/A"
                    )
    
        # Notable events section
        st.header(f"Notable {tab_title} Events")
        
        if ticker_events:
            events_df = pd.DataFrame(ticker_events)
            
            # Format the dataframe for display
            events_df = events_df[['ticker', 'event', 'value', 'importance']]
            events_df.columns = ['Ticker', 'Event', 'Value', 'Significance']
            
            # Apply styling
            st.dataframe(events_df.style.background_gradient(subset=['Significance'], cmap='YlOrRd'))
        else:
            st.info(f"No notable {tab_title.lower()} events detected based on current significance threshold.")
        
        # Individual ticker analysis
        st.header(f"Individual {tab_title} Analysis")
        
        # Create tabs for each ticker
        ticker_tabs = st.tabs(list(ticker_metrics.keys()))
        
        for i, (ticker, ticker_tab) in enumerate(zip(ticker_metrics.keys(), ticker_tabs)):
            render_ticker_analysis(ticker, ticker_metrics[ticker], ticker_tab, show_dividend)
    else:
        st.info(f"No {tab_title.lower()} data available. Please add {tab_title.lower()} in the sidebar.")
