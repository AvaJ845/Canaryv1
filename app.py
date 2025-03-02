# simple_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

st.title("Simplified Stock Dashboard")

# Default tickers
tickers = ['AAPL', 'MSFT', 'GOOGL']

# Fetch some data
@st.cache_data(ttl=300)
def fetch_data(ticker_list):
    data = {}
    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            if not hist.empty:
                data[ticker] = hist
        except Exception as e:
            st.warning(f"Error fetching {ticker}: {e}")
    return data

# Get data
with st.spinner("Fetching data..."):
    stock_data = fetch_data(tickers)

# Display simple metrics
for ticker, data in stock_data.items():
    st.subheader(ticker)
    if not data.empty:
        st.metric(
            label="Current Price",
            value=f"${data['Close'].iloc[-1]:.2f}",
            delta=f"{data['Close'].pct_change().iloc[-1]*100:.2f}%"
        )
        
        # Simple chart
        fig = px.line(data, y='Close', title=f"{ticker} Price")
        st.plotly_chart(fig)