#metrics_utils.py
"""
Utilities for calculating stock metrics and identifying notable events.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(stock_data, dividend_stocks_list=None):
    """
    Calculate key metrics and statistics for the given stock data.
    
    Args:
        stock_data (dict): Dictionary of ticker symbols and their historical data
        dividend_stocks_list (list, optional): List of dividend stock tickers
        
    Returns:
        dict: Dictionary of ticker symbols and their metrics
    """
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
        
        if dividend_stocks_list and ticker in dividend_stocks_list:
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

def identify_notable_events(metrics, threshold):
    """
    Identify notable events based on the calculated metrics.
    
    Args:
        metrics (dict): Dictionary of ticker symbols and their metrics
        threshold (float): Significance threshold for identifying events
        
    Returns:
        list: List of notable events
    """
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
