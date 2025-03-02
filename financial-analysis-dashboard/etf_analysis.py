# etf_analysis.py
# This file contains functions for retrieving and analyzing ETF data

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def get_etf_data(ticker, start_date, end_date):
    """
    Retrieve historical ETF data from Yahoo Finance
    
    Parameters:
    - ticker (str): The ETF ticker symbol (e.g., 'VOO', 'SPY')
    - start_date (datetime): Start date for historical data
    - end_date (datetime): End date for historical data
    
    Returns:
    - pandas.DataFrame: DataFrame containing historical price and volume data
    """
    # Convert dates to string format required by yfinance
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Download data from Yahoo Finance
    etf = yf.Ticker(ticker)
    data = etf.history(start=start_str, end=end_str)
    
    # If data is empty, raise an exception
    if data.empty:
        raise Exception(f"No data found for ETF ticker {ticker}")
    
    # Get ETF info
    try:
        info = etf.info
        data.attrs['info'] = {
            'name': info.get('shortName', ticker),
            'category': info.get('category', 'Unknown'),
            'asset_class': info.get('assetClass', 'Unknown'),
            'net_assets': info.get('totalAssets', 0),
            'expense_ratio': info.get('annualReportExpenseRatio', 0) * 100 if info.get('annualReportExpenseRatio') else 0,
            'dividend_yield': info.get('yield', 0) * 100 if info.get('yield') else 0,
            'beta': info.get('beta', None)
        }
        
        # Try to get holdings information if available
        try:
            holdings = etf.get_holdings()
            if not holdings.empty:
                data.attrs['holdings'] = holdings
        except:
            pass
    except:
        # If info can't be retrieved, set default values
        data.attrs['info'] = {
            'name': ticker,
            'category': 'Unknown',
            'asset_class': 'Unknown',
            'net_assets': 0,
            'expense_ratio': 0,
            'dividend_yield': 0,
            'beta': None
        }
    
    return data

def analyze_etf(etf_data):
    """
    Perform analysis on ETF data
    
    Parameters:
    - etf_data (pandas.DataFrame): DataFrame containing historical ETF data
    
    Returns:
    - dict: Dictionary containing various metrics and analysis results
    """
    # Get current date and relevant past dates
    today = datetime.now()
    ytd_start = datetime(today.year, 1, 1)
    one_year_ago = today - timedelta(days=365)
    three_years_ago = today - timedelta(days=3 * 365)
    five_years_ago = today - timedelta(days=5 * 365)
    
    # Get closing prices
    close_prices = etf_data['Close']
    current_price = close_prices.iloc[-1]
    
    # Calculate returns for different periods
    # YTD return
    ytd_data = etf_data.loc[etf_data.index >= ytd_start]
    if not ytd_data.empty:
        ytd_return = (current_price / ytd_data['Close'].iloc[0] - 1) * 100
    else:
        ytd_return = 0
    
    # 1-year return
    year_data = etf_data.loc[etf_data.index >= one_year_ago]
    if len(year_data) > 0:
        year_return = (current_price / year_data['Close'].iloc[0] - 1) * 100
    else:
        year_return = 0
    
    # 3-year return (annualized)
    three_year_data = etf_data.loc[etf_data.index >= three_years_ago]
    if len(three_year_data) > 0:
        three_year_total_return = (current_price / three_year_data['Close'].iloc[0] - 1)
        three_year_return = ((1 + three_year_total_return) ** (1/3) - 1) * 100
    else:
        three_year_return = 0
    
    # 5-year return (annualized)
    five_year_data = etf_data.loc[etf_data.index >= five_years_ago]
    if len(five_year_data) > 0:
        five_year_total_return = (current_price / five_year_data['Close'].iloc[0] - 1)
        five_year_return = ((1 + five_year_total_return) ** (1/5) - 1) * 100
    else:
        five_year_return = 0
    
    # Calculate volatility (standard deviation of returns)
    daily_returns = etf_data['Close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Calculate Sharpe ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02
    sharpe_ratio = (daily_returns.mean() * 252 - risk_free_rate) / volatility
    
    # Get ETF info if available
    info = etf_data.attrs.get('info', {})
    
    # Calculate risk metrics
    max_drawdown = calculate_max_drawdown(etf_data['Close'])
    
    # Return metrics as a dictionary
    return {
        'name': info.get('name', 'Unknown'),
        'category': info.get('category', 'Unknown'),
        'asset_class': info.get('asset_class', 'Unknown'),
        'net_assets': info.get('net_assets', 0) / 1e9 if info.get('net_assets', 0) > 0 else 0,  # Convert to billions
        'expense_ratio': info.get('expense_ratio', 0),
        'dividend_yield': info.get('dividend_yield', 0),
        'beta': info.get('beta', None),
        'ytd_return': ytd_return,
        '1y_return': year_return,
        '3y_return': three_year_return,
        '5y_return': five_year_return,
        'volatility': volatility * 100,  # Convert to percentage
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100  # Convert to percentage
    }

def calculate_max_drawdown(prices):
    """
    Calculate the maximum drawdown from a series of prices
    
    Parameters:
    - prices (pandas.Series): Series of prices
    
    Returns:
    - float: Maximum drawdown as a decimal (e.g., 0.25 for 25%)
    """
    # Calculate running maximum
    running_max = prices.cummax()
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    # Calculate maximum drawdown
    max_drawdown = drawdown.min()
    
    return max_drawdown

def compare_etfs(tickers, start_date, end_date):
    """
    Compare multiple ETFs based on key metrics
    
    Parameters:
    - tickers (list): List of ETF ticker symbols
    - start_date (datetime): Start date for analysis
    - end_date (datetime): End date for analysis
    
    Returns:
    - pandas.DataFrame: DataFrame with comparison metrics for each ETF
    """
    results = []
    
    for ticker in tickers:
        try:
            # Get ETF data
            etf_data = get_etf_data(ticker, start_date, end_date)
            
            # Analyze ETF
            analysis = analyze_etf(etf_data)
            
            # Add to results
            results.append({
                'Ticker': ticker,
                'Name': analysis['name'],
                'Category': analysis['category'],
                'Expense Ratio': analysis['expense_ratio'],
                'Dividend Yield': analysis['dividend_yield'],
                'YTD Return': analysis['ytd_return'],
                '1Y Return': analysis['1y_return'],
                '3Y Return': analysis['3y_return'],
                '5Y Return': analysis['5y_return'],
                'Volatility': analysis['volatility'],
                'Sharpe Ratio': analysis['sharpe_ratio'],
                'Max Drawdown': analysis['max_drawdown']
            })
        except Exception as e:
            print(f"Error analyzing ETF {ticker}: {str(e)}")
    
    return pd.DataFrame(results)

def get_etf_sector_exposure(ticker):
    """
    Get sector exposure for an ETF
    
    Parameters:
    - ticker (str): ETF ticker symbol
    
    Returns:
    - dict: Dictionary with sector names as keys and percentages as values
    """
    try:
        etf = yf.Ticker(ticker)
        
        # Try to get holdings if method is available
        if hasattr(etf, 'get_sector_holdings'):
            sectors = etf.get_sector_holdings()
            return sectors.to_dict()
        
        # Alternative approach if sector holdings not directly available
        holdings = etf.get_holdings() if hasattr(etf, 'get_holdings') else pd.DataFrame()
        
        if not holdings.empty and 'sector' in holdings.columns:
            # Group by sector and sum percentages
            sector_exposure = holdings.groupby('sector')['% of portfolio'].sum()
            return sector_exposure.to_dict()
        
        return {}
    except:
        return {}