# stock_analysis.py
# This file contains functions for retrieving and analyzing stock data
# It also includes forecasting functionality for stock prices

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

def get_stock_data(ticker, start_date, end_date):
    """
    Retrieve historical stock data from Yahoo Finance
    
    Parameters:
    - ticker (str): The stock ticker symbol (e.g., 'AAPL')
    - start_date (datetime): Start date for historical data
    - end_date (datetime): End date for historical data
    
    Returns:
    - pandas.DataFrame: DataFrame containing historical price and volume data
    """
    # Convert dates to string format required by yfinance
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Download data from Yahoo Finance
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_str, end=end_str)
    
    # If data is empty, raise an exception
    if data.empty:
        raise Exception(f"No data found for ticker {ticker}")
    
    # Add stock info to the data
    info = stock.info
    data.attrs['info'] = {
        'name': info.get('shortName', ticker),
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown'),
        'market_cap': info.get('marketCap', 0),
        'pe_ratio': info.get('trailingPE', None),
        'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
        'beta': info.get('beta', None)
    }
    
    return data

def analyze_stock(stock_data):
    """
    Perform basic analysis on stock data
    
    Parameters:
    - stock_data (pandas.DataFrame): DataFrame containing historical stock data
    
    Returns:
    - dict: Dictionary containing various metrics and analysis results
    """
    # Extract the last year of data for calculating returns
    today = datetime.now()
    one_year_ago = today - timedelta(days=365)
    
    # Calculate returns over different time periods
    current_price = stock_data['Close'].iloc[-1]
    
    # Calculate volatility (standard deviation of returns)
    daily_returns = stock_data['Close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Calculate Sharpe ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02
    sharpe_ratio = (daily_returns.mean() * 252 - risk_free_rate) / volatility
    
    # Calculate different time period returns
    # YTD return
    ytd_start = datetime(today.year, 1, 1)
    ytd_data = stock_data.loc[stock_data.index >= ytd_start]
    if not ytd_data.empty:
        ytd_return = (current_price / ytd_data['Close'].iloc[0] - 1) * 100
    else:
        ytd_return = 0
    
    # 1-month return
    one_month_ago = today - timedelta(days=30)
    month_data = stock_data.loc[stock_data.index >= one_month_ago]
    if not month_data.empty:
        month_return = (current_price / month_data['Close'].iloc[0] - 1) * 100
    else:
        month_return = 0
    
    # 3-month return
    three_months_ago = today - timedelta(days=90)
    three_month_data = stock_data.loc[stock_data.index >= three_months_ago]
    if not three_month_data.empty:
        three_month_return = (current_price / three_month_data['Close'].iloc[0] - 1) * 100
    else:
        three_month_return = 0
        
    # 1-year return
    year_data = stock_data.loc[stock_data.index >= one_year_ago]
    if not year_data.empty:
        year_return = (current_price / year_data['Close'].iloc[0] - 1) * 100
    else:
        year_return = 0
    
    # Get stock info if available
    info = stock_data.attrs.get('info', {})
    
    # Return metrics as a dictionary
    return {
        'Name': info.get('name', 'Unknown'),
        'Sector': info.get('sector', 'Unknown'),
        'Industry': info.get('industry', 'Unknown'),
        'Market Cap': f"${info.get('market_cap', 0) / 1e9:.2f}B",
        'P/E Ratio': f"{info.get('pe_ratio', 'N/A')}",
        'Dividend Yield': f"{info.get('dividend_yield', 0):.2f}%",
        'Beta': f"{info.get('beta', 'N/A')}",
        '1M Return': f"{month_return:.2f}%",
        '3M Return': f"{three_month_return:.2f}%",
        '1Y Return': f"{year_return:.2f}%",
        'YTD Return': f"{ytd_return:.2f}%",
        'Volatility (Annual)': f"{volatility * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}"
    }

def forecast_stock(stock_data, days=30, method='ARIMA'):
    """
    Forecast stock prices for a specified number of days ahead
    
    Parameters:
    - stock_data (pandas.DataFrame): DataFrame containing historical stock data
    - days (int): Number of days to forecast
    - method (str): Forecasting method to use ('ARIMA', 'Prophet', 'Linear Regression', or 'LSTM')
    
    Returns:
    - pandas.DataFrame: DataFrame containing the forecasted prices
    """
    # Extract the closing prices
    close_prices = stock_data['Close']
    
    # Create a date range for the forecast period
    last_date = stock_data.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    if method == 'ARIMA':
        # Fit ARIMA model
        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()
        
        # Make forecast
        forecast_result = model_fit.forecast(steps=days)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_result
        }).set_index('Date')
        
    elif method == 'Prophet':
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': close_prices.index,
            'y': close_prices.values
        })
        
        # Fit Prophet model
        model = Prophet()
        model.fit(prophet_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days)
        
        # Make forecast
        forecast = model.predict(future)
        
        # Filter to only the forecast period
        forecast_result = forecast.iloc[-days:]['yhat']
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_result.values
        }).set_index('Date')
        
    elif method == 'Linear Regression':
        # Simple linear regression forecast
        # Create a sequence of numbers from 0 to the length of the data
        X = np.array(range(len(close_prices))).reshape(-1, 1)
        y = close_prices.values
        
        # Fit linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future values
        future_X = np.array(range(len(close_prices), len(close_prices) + days)).reshape(-1, 1)
        forecast_result = model.predict(future_X)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_result
        }).set_index('Date')
        
    elif method == 'LSTM':
        # LSTM requires more complex implementation
        # For simplicity, we'll use a placeholder implementation
        # In a real application, you would implement a proper LSTM model
        
        # Simple extrapolation based on recent trend
        last_price = close_prices.iloc[-1]
        avg_change = close_prices.pct_change().dropna().mean()
        
        forecast_result = [last_price]
        for _ in range(days - 1):
            next_price = forecast_result[-1] * (1 + avg_change)
            forecast_result.append(next_price)
        
        forecast_result = forecast_result[1:]  # Remove the first value (which is the last actual price)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_result
        }).set_index('Date')
    
    else:
        raise ValueError(f"Unknown forecasting method: {method}")
    
    # Combine historical data with forecast
    historical_df = pd.DataFrame({
        'Close': close_prices,
        'Forecast': np.nan
    })
    
    forecast_df = pd.DataFrame({
        'Close': np.nan,
        'Forecast': forecast_df['Forecast']
    }, index=forecast_df.index)
    
    combined_df = pd.concat([historical_df, forecast_df])
    
    return combined_df