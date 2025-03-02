# dividend_analysis.py
# This file contains functions for retrieving and analyzing dividend stock data
# It includes functionality for rating dividend stocks based on various metrics

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def get_dividend_data(ticker, start_date, end_date):
    """
    Retrieve historical data and dividend information for a stock
    
    Parameters:
    - ticker (str): The stock ticker symbol (e.g., 'KO')
    - start_date (datetime): Start date for historical data
    - end_date (datetime): End date for historical data
    
    Returns:
    - dict: Dictionary containing stock data and dividend information
    """
    # Convert dates to string format required by yfinance
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Download data from Yahoo Finance
    stock = yf.Ticker(ticker)
    price_data = stock.history(start=start_str, end=end_str)
    
    # If data is empty, raise an exception
    if price_data.empty:
        raise Exception(f"No data found for ticker {ticker}")
    
    # Get dividend data
    try:
        # Get dividends directly from the history
        dividends = stock.dividends
        
        # If no dividends found in the period, try to get annual dividends
        if dividends.empty:
            info = stock.info
            annual_dividend = info.get('dividendRate', 0)
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            payout_ratio = info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0
            
            # Create a dataframe with estimated dividend dates (quarterly)
            current_year = datetime.now().year
            est_dates = pd.date_range(start=f"{current_year}-01-01", end=f"{current_year}-12-31", freq='Q')
            est_dividends = pd.Series([annual_dividend / 4] * len(est_dates), index=est_dates)
            
            dividend_history = est_dividends
        else:
            # Filter dividends to the requested date range
            dividend_history = dividends
            
            # Calculate dividend metrics from historical data
            annual_dividend = dividend_history.iloc[-4:].sum() if len(dividend_history) >= 4 else dividend_history.mean() * 4
            current_price = price_data['Close'].iloc[-1]
            dividend_yield = (annual_dividend / current_price) * 100
            
            # For payout ratio, we need earnings data
            info = stock.info
            payout_ratio = info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0
        
        # Calculate dividend growth rates
        if len(dividend_history) >= 8:  # Need at least 2 years of dividend data
            # Group by year and sum to get annual dividends
            annual_dividends = dividend_history.groupby(dividend_history.index.year).sum()
            
            # Calculate year-over-year growth rates
            yoy_growth = annual_dividends.pct_change() * 100
            
            # Calculate average growth rate over the last 5 years
            five_year_growth = yoy_growth.tail(5).mean() if len(yoy_growth) >= 5 else yoy_growth.mean()
        else:
            yoy_growth = pd.Series([0])
            five_year_growth = 0
        
        # Get additional info
        info = stock.info
        beta = info.get('beta', 1)
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # Create a results dictionary
        dividend_data = {
            'price': price_data['Close'].iloc[-1],
            'price_history': price_data['Close'],
            'annual_dividend': annual_dividend,
            'dividend_yield': dividend_yield,
            'payout_ratio': payout_ratio,
            'five_year_growth': five_year_growth,
            'yoy_growth': yoy_growth,
            'dividend_history': dividend_history,
            'beta': beta,
            'sector': sector,
            'industry': industry,
            'name': info.get('shortName', ticker),
            'market_cap': info.get('marketCap', 0),
            'years_paying_dividends': len(dividend_history.groupby(dividend_history.index.year).sum())
        }
        
        return dividend_data
        
    except Exception as e:
        # If dividend data can't be retrieved, return basic data
        return {
            'price': price_data['Close'].iloc[-1],
            'price_history': price_data['Close'],
            'annual_dividend': 0,
            'dividend_yield': 0,
            'payout_ratio': 0,
            'five_year_growth': 0,
            'yoy_growth': pd.Series([0]),
            'dividend_history': pd.Series([], dtype=float),
            'beta': 1,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'name': ticker,
            'market_cap': 0,
            'years_paying_dividends': 0
        }

def analyze_dividend(dividend_data):
    """
    Analyze dividend stock data
    
    Parameters:
    - dividend_data (dict): Dictionary containing dividend information
    
    Returns:
    - dict: Dictionary containing analysis results
    """
    # Extract key metrics
    dividend_yield = dividend_data['dividend_yield']
    payout_ratio = dividend_data['payout_ratio']
    five_year_growth = dividend_data['five_year_growth']
    years_paying_dividends = dividend_data['years_paying_dividends']
    beta = dividend_data['beta']
    
    # Calculate Dividend Growth Model (Gordon Growth Model) fair value
    # Required rate of return (using a simplistic approach)
    risk_free_rate = 0.04  # Approximate 10-year Treasury yield
    market_risk_premium = 0.06  # Approximate equity risk premium
    required_return = risk_free_rate + beta * market_risk_premium
    
    # Estimated growth rate (using five year growth or a default)
    growth_rate = five_year_growth / 100 if five_year_growth > 0 else 0.03
    
    # Cap the growth rate to be less than the required return
    growth_rate = min(growth_rate, required_return - 0.01)
    
    # Current annual dividend
    annual_dividend = dividend_data['annual_dividend']
    
    # Calculate intrinsic value using the Gordon Growth Model
    if annual_dividend > 0 and required_return > growth_rate:
        intrinsic_value = annual_dividend / (required_return - growth_rate)
    else:
        intrinsic_value = 0
    
    # Current price
    current_price = dividend_data['price']
    
    # Calculate if the stock is undervalued or overvalued
    if intrinsic_value > 0:
        value_ratio = current_price / intrinsic_value
        if value_ratio < 0.9:
            valuation = "Undervalued"
        elif value_ratio > 1.1:
            valuation = "Overvalued"
        else:
            valuation = "Fairly Valued"
    else:
        valuation = "Unable to determine"
    
    # Calculate dividend safety
    if payout_ratio < 40:
        payout_safety = "Very Safe"
    elif payout_ratio < 60:
        payout_safety = "Safe"
    elif payout_ratio < 80:
        payout_safety = "Caution"
    else:
        payout_safety = "At Risk"
    
    # Calculate yield attractiveness
    if dividend_yield > 6:
        yield_category = "Very High"
    elif dividend_yield > 4:
        yield_category = "High"
    elif dividend_yield > 2:
        yield_category = "Moderate"
    else:
        yield_category = "Low"
    
    # Calculate growth attractiveness
    if five_year_growth > 10:
        growth_category = "High"
    elif five_year_growth > 5:
        growth_category = "Moderate"
    elif five_year_growth > 0:
        growth_category = "Low"
    else:
        growth_category = "No Growth"
    
    # Calculate consistency score
    if years_paying_dividends > 25:
        consistency = "Excellent (Dividend Aristocrat)"
    elif years_paying_dividends > 10:
        consistency = "Good"
    elif years_paying_dividends > 5:
        consistency = "Moderate"
    else:
        consistency = "Limited History"
    
    # Return analysis results
    return {
        'valuation': {
            'intrinsic_value': intrinsic_value,
            'current_price': current_price,
            'valuation_assessment': valuation
        },
        'dividend_safety': payout_safety,
        'yield_category': yield_category,
        'growth_category': growth_category,
        'consistency': consistency,
        'metrics': {
            'yield': f"{dividend_yield:.2f}%",
            'payout_ratio': f"{payout_ratio:.2f}%",
            'growth_rate': f"{five_year_growth:.2f}%",
            'years_paying': years_paying_dividends,
            'beta': beta
        }
    }

def rate_dividend(dividend_data):
    """
    Rate a dividend stock on a scale of 1-5 based on various metrics
    
    Parameters:
    - dividend_data (dict): Dictionary containing dividend information
    
    Returns:
    - tuple: (overall_rating, rating_details)
    """
    # Extract key metrics
    dividend_yield = dividend_data['dividend_yield']
    payout_ratio = dividend_data['payout_ratio']
    five_year_growth = dividend_data['five_year_growth']
    years_paying_dividends = dividend_data['years_paying_dividends']
    beta = dividend_data['beta']
    
    # Calculate yield score (0-5)
    if dividend_yield >= 6:
        yield_score = 5
    elif dividend_yield >= 4:
        yield_score = 4
    elif dividend_yield >= 3:
        yield_score = 3
    elif dividend_yield >= 2:
        yield_score = 2
    elif dividend_yield > 0:
        yield_score = 1
    else:
        yield_score = 0
    
    # Calculate safety score (0-5)
    if payout_ratio < 30:
        safety_score = 5
    elif payout_ratio < 45:
        safety_score = 4
    elif payout_ratio < 60:
        safety_score = 3
    elif payout_ratio < 75:
        safety_score = 2
    elif payout_ratio < 90:
        safety_score = 1
    else:
        safety_score = 0
    
    # Calculate growth score (0-5)
    if five_year_growth >= 15:
        growth_score = 5
    elif five_year_growth >= 10:
        growth_score = 4
    elif five_year_growth >= 7:
        growth_score = 3
    elif five_year_growth >= 3:
        growth_score = 2
    elif five_year_growth > 0:
        growth_score = 1
    else:
        growth_score = 0
    
    # Calculate consistency score (0-5)
    if years_paying_dividends >= 25:
        consistency_score = 5
    elif years_paying_dividends >= 15:
        consistency_score = 4
    elif years_paying_dividends >= 10:
        consistency_score = 3
    elif years_paying_dividends >= 5:
        consistency_score = 2
    elif years_paying_dividends > 0:
        consistency_score = 1
    else:
        consistency_score = 0
    
    # Calculate volatility score (0-5)
    if beta < 0.5:
        volatility_score = 5
    elif beta < 0.8:
        volatility_score = 4
    elif beta < 1.0:
        volatility_score = 3
    elif beta < 1.2:
        volatility_score = 2
    elif beta < 1.5:
        volatility_score = 1
    else:
        volatility_score = 0
    
    # Calculate overall rating (weighted average)
    overall_rating = (
        yield_score * 0.25 +
        safety_score * 0.25 +
        growth_score * 0.2 +
        consistency_score * 0.2 +
        volatility_score * 0.1
    )
    
    # Round to nearest 0.5
    overall_rating = round(overall_rating * 2) / 2
    
    # Create explanations for each category
    explanations = {
        'Yield': _get_yield_explanation(dividend_yield, yield_score),
        'Safety': _get_safety_explanation(payout_ratio, safety_score),
        'Growth': _get_growth_explanation(five_year_growth, growth_score),
        'Consistency': _get_consistency_explanation(years_paying_dividends, consistency_score),
        'Volatility': _get_volatility_explanation(beta, volatility_score)
    }
    
    # Create a summary based on the overall rating
    if overall_rating >= 4.5:
        summary = "Excellent dividend stock with high yield, sustainable payout, strong growth history, and good stability."
    elif overall_rating >= 3.5:
        summary = "Very good dividend stock with attractive yield and solid fundamentals."
    elif overall_rating >= 2.5:
        summary = "Average dividend stock with some positive qualities but also some areas of concern."
    elif overall_rating >= 1.5:
        summary = "Below average dividend stock with several areas of concern. Consider alternatives."
    else:
        summary = "Poor dividend stock or not primarily focused on dividends. Look for better alternatives."
    
    # Return the rating and details
    return overall_rating, {
        'summary': summary,
        'categories': {
            'Yield': yield_score,
            'Safety': safety_score,
            'Growth': growth_score,
            'Consistency': consistency_score,
            'Volatility': volatility_score
        },
        'explanations': explanations
    }

def _get_yield_explanation(yield_value, score):
    """Generate explanation for the yield score"""
    if score == 5:
        return f"Very high yield of {yield_value:.2f}% is well above market average, offering substantial income."
    elif score == 4:
        return f"High yield of {yield_value:.2f}% is significantly above market average, providing strong income."
    elif score == 3:
        return f"Good yield of {yield_value:.2f}% is above market average, offering reasonable income."
    elif score == 2:
        return f"Moderate yield of {yield_value:.2f}% is around market average, providing some income."
    elif score == 1:
        return f"Low yield of {yield_value:.2f}% is below market average, offering limited income."
    else:
        return f"Very low or no yield of {yield_value:.2f}%, providing minimal income."

def _get_safety_explanation(payout_ratio, score):
    """Generate explanation for the safety score"""
    if score == 5:
        return f"Very safe payout ratio of {payout_ratio:.2f}% indicates dividends are well-covered by earnings with significant room for growth."
    elif score == 4:
        return f"Safe payout ratio of {payout_ratio:.2f}% indicates dividends are well-covered by earnings with room for growth."
    elif score == 3:
        return f"Reasonable payout ratio of {payout_ratio:.2f}% indicates dividends are adequately covered by earnings."
    elif score == 2:
        return f"Elevated payout ratio of {payout_ratio:.2f}% suggests limited room for dividend growth and potential risk if earnings decline."
    elif score == 1:
        return f"High payout ratio of {payout_ratio:.2f}% indicates potential risk to dividend sustainability if earnings decline."
    else:
        return f"Very high payout ratio of {payout_ratio:.2f}% suggests the dividend may not be sustainable."

def _get_growth_explanation(growth_rate, score):
    """Generate explanation for the growth score"""
    if score == 5:
        return f"Excellent dividend growth of {growth_rate:.2f}% over the past five years, significantly outpacing inflation."
    elif score == 4:
        return f"Strong dividend growth of {growth_rate:.2f}% over the past five years, well above inflation."
    elif score == 3:
        return f"Good dividend growth of {growth_rate:.2f}% over the past five years, outpacing inflation."
    elif score == 2:
        return f"Modest dividend growth of {growth_rate:.2f}% over the past five years, roughly keeping pace with inflation."
    elif score == 1:
        return f"Slow dividend growth of {growth_rate:.2f}% over the past five years, below inflation."
    else:
        return "No dividend growth or dividend cuts over the past five years."

def _get_consistency_explanation(years, score):
    """Generate explanation for the consistency score"""
    if score == 5:
        return f"Exceptional dividend history with {years} consecutive years of payments, qualifying as a Dividend Aristocrat."
    elif score == 4:
        return f"Strong dividend history with {years} consecutive years of payments, demonstrating commitment to shareholders."
    elif score == 3:
        return f"Solid dividend history with {years} consecutive years of payments."
    elif score == 2:
        return f"Moderate dividend history with {years} consecutive years of payments, showing some commitment but limited track record."
    elif score == 1:
        return f"Limited dividend history with {years} consecutive years of payments, insufficient to establish a strong track record."
    else:
        return "No established dividend history or recently initiated dividend."

def _get_volatility_explanation(beta, score):
    """Generate explanation for the volatility score"""
    if score == 5:
        return f"Very low volatility with a beta of {beta:.2f}, offering excellent stability during market downturns."
    elif score == 4:
        return f"Low volatility with a beta of {beta:.2f}, providing good stability during market downturns."
    elif score == 3:
        return f"Moderate volatility with a beta of {beta:.2f}, roughly matching market movements."
    elif score == 2:
        return f"Above average volatility with a beta of {beta:.2f}, exhibiting more price swings than the overall market."
    elif score == 1:
        return f"High volatility with a beta of {beta:.2f}, showing significant price swings compared to the market."
    else:
        return f"Very high volatility with a beta of {beta:.2f}, exhibiting extreme price swings compared to the market."