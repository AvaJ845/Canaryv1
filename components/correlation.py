"""
Components for correlation analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def correlation_analysis(stock_data):
    """
    Render correlation analysis for the stock data.
    
    Args:
        stock_data (dict): Dictionary of ticker symbols and their historical data
    """
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
                
            # Visualization of a few highly correlated pairs
            if len(high_pos_corr) > 0:
                st.subheader("Correlation Visualization")
                pair = high_pos_corr[0]  # Take the highest correlated pair
                ticker1, ticker2, corr_value = pair
                
                # Normalize prices for comparison
                normalized_prices = pd.DataFrame()
                normalized_prices[ticker1] = close_prices[ticker1] / close_prices[ticker1].iloc[0] * 100
                normalized_prices[ticker2] = close_prices[ticker2] / close_prices[ticker2].iloc[0] * 100
                normalized_prices.index.name = 'Date'
                normalized_prices = normalized_prices.reset_index()
                
                # Create plot
                fig = px.line(
                    normalized_prices,
                    x='Date',
                    y=[ticker1, ticker2],
                    title=f"Price Movement Comparison: {ticker1} vs {ticker2} (Correlation: {corr_value:.2f})",
                    labels={"value": "Normalized Price (%)", "variable": "Ticker"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        if high_neg_corr:
            st.subheader("Inversely Correlated Pairs")
            for i, j, corr in sorted(high_neg_corr, key=lambda x: x[2]):
                st.write(f"{i} and {j}: {corr:.2f}")
                
            # Visualization of a few inversely correlated pairs
            if len(high_neg_corr) > 0:
                st.subheader("Inverse Correlation Visualization")
                pair = high_neg_corr[0]  # Take the most inversely correlated pair
                ticker1, ticker2, corr_value = pair
                
                # Normalize prices for comparison
                normalized_prices = pd.DataFrame()
                normalized_prices[ticker1] = close_prices[ticker1] / close_prices[ticker1].iloc[0] * 100
                normalized_prices[ticker2] = close_prices[ticker2] / close_prices[ticker2].iloc[0] * 100
                normalized_prices.index.name = 'Date'
                normalized_prices = normalized_prices.reset_index()
                
                # Create plot
                fig = px.line(
                    normalized_prices,
                    x='Date',
                    y=[ticker1, ticker2],
                    title=f"Price Movement Comparison: {ticker1} vs {ticker2} (Correlation: {corr_value:.2f})",
                    labels={"value": "Normalized Price (%)", "variable": "Ticker"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data for correlation analysis. Please add more tickers or extend the time period.")
