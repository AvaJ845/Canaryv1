"""
Components for analyzing dividend stocks.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

def analyze_dividend_stocks(dividend_metrics):
    """
    Render dividend analysis overview.
    
    Args:
        dividend_metrics (dict): Dictionary of dividend stock metrics
    """
    if not dividend_metrics:
        st.info("No dividend data available.")
        return
        
    st.header("Dividend Analysis Overview")
    
    # Calculate and display average dividend yield
    yields = [metric['dividend_yield'] for ticker, metric in dividend_metrics.items() 
              if metric.get('dividend_yield') is not None]
    
    if yields:
        avg_yield = sum(yields) / len(yields)
        highest_yield = max(yields)
        highest_yield_ticker = next((ticker for ticker, metric in dividend_metrics.items() 
                                    if metric.get('dividend_yield') == highest_yield), "Unknown")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Dividend Yield", f"{avg_yield:.2f}%")
        with col2:
            st.metric(f"Highest Yield ({highest_yield_ticker})", f"{highest_yield:.2f}%")
        
        # Create dividend yield comparison chart
        dividend_data = []
        for ticker, metric in dividend_metrics.items():
            if metric.get('dividend_yield') is not None:
                dividend_data.append({
                    'Ticker': ticker,
                    'Dividend Yield (%)': metric['dividend_yield']
                })
        
        if dividend_data:
            dividend_df = pd.DataFrame(dividend_data)
            
            # Sort by yield in descending order
            dividend_df = dividend_df.sort_values('Dividend Yield (%)', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                dividend_df,
                x='Ticker',
                y='Dividend Yield (%)',
                title="Dividend Yield Comparison",
                color='Dividend Yield (%)',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display dividend payout ratios if available
            payout_data = []
            for ticker, metric in dividend_metrics.items():
                if metric.get('dividend_payout') is not None:
                    payout_data.append({
                        'Ticker': ticker,
                        'Payout Ratio (%)': metric['dividend_payout']
                    })
            
            if payout_data:
                st.subheader("Dividend Payout Ratios")
                payout_df = pd.DataFrame(payout_data)
                payout_df = payout_df.sort_values('Payout Ratio (%)', ascending=False)
                
                fig = px.bar(
                    payout_df,
                    x='Ticker',
                    y='Payout Ratio (%)',
                    title="Dividend Payout Ratio Comparison",
                    color='Payout Ratio (%)',
                    color_continuous_scale='Viridis'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No dividend yield data available for the selected stocks.")