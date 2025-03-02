"""
Components for rendering individual ticker analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils.prediction_utils import build_price_prediction_model
from utils.data_utils import fetch_prediction_data

def render_ticker_analysis(ticker, metric, ticker_tab, show_dividend=False):
    """
    Render detailed analysis for an individual ticker.
    
    Args:
        ticker (str): Ticker symbol
        metric (dict): Metrics for the ticker
        ticker_tab: Streamlit tab object
        show_dividend (bool): Whether to show dividend information
    """
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
                            'value': metric['r
