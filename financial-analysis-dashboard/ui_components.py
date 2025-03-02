# ui_components.py
# This file contains UI component functions for the Streamlit app

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

def header_section(title, subtitle=None):
    """
    Display the header section of the app
    
    Parameters:
    - title (str): Main title text
    - subtitle (str, optional): Subtitle text
    """
    st.markdown(f"<h1 class='main-header'>{title}</h1>", unsafe_allow_html=True)
    
    if subtitle:
        st.markdown(f"<p>{subtitle}</p>", unsafe_allow_html=True)
    
    st.markdown("---")

def sidebar_filters(start_date_default=None, end_date_default=None):
    """
    Create sidebar filters for the app
    
    Parameters:
    - start_date_default (datetime, optional): Default start date
    - end_date_default (datetime, optional): Default end date
    
    Returns:
    - dict: Dictionary containing filter values
    """
    with st.sidebar:
        st.header("Filters")
        
        # Default dates if not provided
        if start_date_default is None:
            start_date_default = datetime.now() - timedelta(days=365)
        
        if end_date_default is None:
            end_date_default = datetime.now()
        
        # Date range selector
        start_date = st.date_input("Start Date", 
                                  value=start_date_default, 
                                  max_value=datetime.now() - timedelta(days=1))
        
        end_date = st.date_input("End Date", 
                                value=end_date_default, 
                                max_value=datetime.now())
        
        # Asset class filter
        asset_classes = ["All", "Stocks", "ETFs", "Crypto"]
        selected_asset_class = st.selectbox("Asset Class", asset_classes)
        
        # Return a dictionary of filter values
        return {
            'start_date': start_date,
            'end_date': end_date,
            'asset_class': selected_asset_class
        }

def display_metrics(metrics, columns=3):
    """
    Display a set of metrics in a multi-column layout
    
    Parameters:
    - metrics (dict): Dictionary of metric labels and values
    - columns (int): Number of columns to display
    """
    # Create columns
    cols = st.columns(columns)
    
    # Distribute metrics across columns
    for i, (label, value) in enumerate(metrics.items()):
        col_index = i % columns
        with cols[col_index]:
            st.metric(label=label, value=value)

def plot_price_chart(price_data, title="Price History"):
    """
    Plot a price chart using Plotly
    
    Parameters:
    - price_data (pandas.Series or pandas.DataFrame): Price data to plot
    - title (str): Chart title
    
    Returns:
    - plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # If price_data is a Series, convert to DataFrame
    if isinstance(price_data, pd.Series):
        price_data = pd.DataFrame({'Price': price_data})
    
    # Add each column as a trace
    for column in price_data.columns:
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data[column],
            mode='lines',
            name=column
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
    return fig

def plot_forecast_chart(historical_data, forecast_data, title="Price Forecast"):
    """
    Plot a forecast chart with historical and predicted data
    
    Parameters:
    - historical_data (pandas.Series): Historical price data
    - forecast_data (pandas.Series): Forecasted price data
    - title (str): Chart title
    
    Returns:
    - plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data.values,
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data.values,
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Add a vertical line separating historical and forecast data
    last_historical_date = historical_data.index[-1]
    
    fig.add_vline(
        x=last_historical_date,
        line_dash="dash",
        line_color="gray",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
    return fig

def plot_dividend_history(dividend_data, title="Dividend History"):
    """
    Plot dividend history
    
    Parameters:
    - dividend_data (pandas.Series): Dividend payment history
    - title (str): Chart title
    
    Returns:
    - plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add dividend data as a bar chart
    fig.add_trace(go.Bar(
        x=dividend_data.index,
        y=dividend_data.values,
        name='Dividend Payment',
        marker_color='green'
    ))
    
    # Calculate 4-point moving average for trend
    if len(dividend_data) >= 4:
        rolling_avg = dividend_data.rolling(window=4).mean()
        
        fig.add_trace(go.Scatter(
            x=rolling_avg.index,
            y=rolling_avg.values,
            mode='lines',
            name='4-Period Average',
            line=dict(color='blue', width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Dividend Amount",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
    return fig

def display_dividend_rating(rating, rating_details):
    """
    Display a dividend stock rating with visualization
    
    Parameters:
    - rating (float): Overall rating (0-5)
    - rating_details (dict): Dictionary with rating details
    """
    # Create a container for the rating
    with st.container():
        st.markdown(f"<h3>Dividend Rating: {rating}/5</h3>", unsafe_allow_html=True)
        
        # Create a progress bar for the overall rating
        st.progress(rating / 5)
        
        # Display the summary
        st.markdown(f"<p><strong>Summary:</strong> {rating_details['summary']}</p>", unsafe_allow_html=True)
        
        # Create a radar chart for the category ratings
        if 'categories' in rating_details:
            categories = list(rating_details['categories'].keys())
            values = list(rating_details['categories'].values())
            
            # Create radar chart data
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Rating'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed explanations
        if 'explanations' in rating_details:
            with st.expander("See Detailed Rating Explanations"):
                for category, explanation in rating_details['explanations'].items():
                    score = rating_details['categories'][category]
                    st.markdown(f"<h4>{category}: {score}/5</h4>", unsafe_allow_html=True)
                    st.write(explanation)

def display_portfolio_summary(portfolio_data, portfolio_metrics):
    """
    Display a summary of the portfolio
    
    Parameters:
    - portfolio_data (pandas.DataFrame): Portfolio data
    - portfolio_metrics (dict): Dictionary with portfolio metrics
    """
    # Total value and annual income
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Value", f"${portfolio_metrics['total_value']:,.2f}")
    
    with col2:
        st.metric("Annual Income", f"${portfolio_metrics['annual_income']:,.2f}")
    
    with col3:
        st.metric("Avg. Dividend Yield", f"{portfolio_metrics['avg_yield']:.2f}%")
    
    # Asset allocation
    st.subheader("Asset Allocation")
    
    # Create a pie chart for asset allocation
    fig = px.pie(
        names=portfolio_metrics['asset_allocation'].index,
        values=portfolio_metrics['asset_allocation'].values,
        title="Portfolio Allocation by Value"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display top holdings
    st.subheader("Top Holdings")
    
    # Sort by current value and get top 10
    top_holdings = portfolio_data.sort_values('Current Value', ascending=False).head(10)
    
    # Create a table for top holdings
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Symbol', 'Description', 'Quantity', 'Price', 'Value', 'Weight', 'Yield'],
            font=dict(size=12),
            align='left'
        ),
        cells=dict(
            values=[
                top_holdings['Symbol'],
                top_holdings['Description'],
                top_holdings['Quantity'].apply(lambda x: f"{x:,.2f}"),
                top_holdings['Last Price'].apply(lambda x: f"${x:,.2f}"),
                top_holdings['Current Value'].apply(lambda x: f"${x:,.2f}"),
                top_holdings['Percent Of Account'].apply(lambda x: f"{x:.2f}%"),
                top_holdings['Dist. Yield'].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A")
            ],
            font=dict(size=11),
            align='left'
        )
    )])
    
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    
    st.plotly_chart(fig, use_container_width=True)

def display_income_projection(income_data):
    """
    Display a monthly income projection chart
    
    Parameters:
    - income_data (pandas.DataFrame): DataFrame with month and income projection
    """
    st.subheader("Monthly Income Projection")
    
    # Create a bar chart for income projection
    fig = px.bar(
        income_data,
        x='Month',
        y='Income',
        title="Projected Monthly Dividend Income"
    )
    
    fig.update_traces(marker_color='green')
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Income ($)",
        yaxis_tickprefix="$"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display annual summary
    total_annual = income_data['Income'].sum()
    monthly_avg = income_data['Income'].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Projected Annual Income", f"${total_annual:,.2f}")
    
    with col2:
        st.metric("Average Monthly Income", f"${monthly_avg:,.2f}")

def display_technical_indicators(price_data):
    """
    Display technical indicators for a stock or ETF
    
    Parameters:
    - price_data (pandas.DataFrame): DataFrame with OHLC price data
    """
    st.subheader("Technical Indicators")
    
    # Calculate moving averages
    price_data['MA50'] = price_data['Close'].rolling(window=50).mean()
    price_data['MA200'] = price_data['Close'].rolling(window=200).mean()
    
    # Calculate MACD
    exp1 = price_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = price_data['Close'].ewm(span=26, adjust=False).mean()
    price_data['MACD'] = exp1 - exp2
    price_data['Signal'] = price_data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate RSI
    delta = price_data['Close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    price_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Create tabs for different indicators
    indicator_tabs = st.tabs(["Moving Averages", "MACD", "RSI", "Volume"])
    
    # Moving Averages tab
    with indicator_tabs[0]:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['Close'],
            mode='lines',
            name='Close'
        ))
        
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['MA50'],
            mode='lines',
            name='50-Day MA',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['MA200'],
            mode='lines',
            name='200-Day MA',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        ma50_last = price_data['MA50'].iloc[-1]
        ma200_last = price_data['MA200'].iloc[-1]
        close_last = price_data['Close'].iloc[-1]
        
        if close_last > ma50_last and close_last > ma200_last:
            st.markdown("**Interpretation:** Price is above both 50-day and 200-day moving averages, suggesting a strong **uptrend**.")
        elif close_last > ma50_last and close_last < ma200_last:
            st.markdown("**Interpretation:** Price is above 50-day but below 200-day moving average, suggesting a potential **early uptrend** or recovery.")
        elif close_last < ma50_last and close_last > ma200_last:
            st.markdown("**Interpretation:** Price is below 50-day but above 200-day moving average, suggesting a potential **short-term pullback** in a longer uptrend.")
        else:
            st.markdown("**Interpretation:** Price is below both 50-day and 200-day moving averages, suggesting a **downtrend**.")
    
    # MACD tab
    with indicator_tabs[1]:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=("Price", "MACD"),
                           row_heights=[0.7, 0.3])
        
        # Price plot
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['Close'],
            mode='lines',
            name='Close'
        ), row=1, col=1)
        
        # MACD plot
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='red')
        ), row=2, col=1)
        
        # Add MACD histogram
        colors = ['green' if val >= 0 else 'red' for val in (price_data['MACD'] - price_data['Signal'])]
        
        fig.add_trace(go.Bar(
            x=price_data.index,
            y=price_data['MACD'] - price_data['Signal'],
            name='Histogram',
            marker_color=colors
        ), row=2, col=1)
        
        fig.update_layout(
            title="Moving Average Convergence Divergence (MACD)",
            hovermode="x unified",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        macd_last = price_data['MACD'].iloc[-1]
        signal_last = price_data['Signal'].iloc[-1]
        
        if macd_last > signal_last:
            st.markdown("**Interpretation:** MACD is above the signal line, suggesting bullish momentum.")
        else:
            st.markdown("**Interpretation:** MACD is below the signal line, suggesting bearish momentum.")
    
    # RSI tab
    with indicator_tabs[2]:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=("Price", "RSI"),
                           row_heights=[0.7, 0.3])
        
        # Price plot
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['Close'],
            mode='lines',
            name='Close'
        ), row=1, col=1)
        
        # RSI plot
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        
        # Add overbought/oversold lines
        fig.add_shape(
            type='line',
            x0=price_data.index[0],
            y0=70,
            x1=price_data.index[-1],
            y1=70,
            line=dict(color='red', dash='dash'),
            row=2, col=1
        )
        
        fig.add_shape(
            type='line',
            x0=price_data.index[0],
            y0=30,
            x1=price_data.index[-1],
            y1=30,
            line=dict(color='green', dash='dash'),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Relative Strength Index (RSI)",
            hovermode="x unified",
            height=600
        )
        
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        rsi_last = price_data['RSI'].iloc[-1]
        
        if rsi_last > 70:
            st.markdown("**Interpretation:** RSI is above 70, suggesting the asset may be **overbought** and potentially due for a pullback.")
        elif rsi_last < 30:
            st.markdown("**Interpretation:** RSI is below 30, suggesting the asset may be **oversold** and potentially due for a bounce.")
        elif rsi_last > 50:
            st.markdown("**Interpretation:** RSI is between 50 and 70, suggesting **bullish** momentum but not yet overbought.")
        else:
            st.markdown("**Interpretation:** RSI is between 30 and 50, suggesting **bearish** momentum but not yet oversold.")
    
    # Volume tab
    with indicator_tabs[3]:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=("Price", "Volume"),
                           row_heights=[0.7, 0.3])
        
        # Price plot
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['Close'],
            mode='lines',
            name='Close'
        ), row=1, col=1)
        
        # Volume plot
        colors = ['green' if price_data['Close'].iloc[i] >= price_data['Close'].iloc[i-1] 
                 else 'red' for i in range(1, len(price_data))]
        colors.insert(0, 'gray')  # Add a color for the first bar
        
        fig.add_trace(go.Bar(
            x=price_data.index,
            y=price_data['Volume'],
            name='Volume',
            marker_color=colors
        ), row=2, col=1)
        
        # Add volume moving average
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['Volume'].rolling(window=20).mean(),
            mode='lines',
            name='20-Day Avg Volume',
            line=dict(color='blue', width=1.5)
        ), row=2, col=1)
        
        fig.update_layout(
            title="Volume Analysis",
            hovermode="x unified",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        vol_last = price_data['Volume'].iloc[-1]
        vol_avg = price_data['Volume'].rolling(window=20).mean().iloc[-1]
        vol_ratio = vol_last / vol_avg if not pd.isna(vol_avg) and vol_avg != 0 else 0
        
        if vol_ratio > 1.5:
            st.markdown(f"**Interpretation:** Current volume is **{vol_ratio:.2f}x** the 20-day average, indicating **significantly higher** interest.")
        elif vol_ratio > 1:
            st.markdown(f"**Interpretation:** Current volume is **{vol_ratio:.2f}x** the 20-day average, indicating **above average** interest.")
        elif vol_ratio > 0.5:
            st.markdown(f"**Interpretation:** Current volume is **{vol_ratio:.2f}x** the 20-day average, indicating **below average** interest.")
        else:
            st.markdown(f"**Interpretation:** Current volume is **{vol_ratio:.2f}x** the 20-day average, indicating **very low** interest.")

from plotly.subplots import make_subplots

def display_help_section():
    """
    Display a help section with information about the app
    """
    st.markdown("""
    ## How to Use This Dashboard
    
    This financial dashboard allows you to analyze stocks, ETFs, dividend stocks, and cryptocurrencies. Here's a quick guide on how to use the various features:
    
    ### Stocks Tab
    - Enter a stock ticker to view key metrics, historical prices, and forecast
    - The forecast uses your selected method and timeframe from the sidebar
    - Technical indicators are available to help with analysis
    
    ### Dividends Tab
    - Enter a dividend stock ticker to view yield, payout ratio, and growth history
    - The rating system evaluates dividends based on:
        - Yield: How high is the current dividend yield?
        - Safety: Is the payout ratio sustainable?
        - Growth: Has the company consistently raised dividends?
        - Consistency: How long has the company paid dividends?
        - Volatility: How stable is the stock price?
    
    ### ETFs Tab
    - Analyze ETFs to view performance, expense ratios, and holdings
    - Our pre-selected ETFs are shown for quick reference
    
    ### Crypto Tab
    - Enter a cryptocurrency symbol to view price data and forecasts
    - Technical indicators and market sentiment are provided
    
    ### Portfolio Tab
    - View a summary of your portfolio with allocation, income projections, and more
    - Upcoming dividend payments are shown to help with cash flow planning
    
    ### Using Forecasts
    Forecast methods available:
    - **ARIMA**: Statistical time series forecasting model
    - **Prophet**: Facebook's time series forecasting model
    - **Linear Regression**: Simple trend-based forecast
    - **LSTM**: Neural network approach (simplified implementation)
    
    Remember that forecasts are estimates and not guaranteed predictions of future prices.
    """)
    
    # Add explanation about the dividend rating system
    with st.expander("Understanding the Dividend Rating System"):
        st.markdown("""
        ### Dividend Rating System Explained
        
        Our 5-star rating system evaluates dividend stocks across five key categories:
        
        #### 1. Yield (25% of total rating)
        - 5★: ≥6% yield
        - 4★: 4-6% yield
        - 3★: 3-4% yield
        - 2★: 2-3% yield
        - 1★: <2% yield
        
        #### 2. Safety (25% of total rating)
        - 5★: <30% payout ratio
        - 4★: 30-45% payout ratio
        - 3★: 45-60% payout ratio
        - 2★: 60-75% payout ratio
        - 1★: 75-90% payout ratio
        - 0★: >90% payout ratio
        
        #### 3. Growth (20% of total rating)
        - 5★: ≥15% 5-year dividend growth
        - 4★: 10-15% growth
        - 3★: 7-10% growth
        - 2★: 3-7% growth
        - 1★: 0-3% growth
        - 0★: Dividend cuts
        
        #### 4. Consistency (20% of total rating)
        - 5★: ≥25 years (Dividend Aristocrat)
        - 4★: 15-25 years
        - 3★: 10-15 years
        - 2★: 5-10 years
        - 1★: <5 years
        - 0★: No established history
        
        #### 5. Volatility (10% of total rating)
        - 5★: Beta <0.5
        - 4★: Beta 0.5-0.8
        - 3★: Beta 0.8-1.0
        - 2★: Beta 1.0-1.2
        - 1★: Beta 1.2-1.5
        - 0★: Beta >1.5
        
        The overall rating is a weighted average of these five categories, rounded to the nearest half-star.
        """)