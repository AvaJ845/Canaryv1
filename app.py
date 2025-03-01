import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data for a list of tickers.
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols.
    start_date : str
        Start date in format 'YYYY-MM-DD'.
    end_date : str
        End date in format 'YYYY-MM-DD'.
        
    Returns:
    --------
    dict
        Dictionary with tickers as keys and DataFrames of stock data as values.
    """
    stock_data = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                stock_data[ticker] = data
            else:
                print(f"Warning: No data found for {ticker}")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    return stock_data

def calculate_metrics(stock_data):
    """
    Calculate key financial metrics for each stock.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary with tickers as keys and DataFrames of stock data as values.
        
    Returns:
    --------
    dict
        Dictionary with tickers as keys and metrics as values.
    """
    metrics = {}
    
    for ticker, data in stock_data.items():
        # Skip if data is empty
        if data.empty:
            continue
            
        # Calculate daily returns
        data['Daily_Return'] = data['Adj Close'].pct_change()
        
        # Basic metrics
        metrics[ticker] = {
            'start_price': data['Adj Close'].iloc[0],
            'end_price': data['Adj Close'].iloc[-1],
            'total_return': (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0] - 1) * 100,
            'avg_daily_return': data['Daily_Return'].mean() * 100,
            'volatility': data['Daily_Return'].std() * 100 * np.sqrt(252),  # Annualized volatility
            'max_drawdown': calculate_max_drawdown(data['Adj Close']),
            'sharpe_ratio': calculate_sharpe_ratio(data['Daily_Return']),
            'beta': calculate_beta(data, stock_data.get('^GSPC', None))  # Using S&P 500 as market
        }
        
        # Volume analysis
        metrics[ticker]['avg_volume'] = data['Volume'].mean()
        metrics[ticker]['volume_trend'] = calculate_trend(data['Volume'])
        
        # Price momentum
        metrics[ticker]['momentum_1m'] = calculate_momentum(data, 21)  # ~1 month in trading days
        metrics[ticker]['momentum_3m'] = calculate_momentum(data, 63)  # ~3 months in trading days
        metrics[ticker]['momentum_6m'] = calculate_momentum(data, 126)  # ~6 months in trading days
        
    return metrics

def calculate_max_drawdown(prices):
    """Calculate the maximum drawdown from peak to trough."""
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices/peak - 1.0)
    return drawdown.min() * 100  # Convert to percentage

def calculate_sharpe_ratio(returns, risk_free_rate=0.02/252):
    """Calculate the annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

def calculate_beta(stock_data, market_data, window=60):
    """Calculate the beta of a stock relative to the market."""
    if market_data is None or stock_data is None:
        return None
        
    # Calculate returns
    stock_returns = stock_data['Daily_Return'].dropna()
    market_returns = market_data['Daily_Return'].dropna()
    
    # Align the dates
    returns_df = pd.DataFrame({'stock': stock_returns, 'market': market_returns})
    returns_df = returns_df.dropna()
    
    if len(returns_df) < 2:
        return None
        
    # Calculate beta using covariance / variance
    covariance = returns_df['stock'].cov(returns_df['market'])
    market_variance = returns_df['market'].var()
    
    if market_variance == 0:
        return None
        
    return covariance / market_variance

def calculate_trend(data, window=20):
    """Calculate the trend using linear regression slope."""
    if len(data) < window:
        return 0
        
    x = np.arange(len(data[-window:]))
    y = data[-window:].values
    slope, _, _, _, _ = stats.linregress(x, y)
    
    return slope

def calculate_momentum(data, days):
    """Calculate price momentum over specified number of days."""
    if len(data) < days:
        return 0
    
    return (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[-min(days, len(data))] - 1) * 100

def identify_notable_events(metrics, significance_threshold=2.0):
    """
    Identify statistically significant events in the stock data.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with tickers as keys and metrics as values.
    significance_threshold : float
        Z-score threshold for identifying significant events.
        
    Returns:
    --------
    dict
        Dictionary with tickers as keys and lists of notable events as values.
    """
    notable_events = {}
    
    # Extract metrics for z-score calculation
    momentum_1m = [m['momentum_1m'] for m in metrics.values() if 'momentum_1m' in m]
    momentum_3m = [m['momentum_3m'] for m in metrics.values() if 'momentum_3m' in m]
    volatility = [m['volatility'] for m in metrics.values() if 'volatility' in m]
    
    # Calculate mean and std for z-scores
    mom_1m_mean, mom_1m_std = np.mean(momentum_1m), np.std(momentum_1m) if momentum_1m else (0, 1)
    mom_3m_mean, mom_3m_std = np.mean(momentum_3m), np.std(momentum_3m) if momentum_3m else (0, 1)
    vol_mean, vol_std = np.mean(volatility), np.std(volatility) if volatility else (0, 1)
    
    # Avoid division by zero
    mom_1m_std = max(mom_1m_std, 0.0001)
    mom_3m_std = max(mom_3m_std, 0.0001)
    vol_std = max(vol_std, 0.0001)
    
    for ticker, metric in metrics.items():
        events = []
        
        # Check for significant momentum (positive or negative)
        if 'momentum_1m' in metric:
            z_score = (metric['momentum_1m'] - mom_1m_mean) / mom_1m_std
            if abs(z_score) > significance_threshold:
                direction = "positive" if z_score > 0 else "negative"
                events.append(f"Significant {direction} 1-month momentum (z-score: {z_score:.2f})")
        
        if 'momentum_3m' in metric:
            z_score = (metric['momentum_3m'] - mom_3m_mean) / mom_3m_std
            if abs(z_score) > significance_threshold:
                direction = "positive" if z_score > 0 else "negative"
                events.append(f"Significant {direction} 3-month momentum (z-score: {z_score:.2f})")
        
        # Check for unusual volatility
        if 'volatility' in metric:
            z_score = (metric['volatility'] - vol_mean) / vol_std
            if z_score > significance_threshold:
                events.append(f"Unusually high volatility (z-score: {z_score:.2f})")
        
        # Check for extreme drawdowns
        if 'max_drawdown' in metric and metric['max_drawdown'] < -20:
            events.append(f"Severe drawdown of {metric['max_drawdown']:.2f}%")
        
        # Only add to notable_events if there are actual events
        if events:
            notable_events[ticker] = events
    
    return notable_events

def analyze_dividend_stocks(dividend_metrics):
    """
    Perform additional analysis for dividend stocks.
    
    Parameters:
    -----------
    dividend_metrics : dict
        Dictionary with dividend stock tickers as keys and metrics as values.
    """
    results = {}
    
    for ticker, metrics in dividend_metrics.items():
        # This would typically involve fetching additional dividend-specific data
        # For this example, we'll just print some insights
        print(f"\nDividend Analysis for {ticker}:")
        print(f"- Total Return: {metrics['total_return']:.2f}%")
        print(f"- Volatility: {metrics['volatility']:.2f}%")
        
        if 'dividend_yield' in metrics:
            print(f"- Dividend Yield: {metrics['dividend_yield']:.2f}%")
        
        # In a real implementation, you might calculate:
        # - Dividend growth rate
        # - Dividend consistency
        # - Payout ratio
        # - Dividend-adjusted total return
    
    return results

def correlation_analysis(stock_data):
    """
    Perform correlation analysis between stocks.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary with tickers as keys and DataFrames of stock data as values.
    """
    # Extract adjusted close prices
    close_prices = {}
    for ticker, data in stock_data.items():
        if not data.empty and 'Adj Close' in data.columns:
            close_prices[ticker] = data['Adj Close']
    
    # Create a DataFrame with all close prices
    prices_df = pd.DataFrame(close_prices)
    
    # Calculate correlation matrix
    correlation_matrix = prices_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Stock Price Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    print("\nCorrelation Analysis:")
    print("- Correlation matrix saved to 'correlation_matrix.png'")
    
    # Identify highly correlated pairs
    high_corr_pairs = []
    
    for i, ticker1 in enumerate(correlation_matrix.columns):
        for j, ticker2 in enumerate(correlation_matrix.columns):
            if i < j:  # Only look at upper triangle of correlation matrix
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.8:  # Threshold for "high" correlation
                    high_corr_pairs.append((ticker1, ticker2, corr))
    
    # Print highly correlated pairs
    if high_corr_pairs:
        print("- Highly correlated stock pairs:")
        for ticker1, ticker2, corr in high_corr_pairs:
            print(f"  * {ticker1} and {ticker2}: {corr:.2f}")
    else:
        print("- No highly correlated stock pairs found (threshold: 0.8)")

def render_dashboard(stock_data, metrics, notable_events):
    """
    Create a dashboard with visualizations of stock performance.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary with tickers as keys and DataFrames of stock data as values.
    metrics : dict
        Dictionary with tickers as keys and metrics as values.
    notable_events : dict
        Dictionary with tickers as keys and lists of notable events as values.
    """
    # Create a basic dashboard with multiple figures
    num_stocks = len(stock_data)
    
    # Price chart for each stock
    plt.figure(figsize=(15, 10))
    
    for i, (ticker, data) in enumerate(stock_data.items(), 1):
        if data.empty:
            continue
            
        plt.subplot(num_stocks, 1, i)
        plt.plot(data.index, data['Adj Close'])
        plt.title(f"{ticker} Price")
        plt.ylabel("Price ($)")
        plt.grid(True, alpha=0.3)
        
        # Add annotations for notable events
        if ticker in notable_events:
            # Just mark with vertical lines for simplicity
            plt.axvline(x=data.index[-1], color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('price_charts.png')
    plt.close()
    
    # Performance comparison
    plt.figure(figsize=(12, 8))
    
    # Calculate normalized prices (starting at 100)
    normalized_prices = {}
    for ticker, data in stock_data.items():
        if not data.empty and len(data) > 0:
            start_price = data['Adj Close'].iloc[0]
            if start_price > 0:
                normalized_prices[ticker] = data['Adj Close'] / start_price * 100
    
    # Plot normalized prices
    for ticker, prices in normalized_prices.items():
        plt.plot(prices.index, prices, label=ticker)
    
    plt.title('Comparative Performance (Normalized to 100)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('comparative_performance.png')
    plt.close()
    
    # Key metrics table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    tickers = list(metrics.keys())
    metrics_to_display = ['total_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
    table_data = []
    
    # Add headers
    headers = ['Ticker'] + [m.replace('_', ' ').title() for m in metrics_to_display]
    table_data.append(headers)
    
    # Add data rows
    for ticker in tickers:
        row = [ticker]
        for metric in metrics_to_display:
            if metric in metrics[ticker]:
                value = metrics[ticker][metric]
                # Format based on metric type
                if metric in ['total_return', 'volatility', 'max_drawdown']:
                    row.append(f"{value:.2f}%")
                else:
                    row.append(f"{value:.2f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Create the table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Key Performance Metrics')
    plt.tight_layout()
    plt.savefig('metrics_table.png')
    plt.close()
    
    # Summary of notable events
    if notable_events:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        event_data = []
        event_data.append(['Ticker', 'Notable Events'])
        
        for ticker, events in notable_events.items():
            for i, event in enumerate(events):
                if i == 0:
                    event_data.append([ticker, event])
                else:
                    event_data.append(['', event])
        
        # Create the table
        event_table = ax.table(cellText=event_data, loc='center', cellLoc='left')
        event_table.auto_set_font_size(False)
        event_table.set_fontsize(10)
        event_table.scale(1, 1.5)
        
        # Adjust column widths
        event_table.auto_set_column_width([0, 1])
        
        plt.title('Notable Market Events')
        plt.tight_layout()
        plt.savefig('notable_events.png')
        plt.close()
    
    print("\nDashboard created successfully!")
    print("- Price charts saved to 'price_charts.png'")
    print("- Comparative performance chart saved to 'comparative_performance.png'")
    print("- Metrics table saved to 'metrics_table.png'")
    if notable_events:
        print("- Notable events table saved to 'notable_events.png'")

def main():
    """Main function to run the stock analysis dashboard."""
    # Default parameters
    default_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', '^GSPC']  # Including S&P 500
    default_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    default_end_date = datetime.now().strftime('%Y-%m-%d')
    default_significance_threshold = 2.0
    
    # Known dividend stocks (example)
    dividend_stocks_list = ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO']
    
    # Get user input or use defaults
    print("Stock Analysis Dashboard")
    print("=======================")
    
    # Get tickers
    tickers_input = input(f"Enter stock tickers separated by commas (default: {','.join(default_tickers)}): ")
    tickers = tickers_input.split(',') if tickers_input else default_tickers
    tickers = [ticker.strip().upper() for ticker in tickers]
    
    # Get date range
    start_date_input = input(f"Enter start date (YYYY-MM-DD) (default: {default_start_date}): ")
    start_date = start_date_input if start_date_input else default_start_date
    
    end_date_input = input(f"Enter end date (YYYY-MM-DD) (default: {default_end_date}): ")
    end_date = end_date_input if end_date_input else default_end_date
    
    # Get significance threshold
    threshold_input = input(f"Enter significance threshold for events (default: {default_significance_threshold}): ")
    try:
        significance_threshold = float(threshold_input) if threshold_input else default_significance_threshold
    except ValueError:
        print(f"Invalid threshold. Using default: {default_significance_threshold}")
        significance_threshold = default_significance_threshold
    
    # Fetch stock data
    print("\nFetching stock data...")
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    
    if not stock_data:
        print("No stock data retrieved. Exiting.")
        return
    
    print(f"Data retrieved for {len(stock_data)} stocks.")
    
    # Calculate metrics
    metrics = calculate_metrics(stock_data)
    
    # Identify notable events
    notable_events = identify_notable_events(metrics, significance_threshold)
    
    # Render dashboard with tabs
    render_dashboard(stock_data, metrics, notable_events)
    
    # Add additional analysis for dividend stocks
    dividend_metrics = {ticker: metric for ticker, metric in metrics.items() if ticker in dividend_stocks_list}
    if dividend_metrics:
        analyze_dividend_stocks(dividend_metrics)
    
    # Correlation analysis
    correlation_analysis(stock_data)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()