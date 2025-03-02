# Financial Analysis Dashboard

A comprehensive Streamlit application for analyzing stocks, ETFs, dividend stocks, and cryptocurrencies.

## Features

- **Stocks Analysis**: Track and forecast stock prices with multiple forecasting methods
- **Dividend Analysis**: Analyze dividend stocks with a custom rating system
- **ETF Analysis**: View ETF performance, holdings, and expense ratios
- **Cryptocurrency Analysis**: Track and forecast crypto prices
- **Portfolio Management**: View portfolio allocation, income projections, and metrics

## System Requirements

- Python 3.8+
- Required Python packages are listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/financial-analysis-dashboard.git
   cd financial-analysis-dashboard
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Start the Streamlit app:
```
streamlit run main.py
```

The app will open in your default web browser, usually at http://localhost:8501

## Project Structure

- `main.py`: Main entry point for the Streamlit application
- `stock_analysis.py`: Functions for analyzing stocks
- `crypto_analysis.py`: Functions for analyzing cryptocurrencies
- `dividend_analysis.py`: Functions for analyzing dividend stocks
- `etf_analysis.py`: Functions for analyzing ETFs
- `portfolio.py`: Functions for portfolio management and analysis
- `ui_components.py`: UI component functions for the Streamlit app
- `requirements.txt`: Required Python packages

## Modules Explained

### Main App (`main.py`)
The main entry point for the application. Sets up the Streamlit interface, sidebar filters, and tabs for different sections (Stocks, Dividends, ETFs, Crypto, Portfolio).

### Stock Analysis (`stock_analysis.py`)
- `get_stock_data()`: Retrieves historical stock data from Yahoo Finance
- `analyze_stock()`: Performs basic analysis on stock data (returns, volatility, etc.)
- `forecast_stock()`: Forecasts stock prices using various methods (ARIMA, Prophet, etc.)

### Crypto Analysis (`crypto_analysis.py`)
- `get_crypto_data()`: Retrieves historical cryptocurrency data
- `analyze_crypto()`: Analyzes cryptocurrency data (returns, volatility, etc.)
- `forecast_crypto()`: Forecasts cryptocurrency prices

### Dividend Analysis (`dividend_analysis.py`)
- `get_dividend_data()`: Retrieves dividend information for a stock
- `analyze_dividend()`: Analyzes dividend metrics (yield, payout ratio, etc.)
- `rate_dividend()`: Rates a dividend stock on a scale of 1-5 based on various metrics

### ETF Analysis (`etf_analysis.py`)
- `get_etf_data()`: Retrieves ETF data and holdings
- `analyze_etf()`: Analyzes ETF performance and metrics
- `compare_etfs()`: Compares multiple ETFs based on key metrics
- `get_etf_sector_exposure()`: Gets sector exposure for an ETF

### Portfolio Management (`portfolio.py`)
- `load_portfolio_data()`: Loads the predefined portfolio data
- `calculate_portfolio_metrics()`: Calculates key metrics for the portfolio
- `update_portfolio_prices()`: Updates portfolio with current market prices
- `add_portfolio_position()`: Adds a new position to the portfolio
- `remove_portfolio_position()`: Removes a position from the portfolio
- `get_income_projection()`: Projects dividend income for future months
- `get_portfolio_performance_history()`: Calculates historical performance

### UI Components (`ui_components.py`)
- `header_section()`: Displays the header section of the app
- `sidebar_filters()`: Creates sidebar filters for the app
- `display_metrics()`: Displays a set of metrics in a multi-column layout
- `plot_price_chart()`: Plots a price chart using Plotly
- `plot_forecast_chart()`: Plots a forecast chart with historical and predicted data
- `plot_dividend_history()`: Plots dividend history
- `display_dividend_rating()`: Displays a dividend stock rating with visualization
- `display_portfolio_summary()`: Displays a summary of the portfolio
- `display_income_projection()`: Displays a monthly income projection chart
- `display_technical_indicators()`: Displays technical indicators for a stock or ETF
- `display_help_section()`: Displays a help section with information about the app

## Dividend Rating System

The app uses a comprehensive 5-star rating system to evaluate dividend stocks:

1. **Yield (25%)**: Evaluates the current dividend yield
2. **Safety (25%)**: Assesses the sustainability of the dividend based on payout ratio
3. **Growth (20%)**: Measures the historical dividend growth rate
4. **Consistency (20%)**: Evaluates the company's track record of paying dividends
5. **Volatility (10%)**: Measures the stock's price stability using beta

## Forecasting Methods

The app offers multiple forecasting methods:

1. **ARIMA**: Statistical time-series forecasting that accounts for trends and seasonality
2. **Prophet**: Facebook's forecasting tool designed for business time series
3. **Linear Regression**: Simple trend-based forecasting
4. **LSTM**: Neural network approach for time series forecasting

## Data Sources

- Historical price data is retrieved from **Yahoo Finance** via the `yfinance` package
- Dividend data is calculated from historical dividends and company financial information
- Portfolio data is loaded from a predefined dataset (can be extended to load from CSV or database)

## Future Enhancements

- Add user authentication for personal portfolios
- Implement data caching for improved performance
- Add backtesting for portfolio strategies
- Include more advanced technical indicators
- Add option analysis capabilities
- Integrate news and sentiment analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data provided by Yahoo Finance
- Built with Streamlit
- Forecasting models from statsmodels, Prophet, and scikit-learn