#app.py
if __name__ == "__main__":
    # Calculate metrics
    metrics = calculate_metrics(stock_data)
    
    # Identify notable events
    notable_events = identify_notable_events(metrics, significance_threshold)
    
    # Render dashboard with tabs
    render_dashboard(stock_data, metrics, notable_events)
    
    # Add additional analysis for dividend stocks
    dividend_metrics = {ticker: metric for ticker, metric in metrics.items() if ticker in dividend_stocks_list}
    if dividend_metrics:
        analyze_dividend_stocks(metrics)
    
    # Correlation analysis
    correlation_analysis(stock_data)

    main()