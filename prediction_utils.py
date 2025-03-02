#prediction_utils.py
"""
Utilities for building and applying price prediction models.
"""

import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils.data_utils import fetch_prediction_data, create_features
import config

def build_price_prediction_model(ticker, model_type, prediction_days=7):
    """
    Build and train a price prediction model for the given ticker.
    
    Args:
        ticker (str): Ticker symbol
        model_type (str): Type of model ('Linear Regression' or 'Random Forest')
        prediction_days (int): Number of days to predict
        
    Returns:
        tuple: (predictions, future_dates, model_metrics)
    """
    # Fetch data
    data = fetch_prediction_data(ticker)
    if data is None or data.empty:
        return None, None, None
    
    # Create features
    features_df = create_features(data)
    if len(features_df) < 100:  # Not enough data
        return None, None, None
    
    # Prepare features and target
    feature_columns = config.FEATURE_COLUMNS
    
    X = features_df[feature_columns].values
    y = features_df['Target'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:  # Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    # Get latest feature values for prediction
    latest_features = features_df[feature_columns].iloc[-1].values.reshape(1, -1)
    latest_features_scaled = scaler.transform(latest_features)
    
    # Generate predictions for future days
    predictions = []
    latest_price = features_df['Close'].iloc[-1]
    current_features = latest_features_scaled.copy()
    
    # For visualization, include the last known price
    predictions.append(latest_price)
    
    for i in range(prediction_days):
        # Predict next price
        next_price = model.predict(current_features)[0]
        predictions.append(next_price)
        
        # Update features for next prediction (simplified approach)
        # In a real application, you would need to update all features
        # This is a simplified version that just updates price-related features
        price_change = ((next_price / latest_price) - 1) * 100
        current_features[0][feature_columns.index('Price_Change')] = price_change
        
        # Update latest price for next iteration
        latest_price = next_price
    
    # Feature importance (for Random Forest)
    feature_importance = None
    if model_type == "Random Forest":
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
    
    # Calculate accuracy metrics
    y_pred_test = model.predict(X_test_scaled)
    mse = np.mean((y_pred_test - y_test) ** 2)
    accuracy = 1 - np.mean(np.abs((y_test - y_pred_test) / y_test))
    
    # Prepare dates for visualization
    last_date = features_df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(prediction_days + 1)]
    
    return predictions, future_dates, {
        'mse': mse,
        'accuracy': accuracy,
        'feature_importance': feature_importance
    }
