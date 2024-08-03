import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

def predict_future(model, last_known_data, scaler, days=30):
    """
    Predict future stock prices using the trained model with improved realism.
    
    :param model: Trained prediction model
    :param last_known_data: Last known data point (features and target)
    :param scaler: Fitted scaler used for inverse transformation
    :param days: Number of days to predict into the future
    :return: Array of predicted future prices
    """
    print(f"Debug: last_known_data shape: {last_known_data.shape}")
    print(f"Debug: last_known_data: {last_known_data}")

    future_pred = []
    current_pred = last_known_data[:, :-1]  # Use all features except the last one (target)
    
    for _ in range(days):
        # Make prediction
        next_pred = model.predict(current_pred)
        
        # Add some randomness to simulate market volatility
        volatility = 0.02  # Adjust this value to increase/decrease randomness
        next_pred = next_pred * (1 + np.random.normal(0, volatility))
        
        # Ensure the prediction doesn't go negative
        next_pred = np.maximum(next_pred, 0)
        
        # Append the prediction
        future_pred.append(next_pred[0])
        
        # Update current_pred for next iteration (assuming the model expects the same input shape)
        current_pred[0, -1] = next_pred

    # Prepare data for inverse transform
    future_pred_array = np.array(future_pred).reshape(-1, 1)
    last_known_features = last_known_data[0, :-1]  # All features except the last one (which is the target)
    last_known_features_repeated = np.tile(last_known_features, (days, 1))
    
    # Combine features with predictions
    future_pred_with_features = np.column_stack((last_known_features_repeated, future_pred_array))
    
    # Inverse transform predictions
    future_pred_inv = scaler.inverse_transform(future_pred_with_features)[:, -1]
    
    print(f"Debug: First few future predictions: {future_pred_inv[:5]}")
    
    return future_pred_inv

def plot_future_prediction(historical_data, future_pred):
    """
    Plot historical data and future predictions.
    
    :param historical_data: Historical stock prices
    :param future_pred: Predicted future prices
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(historical_data)), historical_data, label='Historical Prices', linewidth=1)
    plt.plot(range(len(historical_data)-1, len(historical_data) + len(future_pred)), 
             [historical_data[-1]] + list(future_pred), label='Future Predictions', linewidth=1)
    plt.legend()
    plt.title('Stock Price Forecast')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    # Generate some dummy data
    np.random.seed(42)
    dates = np.array([i for i in range(1000)])
    prices = np.cumsum(np.random.randn(1000)) + 100
    X = dates.reshape(-1, 1)
    y = prices

    # Create and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create a scaler and fit it to the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(np.column_stack((X, y.reshape(-1, 1))))

    # Get last known data point
    last_known_data = scaled_data[-1:, :]  # Include all features and the target

    # Predict future prices
    future_prices = predict_future(model, last_known_data, scaler, days=30)

    # Plot predictions
    plot_future_prediction(y, future_prices)

    print("Last known price:", y[-1])
    print("First predicted price:", future_prices[0])
    print("Last predicted price:", future_prices[-1])