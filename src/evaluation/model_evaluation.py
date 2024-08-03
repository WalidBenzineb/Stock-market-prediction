import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.lstm import LSTMModel

def evaluate_models(models, X_test, y_test, scaler):
    """
    Evaluate multiple models and compare their performance.
    
    :param models: Dictionary of trained models
    :param X_test: Test feature matrix
    :param y_test: Test target vector
    :param scaler: Fitted scaler used for inverse transformation
    :return: Dictionary with evaluation results for each model
    """
    results = {}
    
    for name, model in models.items():
        predictions = model.predict(X_test)
        
        # Inverse transform predictions and actual values
        X_y_test = np.column_stack((X_test, y_test))
        X_y_pred = np.column_stack((X_test, predictions))
        
        X_y_test_inv = scaler.inverse_transform(X_y_test)
        X_y_pred_inv = scaler.inverse_transform(X_y_pred)
        
        y_test_inv = X_y_test_inv[:, -1]
        predictions_inv = X_y_pred_inv[:, -1]
        
        mse = mean_squared_error(y_test_inv, predictions_inv)
        r2 = r2_score(y_test_inv, predictions_inv)
        
        results[name] = {
            'mse': mse,
            'r2': r2,
            'predictions': predictions_inv
        }
        
        print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")
    
    return results

def plot_predictions(results, y_test, scaler, X_test):
    """
    Plot predictions vs actual values for each model with thinner lines.
    
    :param results: Dictionary with evaluation results for each model
    :param y_test: Test target vector
    :param scaler: Fitted scaler used for inverse transformation
    :param X_test: Test feature matrix
    """
    X_y_test = np.column_stack((X_test, y_test))
    X_y_test_inv = scaler.inverse_transform(X_y_test)
    y_test_inv = X_y_test_inv[:, -1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual', linewidth=0.8, color='black')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':']
    
    for i, (name, result) in enumerate(results.items()):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        plt.plot(result['predictions'], label=f'{name} Prediction', 
                 linewidth=0.8, color=color, linestyle=linestyle)
    
    plt.legend()
    plt.title('Stock Price Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

def evaluate_single_model(model, X_test, y_test, scaler):
    """
    Evaluate a single model and return its performance metrics.
    
    :param model: Trained model
    :param X_test: Test feature matrix
    :param y_test: Test target vector
    :param scaler: Fitted scaler used for inverse transformation
    :return: Dictionary with evaluation results
    """
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    X_y_test = np.column_stack((X_test, y_test))
    X_y_pred = np.column_stack((X_test, predictions))
    
    X_y_test_inv = scaler.inverse_transform(X_y_test)
    X_y_pred_inv = scaler.inverse_transform(X_y_pred)
    
    y_test_inv = X_y_test_inv[:, -1]
    predictions_inv = X_y_pred_inv[:, -1]
    
    mse = mean_squared_error(y_test_inv, predictions_inv)
    r2 = r2_score(y_test_inv, predictions_inv)
    
    return {
        'mse': mse,
        'r2': r2,
        'predictions': predictions_inv
    }

if __name__ == "__main__":
    # Example usage
    from sklearn.preprocessing import MinMaxScaler
    
    # Generate some dummy data
    X = np.random.rand(100, 8)
    y = np.random.rand(100)
    
    # Create and train models
    models = {
        'Linear Regression': LinearRegressionModel(),
        'Random Forest': RandomForestModel(),
        'LSTM': LSTMModel(input_shape=(1, 8))
    }
    
    for name, model in models.items():
        model.train(X, y)
    
    # Prepare test data
    X_test = X[-20:]
    y_test = y[-20:]
    
    # Create a dummy scaler for this example
    scaler = MinMaxScaler()
    scaler.fit(np.column_stack([X, y.reshape(-1, 1)]))
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test, scaler)
    
    # Plot predictions
    plot_predictions(results, y_test, scaler, X_test)