import sys
import os
from PyQt5.QtWidgets import QApplication

from src.data.data_acquisition import get_stock_data, update_stock_data
from src.data.data_cleaning import clean_data
from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.lstm import LSTMModel
from src.gui.main_window import StockPredictionGUI

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data/raw', 'data/processed']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    # Create necessary directories
    create_directories()

    # Define stock tickers
    tickers = ["GOOGL", "AAPL", "MSFT"]
    
    # Data acquisition
    stocks = {}
    for ticker in tickers:
        try:
            stocks[ticker] = update_stock_data(ticker)
        except FileNotFoundError:
            stocks[ticker] = get_stock_data(ticker, "2020-01-01", "2024-07-01")
    
    # Data cleaning
    cleaned_data = {}
    for ticker, data in stocks.items():
        X, y, scaler = clean_data(data)
        cleaned_data[ticker] = (X, y, scaler)
    
    # Model creation
    models = {
        'Linear Regression': LinearRegressionModel(),
        'Random Forest': RandomForestModel(),
        'LSTM': LSTMModel(input_shape=(1, X.shape[1]))
    }
    
    # Model training
    for ticker, (X, y, _) in cleaned_data.items():
        print(f"Training models for {ticker}...")
        for name, model in models.items():
            model.train(X, y)
        print(f"Finished training models for {ticker}")
    
    # GUI
    app = QApplication(sys.argv)
    gui = StockPredictionGUI(stocks, models, cleaned_data)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()