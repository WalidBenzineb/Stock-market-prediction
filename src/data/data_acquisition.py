import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def get_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance.
    
    :param ticker: Stock ticker symbol
    :param start_date: Start date for data retrieval (format: 'YYYY-MM-DD')
    :param end_date: End date for data retrieval (format: 'YYYY-MM-DD')
    :return: DataFrame with stock data
    """
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    # Save data to CSV
    filename = f"data/raw/{ticker}_data.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
    data.to_csv(filename)
    print(f"Data saved to {filename}")
    
    return data

def update_stock_data(ticker):
    """
    Update stock data for a given ticker, fetching only the new data since the last update.
    
    :param ticker: Stock ticker symbol
    :return: Updated DataFrame with stock data
    """
    filename = f"data/raw/{ticker}_data.csv"
    try:
        existing_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        last_date = existing_data.index[-1].strftime('%Y-%m-%d')
        start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        new_data = get_stock_data(ticker, start_date, end_date)
        updated_data = pd.concat([existing_data, new_data])
        updated_data.to_csv(filename)
        print(f"Data updated in {filename}")
        return updated_data
    except FileNotFoundError:
        print(f"No existing data found for {ticker}. Fetching all available data.")
        return get_stock_data(ticker, "2000-01-01", datetime.now().strftime('%Y-%m-%d'))

if __name__ == "__main__":
    # Example usage
    google_data = get_stock_data("GOOGL", "2020-01-01", "2024-07-01")
    apple_data = get_stock_data("AAPL", "2020-01-01", "2024-07-01")
    microsoft_data = get_stock_data("MSFT", "2020-01-01", "2024-07-01")
    
    # Example of updating data
    updated_google_data = update_stock_data("GOOGL")