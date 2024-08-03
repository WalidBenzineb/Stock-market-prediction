import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    """
    Clean and prepare stock data for prediction models.
    
    :param df: DataFrame with raw stock data
    :return: Tuple of (X, y, scaler) where X is the feature matrix, y is the target vector, and scaler is the fitted MinMaxScaler
    """
    # Drop any rows with missing values
    df = df.dropna()
    
    # Create features
    df['Returns'] = df['Close'].pct_change()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Drop rows with NaN values created by rolling averages
    df = df.dropna()
    
    # Create target variable (next day's closing price)
    df['Target'] = df['Close'].shift(-1)
    
    # Drop the last row (which will have NaN in the Target column)
    df = df[:-1]
    
    # Select features for prediction
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_50', 'SMA_200']
    X = df[features]
    y = df['Target']
    
    # Normalize features and target
    scaler = MinMaxScaler()
    X_y_scaled = scaler.fit_transform(np.column_stack((X, y)))
    
    X_scaled = X_y_scaled[:, :-1]
    y_scaled = X_y_scaled[:, -1]
    
    # Save processed data
    processed_df = pd.DataFrame(X_y_scaled, columns=features + ['Target'])
    processed_df.to_csv(f"data/processed/{df.index[0].strftime('%Y-%m-%d')}_{df.index[-1].strftime('%Y-%m-%d')}_processed.csv")
    
    return X_scaled, y_scaled, scaler

if __name__ == "__main__":
    # Example usage
    raw_data = pd.read_csv("data/raw/GOOGL_data.csv", index_col=0, parse_dates=True)
    X, y, scaler = clean_data(raw_data)
    print("Data cleaned and processed.")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")