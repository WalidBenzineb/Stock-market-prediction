import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = MinMaxScaler()

    def train(self, X, y, epochs=50, batch_size=32):
        """
        Train the LSTM model.
        
        :param X: Feature matrix
        :param y: Target vector
        :param epochs: Number of epochs to train
        :param batch_size: Batch size for training
        """
        # Scale the input data
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape X for LSTM [samples, time steps, features]
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        :param X: Feature matrix
        :return: Array of predictions
        """
        # Scale the input data
        X_scaled = self.scaler.transform(X)
        
        # Reshape X for LSTM [samples, time steps, features]
        if X_scaled.ndim == 2:
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        elif X_scaled.ndim == 1:
            X_reshaped = X_scaled.reshape((1, 1, X_scaled.shape[0]))
        else:
            raise ValueError("Input shape not supported")
        
        return self.model.predict(X_reshaped).flatten()
    
    def evaluate(self):
        """
        Evaluate the model's performance.
        
        :return: Dictionary with model performance metrics
        """
        train_loss = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        test_loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return {
            "train_loss": train_loss,
            "test_loss": test_loss
        }

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate some dummy data
    X = np.random.rand(100, 8)
    y = np.random.rand(100)
    
    # Create and train the LSTM model
    lstm_model = LSTMModel(input_shape=(1, 8))
    lstm_model.train(X, y, epochs=10)
    
    # Make predictions
    X_test = np.random.rand(10, 8)
    predictions = lstm_model.predict(X_test)
    
    print("Predictions shape:", predictions.shape)
    print("First few predictions:", predictions[:5])
    
    # Evaluate the model
    evaluation = lstm_model.evaluate()
    print("Evaluation results:", evaluation)