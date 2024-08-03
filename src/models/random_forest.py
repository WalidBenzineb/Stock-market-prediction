from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, X, y):
        """
        Train the random forest model.
        
        :param X: Feature matrix
        :param y: Target vector
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        :param X: Feature matrix
        :return: Array of predictions
        """
        return self.model.predict(X)
    
    def evaluate(self):
        """
        Evaluate the model's performance.
        
        :return: Dictionary with model performance metrics
        """
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        return {
            "train_r2": train_score,
            "test_r2": test_score
        }

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate some dummy data
    X = np.random.rand(100, 8)
    y = np.random.rand(100)
    
    model = RandomForestModel()
    model.train(X, y)
    
    print("Model trained.")
    print("Evaluation metrics:", model.evaluate())
    
    # Make a prediction
    new_data = np.random.rand(1, 8)
    prediction = model.predict(new_data)
    print("Prediction:", prediction)