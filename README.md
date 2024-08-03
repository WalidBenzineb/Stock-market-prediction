# Stock Price Prediction Application

## Project Overview

This Stock Price Prediction Application demonstrates the implementation of machine learning models for financial forecasting. The project showcases skills in data processing, model development, and GUI creation, applied to the challenge of predicting stock prices.

![google-linear](https://github.com/user-attachments/assets/efe198fc-1acc-4ca4-b8af-ae069b6ef9c9)
![aapl-random](https://github.com/user-attachments/assets/b800b4bd-f487-4457-ad0a-cbff99f57db7)
![MSFT-LSTM](https://github.com/user-attachments/assets/61cc664d-51f7-4249-8869-9c762d6bacff)


## Key Features

- Data acquisition and preprocessing from Yahoo Finance
- Implementation of multiple prediction models:
  - Linear Regression
  - Random Forest
  - Long Short-Term Memory (LSTM) Neural Network
- Interactive GUI for model selection and result visualization
- Historical data analysis and future price prediction capabilities

## Technologies Utilized

- Python 3.8
- PyQt5 for GUI development
- scikit-learn for traditional machine learning models
- TensorFlow for deep learning (LSTM) implementation
- yfinance for financial data acquisition
- Matplotlib for data visualization
- NumPy and Pandas for data manipulation

## Implementation Details

### Data Acquisition and Preprocessing

The data pipeline is implemented in `src/data/data_acquisition.py` and `src/data/data_cleaning.py`:

```python
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def preprocess_data(data):
    data = data.dropna()
    scaler = MinMaxScaler()
    data['Normalized_Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return data, scaler
```
This approach ensures efficient data retrieval and standardized preprocessing, crucial for model performance.
### Model Implementation
The project implements three distinct models, each offering different approaches to time series prediction:

**Linear Regression (src/models/linear_regression.py):**
``` python
sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test
``` 
**Random Forest (src/models/random_forest.py):**
``` python
sklearn.ensemble import RandomForestRegressor

def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
``` 
**LSTM (src/models/lstm.py):**

``` python
tensorflow as tf

def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
``` 

Each model is designed to capture different aspects of the time series data, from linear trends to complex patterns.
### GUI Development
The application's interface is built using PyQt5, providing an intuitive user experience:

```python
import QApplication, QMainWindow, QPushButton, QComboBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class StockPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Prediction Application")
        self.setGeometry(100, 100, 800, 600)

        self.setup_ui()

    def setup_ui(self):
        self.stock_combo = QComboBox(self)
        self.stock_combo.addItems(["AAPL", "GOOGL", "MSFT"])
        self.stock_combo.move(50, 50)

        self.model_combo = QComboBox(self)
        self.model_combo.addItems(["Linear Regression", "Random Forest", "LSTM"])
        self.model_combo.move(50, 100)

        self.predict_button = QPushButton("Predict", self)
        self.predict_button.move(50, 150)
        self.predict_button.clicked.connect(self.on_predict)

        self.figure = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.move(200, 50)

    def on_predict(self):
        # Prediction and plotting logic
        pass
``` 
## Challenges and Solutions

Throughout the development process, several challenges were addressed:

* Data Consistency: Ensuring consistent data quality across different stocks and time periods required robust error handling and data validation mechanisms.
* Model Performance: Balancing model complexity with performance led to the implementation of multiple models, each with its strengths in different market conditions.
* Real-time Processing: Optimizing the application for responsive real-time predictions involved efficient data handling and asynchronous processing techniques.

## Potential Enhancements

Future iterations of this project could explore:

* Integration of additional financial indicators and alternative data sources
* Implementation of more advanced models, such as transformer networks
* Enhanced visualization capabilities, including interactive charts and performance metrics
* Deployment as a web application for broader accessibility

## Setup and Usage

* Clone the repository.

* Install dependencies: Copypip install -r requirements.txt

* Run the application: main.py .
