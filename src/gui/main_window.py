from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from src.evaluation.model_evaluation import evaluate_models
from src.prediction.future_prediction import predict_future

class PredictionThread(QThread):
    finished = pyqtSignal(object, object, object, str)

    def __init__(self, model, model_name, X, y, scaler):
        QThread.__init__(self)
        self.model = model
        self.model_name = model_name
        self.X = X
        self.y = y
        self.scaler = scaler

    def run(self):
        results = evaluate_models({self.model_name: self.model}, self.X, self.y, self.scaler)
        
        # Prepare last_known_data
        last_x = self.X[-1].reshape(1, -1)  # Ensure it's 2D
        last_y = self.y[-1].reshape(1, -1)  # Ensure it's 2D
        last_known_data = np.column_stack((last_x, last_y))
        
        print(f"Debug: last_known_data shape: {last_known_data.shape}")
        print(f"Debug: last_known_data: {last_known_data}")
        
        future_prices = predict_future(self.model, last_known_data, self.scaler)
        self.finished.emit(results, future_prices, self.scaler, self.model_name)

class StockPredictionGUI(QWidget):
    def __init__(self, stocks, models, cleaned_data):
        super().__init__()
        self.stocks = stocks
        self.models = models
        self.cleaned_data = cleaned_data
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Stock selection
        stock_layout = QHBoxLayout()
        stock_label = QLabel('Select Stock:')
        self.stock_combo = QComboBox()
        self.stock_combo.addItems(list(self.stocks.keys()))
        stock_layout.addWidget(stock_label)
        stock_layout.addWidget(self.stock_combo)
        layout.addLayout(stock_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel('Select Model:')
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.models.keys()))
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # Predict button
        self.predict_button = QPushButton('Predict')
        self.predict_button.clicked.connect(self.on_predict)
        layout.addWidget(self.predict_button)

        # Matplotlib Figure
        self.figure = plt.figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.setWindowTitle('Stock Price Prediction')
        self.show()

    def on_predict(self):
        self.predict_button.setEnabled(False)
        stock = self.stock_combo.currentText()
        model_name = self.model_combo.currentText()
        
        X, y, scaler = self.cleaned_data[stock]
        model = self.models[model_name]
        
        self.thread = PredictionThread(model, model_name, X, y, scaler)
        self.thread.finished.connect(self.on_prediction_finished)
        self.thread.start()

    def on_prediction_finished(self, results, future_prices, scaler, model_name):
        self.predict_button.setEnabled(True)
        stock = self.stock_combo.currentText()
        X, y, _ = self.cleaned_data[stock]
        self.update_plot(stock, model_name, results[model_name]['predictions'], y, future_prices, scaler, X)

    def update_plot(self, stock, model, predictions, actual, future_prices, scaler, X):
        self.figure.clear()
        
        ax = self.figure.add_subplot(111)
        
        # Inverse transform actual prices
        X_y_actual = np.column_stack((X, actual))
        X_y_actual_inv = scaler.inverse_transform(X_y_actual)
        actual_inv = X_y_actual_inv[:, -1]
        
        ax.plot(actual_inv, label='Actual Prices', linewidth=1)
        ax.plot(predictions, label='Predicted Prices', linewidth=1)
        ax.plot(range(len(actual), len(actual) + len(future_prices)), future_prices, label='Future Predictions', linewidth=1)
        
        ax.set_title(f'{stock} Stock Price - {model} Model')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()

        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # This is just for demonstration. In a real application, you would load your actual data and models here.
    stocks = {'GOOGL': None, 'AAPL': None, 'MSFT': None}
    models = {'Linear Regression': None, 'Random Forest': None, 'LSTM': None}
    cleaned_data = {'GOOGL': (None, None, None), 'AAPL': (None, None, None), 'MSFT': (None, None, None)}
    
    ex = StockPredictionGUI(stocks, models, cleaned_data)
    sys.exit(app.exec_())