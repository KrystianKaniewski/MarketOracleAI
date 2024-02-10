import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# def get_stock_data(symbol, start_date, end_date):
#     """
#     Pobiera dane giełdowe dla danego symbolu z zakresu dat.
#     """
#     ticker = yf.Ticker(symbol)
#     print(ticker.history_metadata['currency'])
#     stock_data = ticker.history(start=start_date, end=end_date)
#     return stock_data


# def prepare_data(stock_data, look_back=60):
#     """
#     Przygotowuje dane do modelu LSTM.
#     """
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
#
#     x_train = []
#     y_train = []
#     for i in range(look_back, len(scaled_data)):
#         x_train.append(scaled_data[i - look_back:i, 0])
#         y_train.append(scaled_data[i, 0])
#     x_train, y_train = np.array(x_train), np.array(y_train)
#
#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#
#     return x_train, y_train, scaler
def prepare_data(stock_data, look_back):
    """
    Przygotowuje dane do modelu LSTM.
    """
    # Zapisanie oryginalnych wartości danych treningowych
    original_data = stock_data['Close'].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(original_data.reshape(-1, 1))

    x_train = []
    y_train = []
    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i - look_back:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler, original_data


def create_lstm_model(look_back):
    """
    Tworzy model LSTM.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def plot_stock_data(stock_data, predictions, scaler, ticker):
    """
    Przedstawia dane giełdowe na wykresie wraz z prognozami.
    """
    #actual_prices = scaler.transform(stock_data['Close'].values.reshape(-1, 1))
    actual_prices = stock_data['Close']
    plt.figure(figsize=(10, 6))
    actual_prices = actual_prices[-21:]
    plt.plot(stock_data.index[-21:], actual_prices, label='Actual Prices')

    future_dates = [stock_data.index[-1] + datetime.timedelta(days=i) for i in range(0, len(predictions))]
    # Odwrócenie transformacji przewidywanych danych
    #predictions_unscaled = scaler.inverse_transform(predictions.values.reshape(-1, 1))
    predictions_unscaled = predictions
    plt.plot(future_dates, predictions_unscaled, label='Predicted Prices')

    plt.title('Market Oracle')
    plt.xlabel('Date')
    plt.ylabel('Price ' + ticker.history_metadata['currency'])
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Pobranie danych giełdowych dla danego symbolu z zakresu ostatnich dwóch lat
    stock_symbol = 'CDR.WA'
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=2 * 365)  # Zakres ostatnich dwóch lat

    ticker = yf.Ticker(stock_symbol)
    stock_data = ticker.history(start=start_date, end=end_date)

    # Przygotowanie danych do modelu LSTM
    look_back = 365  # Liczba dni, które będą brane pod uwagę przy przewidywaniu
    x_train, y_train, scaler, original_data = prepare_data(stock_data, look_back)

    # Stworzenie i trenowanie modelu LSTM
    model = create_lstm_model(look_back)
    model.fit(x_train, y_train, epochs=100, batch_size=64)

    # Przewidywanie cen na kolejne dni
    last_window = original_data[-look_back:].reshape(1, -1)
    x_input = np.array(last_window).reshape(1, look_back, 1)
    predictions = []
    predictions.append([stock_data['Close'][-1]])

    for i in range(1, 8):  # Przewidywanie na kolejne 7 dni
        pred = model.predict(x_input)
        predictions.append(pred[0])
        # Aktualizacja danych wejściowych na podstawie przewidywań
        x_input = np.append(x_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)  # Usunięcie pierwszej wartości z danych wejściowych

    # Wyświetlenie wyników
    plot_stock_data(stock_data, predictions, scaler, ticker)


