import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import datetime as dt

# Fetch Indian stock data (example: TCS.NS)
symbol = "TCS.NS"
data = yf.download(symbol, start="2018-01-01", end=dt.datetime.now().strftime("%Y-%m-%d"))

# Prepare data
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)
X = data[['Close']].values
y = data['Target'].values

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], 1, 1))
X_test = X_test.reshape((X_test.shape[0], 1, 1))

# Build model
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(1, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

# Predict and save
pred = model.predict(X_test)
output = pd.DataFrame({
    "Date": data.index[train_size:],
    "Actual": y_test,
    "Predicted": pred.flatten()
})
output.to_csv("../outputs/forecast_output.csv", index=False)
print("Forecast saved to outputs/forecast_output.csv")
