# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Download historical stock data for Apple Inc. (AAPL)
stock_data = yf.download('AAPL', start='2010-01-01', end='2024-01-01')

# Calculate moving averages
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()

# Create target variable: 1 for buy signal, -1 for sell signal
stock_data['Signal'] = 0
stock_data.loc[stock_data['SMA_50'] > stock_data['SMA_200'], 'Signal'] = 1
stock_data.loc[stock_data['SMA_50'] < stock_data['SMA_200'], 'Signal'] = -1

# Drop rows with NaN values
stock_data.dropna(inplace=True)

# Features and target
features = stock_data[['SMA_50', 'SMA_200']]
target = stock_data['Signal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')

# Initialize and train the Support Vector Classifier
svc_model = SVC(kernel='rbf')
svc_model.fit(X_train, y_train)

# Predict on the test set
svc_pred = svc_model.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_pred)
print(f'SVC Accuracy: {svc_accuracy:.2f}')

# Initialize and train the Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predict on the test set
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f'Logistic Regression Accuracy: {lr_accuracy:.2f}')

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['Close']])

# Create dataset for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10
X, Y = create_dataset(scaled_data, look_back)

# Split into training and test sets for LSTM
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
X_train_lstm, X_test_lstm = X[0:train_size], X[train_size:len(X)]
Y_train_lstm, Y_test_lstm = Y[0:train_size], Y[train_size:len(Y)]

# Reshape input to be [samples, time steps, features]
X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], 1, X_train_lstm.shape[1]))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], 1, X_test_lstm.shape[1]))

# Create and fit the LSTM network
lstm_model = Sequential()
lstm_model.add(LSTM(4, input_shape=(1, look_back)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train_lstm, Y_train_lstm, epochs=100, batch_size=1, verbose=0)  # Set verbose to 0 to disable epoch output

# Make predictions with LSTM
lstm_train_predict = lstm_model.predict(X_train_lstm)
lstm_test_predict = lstm_model.predict(X_test_lstm)

# Invert predictions
lstm_train_predict = scaler.inverse_transform(lstm_train_predict)
Y_train_lstm = scaler.inverse_transform([Y_train_lstm])
lstm_test_predict = scaler.inverse_transform(lstm_test_predict)
Y_test_lstm = scaler.inverse_transform([Y_test_lstm])

# Calculate root mean squared error
train_score = np.sqrt(np.mean((lstm_train_predict - Y_train_lstm) ** 2))
print(f'LSTM Train Score: {train_score:.2f} RMSE')
test_score = np.sqrt(np.mean((lstm_test_predict - Y_test_lstm) ** 2))
print(f'LSTM Test Score: {test_score:.2f} RMSE')

# Use the best model (RandomForestClassifier) to make predictions on the full dataset
stock_data['Predicted_Signal'] = rf_model.predict(features)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(stock_data.index, stock_data['Close'], label='Close Price')
plt.plot(stock_data.index, stock_data['SMA_50'], label='50-day SMA')
plt.plot(stock_data.index, stock_data['SMA_200'], label='200-day SMA')

# Plot buy signals
buy_signals = stock_data[stock_data['Predicted_Signal'] == 1]
plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal')

# Plot sell signals
sell_signals = stock_data[stock_data['Predicted_Signal'] == -1]
plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal')

plt.legend()
plt.title('Stock Price and Trading Signals')
plt.show()
