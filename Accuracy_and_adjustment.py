import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from functools import reduce

data = pd.read_excel("C:/Users/adrie/OneDrive/Documents/PA_10/PJE_CFM/input_training/Input_action_0.xlsx")

input_columns = [f'r{i}' for i in range(40)]  # r0 to r39
output_columns = [f'r{i}' for i in range(40, 53)]  # r40 to r52

X = data[input_columns].values
y = data[output_columns].values

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


accuracy_list = []
acc = 0


for _ in range(50):

    model = Sequential([
        LSTM(100, activation='relu', input_shape=(40, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(100, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(13)
    ])

    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32, callbacks=[early_stopping])

    model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)

    y_test_real = scaler.inverse_transform(y_test)
    y_pred_real = scaler.inverse_transform(y_pred)

    last_day_pred = model.predict(X_test[-1].reshape(1, 40, 1))
    last_day_pred = scaler.inverse_transform(last_day_pred)

    last_day_actual = y_test_real[-1]

    columns = [f'r{i}' for i in range(40, 53)]


    reod_reel_list = []
    reod_pred_list = []


    for i in range(len(y_test_real)):

        prod_reel = [(1 + y_test_real[i][j]*10**(-4)) for j in range(len(y_test_real[i]))]
        prod_reel = reduce(lambda x, y: x * y, prod_reel)

        reod_reel = 0
        if (prod_reel - 1)*10**4 < -25:
            reod_reel = -1
        elif (prod_reel - 1)*10**4 > 25:
            reod_reel = 1
        else:
            reod_reel = 0

        reod_reel_list.append(reod_reel)


        prod_pred = [(1 + y_pred_real[i][j]*10**(-4)) for j in range(len(y_pred_real[i]))]
        prod_pred = reduce(lambda x, y: x * y, prod_pred)

        reod_pred = 0
        if (prod_pred - 1)*10**4 < -25:
            reod_pred = -1
        elif (prod_pred - 1)*10**4 > 25:
            reod_pred = 1
        else:
            reod_pred = 0

        reod_pred_list.append(reod_pred)


    v = sum(i == j for i, j in zip(reod_reel_list, reod_pred_list))
    accuracy = v / len(reod_reel_list) * 100
    acc += accuracy
    accuracy_list.append(accuracy)

print(acc/30)
