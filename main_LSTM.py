import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Upload train & test data
train_input_file = "C:/Users/adrie/OneDrive/Documents/PA_10/PJE_CFM/Dataset/input_training.xlsx"
train_output_file = "C:/Users/adrie/OneDrive/Documents/PA_10/PJE_CFM/Dataset/output_training.xlsx"
test_input_file = "C:/Users/adrie/OneDrive/Documents/PA_10/PJE_CFM/Dataset/input_test.xlsx"

train_inputs = pd.read_excel(train_input_file, sheet_name='input_training')
train_outputs = pd.read_excel(train_output_file, sheet_name='output_training')
test_inputs = pd.read_excel(test_input_file, sheet_name='input_test')

# Select the columns from 'r0' to 'r52' - historical first 4,5 hours of the day (53 x 5 minutes)
train_inputs = train_inputs[[f'r{i}' for i in range(53)]].values
train_outputs = train_outputs['Reod'].values
test_inputs = test_inputs[[f'r{i}' for i in range(53)]].values

# Compute the 3 different trends for the retruns (-1, 0, 1) in one-hot encoding
label_encoder = LabelEncoder()
train_outputs_encoded = label_encoder.fit_transform(train_outputs)
train_outputs_onehot = to_categorical(train_outputs_encoded)

# Divide the dataset betwween a train set and a validation set
X_train, X_val, y_train, y_val = train_test_split(train_inputs, train_outputs_onehot, test_size=0.2, random_state=42)

# Pretreatment of the data with Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_inputs_scaled = scaler.transform(test_inputs)

# Building the LSTM model and its different layers
model = Sequential()
model.add(Bidirectional(LSTM(units=100, activation='relu', return_sequences=True), input_shape=(X_train_scaled.shape[1], 1)))
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(units=50, activation='relu')))
model.add(Dropout(0.2))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
model.fit(np.expand_dims(X_train_scaled, axis=-1), y_train, epochs=50, batch_size=64, callbacks=[early_stopping], validation_data=(np.expand_dims(X_val_scaled, axis=-1), y_val))

# Test the prediction on the test data
test_predictions = model.predict(np.expand_dims(test_inputs_scaled, axis=-1))
test_predictions_classes = label_encoder.inverse_transform(np.argmax(test_predictions, axis=1))

# Create a dataframe with all the predictions
output_df = pd.DataFrame({'reod': test_predictions_classes})

# Save the predictions in an xlsx format
output_df.to_excel('C:/Users/adrie/OneDrive/Documents/PA_10/PJE_CFM/Dataset/Y_test_3.xlsx', index=False)
