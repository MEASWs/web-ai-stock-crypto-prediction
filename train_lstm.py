import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

processed_path = "data/processed/"
files = os.listdir(processed_path)

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

for file in files:
    if file.endswith(".csv"):
        print(f"\nTraining LSTM for {file}...")

        df = pd.read_csv(os.path.join(processed_path, file))

        features = ['Close', 'Volume', 'MA_7', 'MA_14', 'RSI', 'MACD']
        target = 'Target'

        X = df[features].values
        y = df[target].values.reshape(-1, 1)

        # Scale ทั้ง X และ y
        feature_scaler = MinMaxScaler()
        X_scaled = feature_scaler.fit_transform(X)

        y_scaler = MinMaxScaler()
        y_scaled = y_scaler.fit_transform(y)

        split_index = int(len(X_scaled) * 0.8)

        X_train_raw = X_scaled[:split_index]
        X_test_raw = X_scaled[split_index:]
        y_train_raw = y_scaled[:split_index]
        y_test_raw = y_scaled[split_index:]

        time_steps = 10

        X_train, y_train = create_sequences(X_train_raw, y_train_raw.flatten(), time_steps)
        X_test, y_test = create_sequences(X_test_raw, y_test_raw.flatten(), time_steps)

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )

        predictions_scaled = model.predict(X_test)

        # แปลงกลับเป็นราคาจริง
        predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        mae = mean_absolute_error(y_test_actual, predictions)

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")

        os.makedirs("models/lstm", exist_ok=True)
        model.save(f"models/lstm/{file.replace('.csv','')}_lstm.h5")

        # บันทึก scaler เพื่อใช้ใน app.py
        joblib.dump(feature_scaler, f"models/lstm/{file.replace('.csv','')}_feature_scaler.pkl")
        joblib.dump(y_scaler, f"models/lstm/{file.replace('.csv','')}_y_scaler.pkl")

print("\nAll LSTM Models Trained Successfully!")