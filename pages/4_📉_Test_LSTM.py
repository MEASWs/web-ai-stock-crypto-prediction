import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Test LSTM", page_icon="📉", layout="wide")

st.title("📉 ทดสอบโมเดล LSTM")
st.markdown("---")

st.markdown("""
ทดสอบการทำนายราคาด้วยโมเดล **LSTM (Long Short-Term Memory)** 
ซึ่งเป็น Neural Network ที่ออกแบบมาสำหรับข้อมูล Time Series โดยเฉพาะ
""")

# ── Asset Selection ──
all_assets = [f.replace(".csv", "") for f in os.listdir("data/processed") if f.endswith(".csv")]
stocks = [a for a in all_assets if not a.endswith("-USD")]
cryptos = [a for a in all_assets if a.endswith("-USD")]

asset_type = st.selectbox("เลือกประเภท", ["Stock", "Crypto"])
if asset_type == "Stock":
    selected_asset = st.selectbox("เลือกสินทรัพย์", stocks)
else:
    selected_asset = st.selectbox("เลือกสินทรัพย์", cryptos)

# ── Load Data ──
df = pd.read_csv(f"data/processed/{selected_asset}.csv")

features = ['Close', 'Volume', 'MA_7', 'MA_14', 'RSI', 'MACD']
target = 'Target'

X = df[features]
y = df[target]

# ── Sequence helper ──
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# ── Load Model & Predict ──
model_path = f"models/lstm/{selected_asset}_lstm.h5"
feature_scaler_path = f"models/lstm/{selected_asset}_feature_scaler.pkl"
y_scaler_path = f"models/lstm/{selected_asset}_y_scaler.pkl"

if os.path.exists(model_path) and os.path.exists(feature_scaler_path) and os.path.exists(y_scaler_path):
    model = load_model(model_path, compile=False)

    feature_scaler = joblib.load(feature_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    X_scaled = feature_scaler.transform(X)

    split_index = int(len(X_scaled) * 0.8)
    X_test_raw = X_scaled[split_index:]

    y_scaled = y_scaler.transform(y.values.reshape(-1, 1)).flatten()
    y_test_raw = y_scaled[split_index:]

    time_steps = 10
    X_test_seq, y_test_seq = create_sequences(X_test_raw, y_test_raw, time_steps)

    predictions_scaled = model.predict(X_test_seq)

    predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    y_test_actual = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    st.subheader("📉 Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test_actual, label="Actual", linewidth=1.5)
    ax.plot(predictions, label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price")
    ax.set_title(f"{selected_asset} — LSTM Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ── Metrics ──
    rmse = np.sqrt(np.mean((y_test_actual - predictions) ** 2))
    mae = np.mean(np.abs(y_test_actual - predictions))

    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("MAE", f"{mae:.2f}")

    # ── Show sample predictions ──
    st.subheader("📋 ตัวอย่างผลทำนาย")
    results = pd.DataFrame({
        "Actual": y_test_actual,
        "Predicted": predictions,
        "Error": np.abs(y_test_actual - predictions)
    })
    st.dataframe(results.head(20), use_container_width=True)
else:
    st.error(f"❌ ไม่พบโมเดลหรือ scaler สำหรับ {selected_asset} กรุณา train โมเดลก่อน")
