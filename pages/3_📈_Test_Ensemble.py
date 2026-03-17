import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Test Ensemble", page_icon="📈", layout="wide")

st.title("ทดสอบโมเดล Ensemble")
st.markdown("---")

st.markdown("""
ทดสอบการทำนายราคาด้วยโมเดล **Ensemble (VotingRegressor)** 
ซึ่งรวม Random Forest, Gradient Boosting และ XGBoost เข้าด้วยกัน
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

split_index = int(len(df) * 0.8)
X_test = X[split_index:]
y_test = y[split_index:]

# ── Load Model & Predict ──
model_path = f"models/ensemble/{selected_asset}_ensemble.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.values, label="Actual", linewidth=1.5)
    ax.plot(predictions, label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price")
    ax.set_title(f"{selected_asset} — Ensemble Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ── Metrics ──
    rmse = np.sqrt(np.mean((y_test.values - predictions) ** 2))
    mae = np.mean(np.abs(y_test.values - predictions))

    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("MAE", f"{mae:.2f}")

    # ── Show sample predictions ──
    st.subheader("ตัวอย่างผลทำนาย")
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions,
        "Error": np.abs(y_test.values - predictions)
    })
    st.dataframe(results.head(20), use_container_width=True)
else:
    st.error(f"❌ ไม่พบโมเดลสำหรับ {selected_asset} กรุณา train โมเดลก่อน")
