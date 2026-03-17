import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Stock & Crypto Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("Stock & Crypto Price Prediction")
st.markdown("---")

# ── Project Overview ──
st.header("Project Overview")
st.markdown("""
โปรเจกต์นี้เป็นการพัฒนาระบบทำนายราคาหุ้นและคริปโตเคอร์เรนซี โดยใช้โมเดล **Machine Learning (Ensemble)** 
และ **Neural Network (LSTM)** เพื่อทำนายราคาปิดของวันถัดไป

**Dataset** ดาวน์โหลดจาก **Yahoo Finance** ผ่านไลบรารี `yfinance` ประกอบด้วย:
- **หุ้น 10 ตัว**: AAPL, MSFT, AMZN, TSLA, GOOGL, META, NVDA, BRK-B, JNJ, JPM
- **คริปโต 2 ตัว**: BTC-USD, ETH-USD
- **ช่วงเวลา**: 1 มกราคม 2018 – 31 ธันวาคม 2024
""")

st.markdown("---")

# ── Dataset Features ──
st.header("Features ของ Dataset")

st.subheader("Features ดั้งเดิม (จาก Yahoo Finance)")
st.markdown("""
| Feature | คำอธิบาย |
|---------|----------|
| **Date** | วันที่ของข้อมูล |
| **Open** | ราคาเปิดตลาด |
| **High** | ราคาสูงสุดในวัน |
| **Low** | ราคาต่ำสุดในวัน |
| **Close** | ราคาปิดตลาด |
| **Adj Close** | ราคาปิดที่ปรับแล้ว (หลังหักปันผล/split) |
| **Volume** | ปริมาณการซื้อขาย |
""")

st.subheader("Features ที่สร้างเพิ่ม (Feature Engineering)")
st.markdown("""
| Feature | คำอธิบาย |
|---------|----------|
| **MA_7** | Moving Average 7 วัน — ค่าเฉลี่ยราคาปิดย้อนหลัง 7 วัน ใช้ดูแนวโน้มระยะสั้น |
| **MA_14** | Moving Average 14 วัน — ค่าเฉลี่ยราคาปิดย้อนหลัง 14 วัน ใช้ดูแนวโน้มระยะกลาง |
| **RSI** | Relative Strength Index — ดัชนีวัดแรงซื้อ/ขาย (0–100) ถ้า >70 = Overbought, <30 = Oversold |
| **MACD** | Moving Average Convergence Divergence — ตัวชี้วัดโมเมนตัมของราคา ใช้ดูสัญญาณซื้อ/ขาย |
| **Target** | ราคาปิดของวันถัดไป (`Close.shift(-1)`) — ค่าที่โมเดลต้องทำนาย |
""")

st.subheader("Features ที่ใช้เข้าโมเดล")
st.code("features = ['Close', 'Volume', 'MA_7', 'MA_14', 'RSI', 'MACD']", language="python")

st.markdown("---")

# ── Data Preparation ──
st.header("การเตรียมข้อมูล")
st.markdown("""
1. **ดาวน์โหลดข้อมูล** — ใช้ `yfinance` ดึงราคาหุ้นและคริปโตจาก Yahoo Finance
2. **จัดการ Missing Values** — ใช้ Forward Fill (`ffill`) เติมค่าที่หายไป
3. **สร้าง Technical Indicators** — คำนวณ MA_7, MA_14, RSI, MACD จากราคาปิด
4. **สร้าง Target** — ใช้ราคาปิดวันถัดไป (`Close.shift(-1)`) เป็นค่าเป้าหมาย
5. **ลบแถวที่มี NaN** — Technical Indicators จะมี NaN ในช่วงแรก ถูกลบออกด้วย `dropna()`
""")

st.markdown("---")

# ── Sample Data Preview ──
st.header("ตัวอย่างข้อมูล")

all_assets = [f.replace(".csv", "") for f in os.listdir("data/processed") if f.endswith(".csv")]
stocks = [a for a in all_assets if not a.endswith("-USD")]
cryptos = [a for a in all_assets if a.endswith("-USD")]

asset_type = st.selectbox("เลือกประเภท", ["Stock", "Crypto"])
if asset_type == "Stock":
    selected = st.selectbox("เลือกสินทรัพย์", stocks)
else:
    selected = st.selectbox("เลือกสินทรัพย์", cryptos)

if selected:
    df = pd.read_csv(f"data/processed/{selected}.csv")
    st.write(f"**จำนวนข้อมูล**: {len(df)} แถว, {len(df.columns)} คอลัมน์")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("กราฟราคาปิด (Close)")
    st.line_chart(df.set_index(df.index)["Close"])

st.markdown("---")
st.markdown("ใช้เมนูด้านซ้ายเพื่อดูรายละเอียดโมเดลและทดสอบการทำนาย")