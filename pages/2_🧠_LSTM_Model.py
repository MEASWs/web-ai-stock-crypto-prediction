import streamlit as st

st.set_page_config(page_title="LSTM Model", page_icon="🧠", layout="wide")

st.title("LSTM Model (Neural Network)")
st.markdown("---")

# ── แนวทางการพัฒนา ──
st.header("แนวทางการพัฒนา")
st.markdown("""
เลือกใช้ **LSTM (Long Short-Term Memory)** ซึ่งเป็น Neural Network ประเภท Recurrent Neural Network (RNN) 
ที่ออกแบบมาเพื่อจัดการกับข้อมูลแบบ **Time Series** โดยเฉพาะ เพราะสามารถจดจำ Pattern 
ในข้อมูลที่เป็นลำดับเวลาได้ดี
""")

st.markdown("---")

# ── ทฤษฎีและอัลกอริทึม ──
st.header("ทฤษฎีและอัลกอริทึมที่ใช้")

st.subheader("LSTM (Long Short-Term Memory)")
st.markdown("""
LSTM เป็น RNN แบบพิเศษที่แก้ปัญหา **Vanishing Gradient** ของ RNN ธรรมดา 
โดยมี **Gate** 3 ตัวควบคุมการไหลของข้อมูล:

| Gate | หน้าที่ |
|------|---------|
| **Forget Gate** | ตัดสินใจว่าจะ "ลืม" ข้อมูลเก่าส่วนไหน |
| **Input Gate** | ตัดสินใจว่าจะ "จำ" ข้อมูลใหม่ส่วนไหน |
| **Output Gate** | ตัดสินใจว่าจะ "ส่งออก" ข้อมูลส่วนไหนไปยัง time step ถัดไป |

**ข้อดีของ LSTM**:
- จับ Long-term Dependencies ได้ (จำข้อมูลย้อนหลังได้ไกล)
- เหมาะกับข้อมูล Time Series เช่น ราคาหุ้น
- จัดการกับข้อมูลที่มีรูปแบบซับซ้อนได้ดี
""")

st.markdown("---")

# ── สถาปัตยกรรมโมเดล ──
st.header("สถาปัตยกรรมโมเดล")
st.markdown("""
โมเดลที่ออกแบบมีโครงสร้างดังนี้:

| Layer | รายละเอียด |
|-------|-----------|
| **LSTM (64 units)** | `return_sequences=True` — ส่ง output ทุก time step ไปยัง layer ถัดไป |
| **Dropout (0.2)** | ปิด 20% ของ neurons เพื่อป้องกัน Overfitting |
| **LSTM (32 units)** | ส่ง output เฉพาะ time step สุดท้าย |
| **Dense (1 unit)** | Output layer — ทำนายราคาปิดวันถัดไป |
""")

st.code("""
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(10, 6)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
""", language="python")

st.markdown("---")

# ── การเตรียมข้อมูล ──
st.header("การเตรียมข้อมูล")
st.markdown("""
1. โหลดข้อมูลที่ผ่าน Preprocess แล้ว
2. เลือก Features: `Close, Volume, MA_7, MA_14, RSI, MACD` (6 features)
3. **Scale ข้อมูล** ด้วย **MinMaxScaler** (ทั้ง X และ y) ให้อยู่ในช่วง [0, 1]
4. แบ่งข้อมูล **80% Train / 20% Test**
5. **สร้าง Sequences** ด้วย Sliding Window ขนาด **10 time steps**
   - input: ข้อมูล 10 วันย้อนหลัง → output: ราคาวันที่ 11
""")

st.code("""
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)
""", language="python")

st.markdown("---")

# ── ขั้นตอนการพัฒนาโมเดล ──
st.header("ขั้นตอนการพัฒนาโมเดล")
st.markdown("""
1. **Scale ข้อมูล** ด้วย MinMaxScaler (แยก scaler สำหรับ features และ target)
2. **สร้าง Sequences** ขนาด 10 time steps (ใช้ 10 วันทำนาย 1 วัน)
3. **สร้างโมเดล LSTM** ตามสถาปัตยกรรมด้านบน
4. **Train** ด้วย:
   - `epochs=50`, `batch_size=32`
   - **EarlyStopping**: หยุด train เมื่อ `val_loss` ไม่ลดลง 5 epochs ติดต่อกัน
   - `validation_split=0.1` (แบ่ง 10% จาก training set เป็น validation)
5. **ทำนาย** บน Test Set แล้ว **Inverse Transform** กลับเป็นราคาจริง
6. **วัดผล** ด้วย RMSE และ MAE
7. **บันทึกโมเดล** เป็นไฟล์ `.h5` พร้อม scaler (`.pkl`)
""")

st.markdown("---")

# ── แหล่งอ้างอิง ──
st.header("แหล่งอ้างอิง")
st.markdown("""
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Understanding LSTM Networks — colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Yahoo Finance](https://finance.yahoo.com/) — แหล่งข้อมูลราคาหุ้นและคริปโต
- [scikit-learn: MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
""")
