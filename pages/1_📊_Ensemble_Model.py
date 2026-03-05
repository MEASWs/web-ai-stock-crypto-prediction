import streamlit as st

st.set_page_config(page_title="Ensemble Model", page_icon="📊", layout="wide")

st.title("📊 Ensemble Model (Machine Learning)")
st.markdown("---")

# ── แนวทางการพัฒนา ──
st.header("🎯 แนวทางการพัฒนา")
st.markdown("""
เลือกใช้แนวทาง **Ensemble Learning** ซึ่งเป็นเทคนิคที่รวมโมเดลหลายตัวเข้าด้วยกัน 
เพื่อให้ได้ผลลัพธ์ที่แม่นยำและเสถียรกว่าโมเดลเดี่ยว โดยใช้ **VotingRegressor** 
รวมผลทำนายจากโมเดล 3 ตัวเข้าด้วยกัน
""")

st.markdown("---")

# ── ทฤษฎีและอัลกอริทึม ──
st.header("📚 ทฤษฎีและอัลกอริทึมที่ใช้")

st.subheader("1. Random Forest Regressor")
st.markdown("""
- เป็น **Bagging** method ที่สร้าง Decision Tree หลายต้น แล้วเฉลี่ยผลลัพธ์
- แต่ละต้นจะถูก train ด้วยข้อมูลสุ่ม (Bootstrap Sampling) และ features สุ่ม
- **ข้อดี**: ลด Overfitting, ทนต่อ Outliers, ไม่ต้อง Scale ข้อมูล
- **Parameters**: `n_estimators=100, random_state=42`
""")

st.subheader("2. Gradient Boosting Regressor")
st.markdown("""
- เป็น **Boosting** method ที่สร้าง Decision Tree ทีละต้น แต่ละต้นจะเรียนรู้จาก Error ของต้นก่อนหน้า
- ใช้ Gradient Descent ในการหาค่าที่ลด Loss Function ได้ดีที่สุด
- **ข้อดี**: แม่นยำสูง, จับ Pattern ที่ซับซ้อนได้
- **Parameters**: `n_estimators=100, random_state=42`
""")

st.subheader("3. XGBoost (Extreme Gradient Boosting)")
st.markdown("""
- เป็น Gradient Boosting ที่ถูกปรับปรุง ให้เร็วขึ้นและแม่นยำขึ้น
- เพิ่ม Regularization (L1/L2) เพื่อป้องกัน Overfitting
- ใช้ Second-order Gradient (Hessian) ทำให้ converge เร็วกว่า
- **ข้อดี**: เร็ว, แม่นยำ, จัดการ Missing Values ได้
- **Parameters**: `n_estimators=100, random_state=42`
""")

st.subheader("4. Voting Regressor (Ensemble)")
st.markdown("""
- รวมผลทำนายจาก 3 โมเดลข้างต้นด้วยการ **เฉลี่ย** (Average)
- ทำให้ผลลัพธ์มีความเสถียรมากขึ้น ลด Variance ได้ดี
""")

st.markdown("---")

# ── การเตรียมข้อมูล ──
st.header("🛠️ การเตรียมข้อมูล")
st.markdown("""
1. โหลดข้อมูลที่ผ่านการ Preprocess แล้ว (มี Technical Indicators)
2. เลือก Features: `Close, Volume, MA_7, MA_14, RSI, MACD`
3. Target: `ราคาปิดวันถัดไป`
4. แบ่งข้อมูล **80% Train / 20% Test** แบบ Time-series Split (ไม่ Shuffle)
""")

st.code("""
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
""", language="python")

st.markdown("---")

# ── ขั้นตอนการพัฒนาโมเดล ──
st.header("⚙️ ขั้นตอนการพัฒนาโมเดล")
st.markdown("""
1. **สร้างโมเดล 3 ตัว**: Random Forest, Gradient Boosting, XGBoost
2. **รวมด้วย VotingRegressor**: เฉลี่ยผลทำนายของ 3 โมเดล
3. **Train** ด้วยข้อมูล Training Set (80%)
4. **ทำนาย** บน Test Set (20%)
5. **วัดผล** ด้วย RMSE (Root Mean Squared Error) และ MAE (Mean Absolute Error)
6. **บันทึกโมเดล** เป็นไฟล์ `.pkl` ด้วย `joblib`
""")

st.code("""
ensemble = VotingRegressor([
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('gb', GradientBoostingRegressor(n_estimators=100)),
    ('xgb', XGBRegressor(n_estimators=100))
])
ensemble.fit(X_train, y_train)
""", language="python")

st.markdown("---")

# ── แหล่งอ้างอิง ──
st.header("📖 แหล่งอ้างอิง")
st.markdown("""
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- Friedman, J.H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.
- [scikit-learn: Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Yahoo Finance](https://finance.yahoo.com/) — แหล่งข้อมูลราคาหุ้นและคริปโต
""")
