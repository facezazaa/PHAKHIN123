from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Cirrhosis Survival Prediction", layout="wide")

st.title('🔮 การคาดการณ์อัตราการรอดชีวิตของผู้ป่วยโรคตับแข็งด้วย K-Nearest Neighbor')

# -------------------------------
# โหลดข้อมูล (พร้อม debug path)
# -------------------------------
@st.cache_data
def load_data():
    # หา path ของ root project
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root_dir, "data", "cirrhosis.csv")

    # แสดง path ที่โปรแกรมจะพยายามอ่าน
    st.write(f"📂 กำลังโหลดไฟล์จาก: {file_path}")

    # อ่าน CSV
    return pd.read_csv(file_path)

# โหลดข้อมูล
try:
    dt = load_data()
except FileNotFoundError as e:
    st.error("❌ ไม่พบไฟล์ cirrhosis.csv กรุณาตรวจสอบตำแหน่งไฟล์อีกครั้ง")
    st.stop()

# -------------------------------
# แสดงตัวอย่างข้อมูล
# -------------------------------
st.subheader("👀 ข้อมูลตัวอย่าง")
col1, col2 = st.columns(2)
with col1:
    st.write("ข้อมูลส่วนแรก 10 แถว")
    st.write(dt.head(10))
with col2:
    st.write("ข้อมูลส่วนสุดท้าย 10 แถว")
    st.write(dt.tail(10))

# -------------------------------
# สถิติพื้นฐาน
# -------------------------------
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe(include="all"))

# -------------------------------
# เลือกฟีเจอร์และแสดงกราฟ
# -------------------------------
target_col = "Status"   # สมมติว่าคอลัมน์นี้บอกผลลัพธ์ (0=เสียชีวิต, 1=รอดชีวิต)

if target_col not in dt.columns:
    st.error(f"⚠️ ไม่พบคอลัมน์ '{target_col}' ในไฟล์ CSV ของคุณ กรุณาตรวจสอบอีกครั้ง")
    st.stop()

st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", [c for c in dt.columns if c != target_col])

st.write(f"### 🎯 Boxplot: {feature} เทียบกับสถานะผู้ป่วย")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x=target_col, y=feature, ax=ax)
st.pyplot(fig)

# -------------------------------
# Preprocess
# -------------------------------
def preprocess(df):
    df2 = df.copy()

    # จัดการ categorical
    for col in df2.columns:
        if df2[col].dtype == "object":
            df2[col] = df2[col].astype("category")

    # เติมค่า missing
    for col in df2.columns:
        if str(df2[col].dtype) == "category":
            df2[col] = df2[col].cat.add_categories("Unknown").fillna("Unknown")
            df2[col] = df2[col].cat.codes
        else:
            df2[col] = df2[col].fillna(df2[col].mean())

    return df2

dt_proc = preprocess(dt)

X = dt_proc.drop(target_col, axis=1)
y = dt_proc[target_col]

# -------------------------------
# Train Model
# -------------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# -------------------------------
# เลือกข้อมูลตัวอย่างจาก dataset
# -------------------------------
st.subheader("📝 ตัวอย่างข้อมูลผู้ป่วยจากไฟล์เพื่อทำนาย")
rand_row = dt.sample(1, random_state=42)  # สุ่ม 1 แถว
st.write("ข้อมูลจริงจากไฟล์ (ก่อนแปลง):")
st.write(rand_row)

x_input_proc = preprocess(rand_row)
x_input_proc = x_input_proc.reindex(columns=X.columns, fill_value=0)
x_input_proc = x_input_proc.fillna(0)   # กัน NaN ตกค้าง

# -------------------------------
# ทำนายผล
# -------------------------------
st.subheader("🔍 ผลการทำนาย")
prediction = model.predict(x_input_proc)[0]
prob = model.predict_proba(x_input_proc)[0]

if prediction == 1:   # สมมติ 1 = รอดชีวิต
    st.success(f"✅ ผู้ป่วยมีโอกาสรอดชีวิตสูง (ความมั่นใจ {prob[1]*100:.2f}%)")
else:
    st.error(f"⚠️ ผู้ป่วยมีความเสี่ยงสูงต่อการเสียชีวิต (ความมั่นใจ {prob[0]*100:.2f}%)")

