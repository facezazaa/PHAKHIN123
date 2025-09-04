from sklearn.naive_bayes import GaussianNB
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cirrhosis Survival Prediction - Naive Bayes", layout="wide")

st.title('🧠 การคาดการณ์อัตราการรอดชีวิตของผู้ป่วยโรคตับแข็งด้วย Naive Bayes')

# -------------------------------
# โหลดข้อมูล
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("./Data/cirrhosis.csv")

df = load_data()

st.subheader("👀 ข้อมูลตัวอย่าง")
col1, col2 = st.columns(2)
with col1:
    st.write("ข้อมูลส่วนแรก 10 แถว")
    st.write(df.head(10))
with col2:
    st.write("ข้อมูลส่วนสุดท้าย 10 แถว")
    st.write(df.tail(10))

# -------------------------------
# ตรวจสอบคอลัมน์เป้าหมาย
# -------------------------------
target_col = "Status"   # 👈 แก้ตรงนี้ให้ตรงกับชื่อคอลัมน์ target ของคุณ

if target_col not in df.columns:
    st.error(f"⚠️ ไม่พบคอลัมน์ '{target_col}' กรุณาตรวจสอบชื่อคอลัมน์อีกครั้ง")
    st.write("📌 คอลัมน์ทั้งหมด:", df.columns.tolist())
    st.stop()

# -------------------------------
# สถิติพื้นฐาน
# -------------------------------
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(df.describe(include="all"))

# -------------------------------
# เลือกฟีเจอร์และแสดงกราฟ
# -------------------------------
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", [c for c in df.columns if c != target_col])

st.write(f"### 🎯 Boxplot: {feature} เทียบกับสถานะผู้ป่วย")
fig, ax = plt.subplots()
sns.boxplot(data=df,
)