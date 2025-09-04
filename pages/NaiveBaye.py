from sklearn.naive_bayes import GaussianNB
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cirrhosis Survival Prediction (Naive Bayes)", layout="wide")

st.title('🧪 การคาดการณ์อัตราการรอดชีวิตของผู้ป่วยโรคตับแข็งด้วย Naive Bayes')

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
# สถิติพื้นฐาน
# -------------------------------
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(df.describe(include="all"))

# -------------------------------
# ตรวจสอบ target column
# -------------------------------
target_col = "Status"   # ✅ ตั้งชื่อ target column ให้ตรงกับ dataset ของคุณ

if target_col not in df.columns:
    st.error(f"⚠️ ไม่พบคอลัมน์ '{target_col}' ในไฟล์ CSV กรุณาตรวจสอบชื่อคอลัมน์อีกครั้ง")
    st.stop()

# -------------------------------
# เลือกฟีเจอร์และแสดงกราฟ
# -------------------------------
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", [c for c in df.columns if c != target_col])

st.write(f"### 🎯 Boxplot: {feature} เทียบกับสถานะผู้ป่วย")
fig, ax = plt.subplots()
sns.boxplot(data=df, x=target_col, y=feature, ax=ax)
st.pyplot(fig)

if st.checkbox("✅ แสดง Pairplot (ใช้เวลาประมวลผล)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(df, hue=target_col)
    st.pyplot(fig2.fig)

# -------------------------------
# Preprocess
# -------------------------------
def preprocess(df):
    df2 = df.copy()

    # จัดการ categorical ก่อน
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

df_proc = preprocess(df)

X = df_proc.drop(target_col, axis=1)
y = df_proc[target_col]

# -------------------------------
# Train Model (Naive Bayes)
# -------------------------------
model = GaussianNB()
model.fit(X, y)

# -------------------------------
# เลือกข้อมูลตัวอย่างจาก dataset
# -------------------------------
st.subheader("📝 ตัวอย่างข้อมูลผู้ป่วยจากไฟล์เพื่อทำนาย")
rand_row = df.sample(1)   # ✅ random จริงทุกครั้ง
st.write("ข้อมูลจริงจากไฟล์ (ก่อนแปลง):")
st.write(rand_row)

x_input_proc = preprocess(rand_row)
x_input_proc = x_input_proc.reindex(columns=X.columns, fill_value=0)
x_input_proc = x_input_proc.fillna(0)

# -------------------------------
# ทำนายผล
# -------------------------------
st.subheader("🔍 ผลการทำนาย")
prediction = model.predict(x_input_proc)[0]
prob = model.predict_proba(x_input_proc)[0]

if prediction == 1:   # สมมติ 1 = รอดชีวิต
    st.success(f"✅ ผู้ป่วยมีโอกาสรอดชีวิตสูง (ความมั่นใจ {prob[1]*100:.2f}%)")
    st.image("./img/12.jpg", width=300)
else:
    st.error(f"⚠️ ผู้ป่วยมีความเสี่ยงสูงต่อการเสียชีวิต (ความมั่นใจ {prob[0]*100:.2f}%)")
    st.image("./img/13.jpg", width=300)

# -------------------------------
# ความน่าจะเป็นแบบ bar chart
# -------------------------------
st.write("📊 ความน่าจะเป็นแต่ละคลาส")
fig3, ax3 = plt.subplots()
ax3.bar(["เสียชีวิต", "รอดชีวิต"], prob)
st.pyplot(fig3)
