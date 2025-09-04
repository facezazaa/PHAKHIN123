import os
import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.header("🌳 Decision Tree สำหรับการทำนายโรคตับแข็ง")

# -------------------------------
# โหลดข้อมูล
# -------------------------------
try:
    # หา path ของโฟลเดอร์หลัก (parent ของ pages/)
    root_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(root_dir, "Data", "cirrhosis.csv")

    df = pd.read_csv(file_path)
    st.subheader("👀 ข้อมูลตัวอย่าง")
    st.write(df.head(10))
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์ cirrhosis.csv กรุณาตรวจสอบว่าอยู่ในโฟลเดอร์ 'Data' ข้างๆ project")
    st.stop()

# -------------------------------
# กำหนด target และ features
# -------------------------------
target_col = "Status"  # สมมติว่าคอลัมน์นี้คือผลลัพธ์ (0=เสียชีวิต, 1=รอดชีวิต)
if target_col not in df.columns:
    st.error(f"⚠️ ไม่พบคอลัมน์ '{target_col}' ใน dataset กรุณาตรวจสอบไฟล์ cirrhosis.csv")
    st.stop()

X = df.drop(target_col, axis=1)
y = df[target_col]

# -------------------------------
# จัดการ Missing values
# -------------------------------
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("Unknown").astype("category").cat.codes
    else:
        X[col] = X[col].fillna(X[col].mean())

# -------------------------------
# แบ่งข้อมูล train / test
# -------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=200
)

# -------------------------------
# สร้างโมเดล Decision Tree
# -------------------------------
model = DecisionTreeClassifier(random_state=42, max_depth=4)
dtree = model.fit(x_train, y_train)

# -------------------------------
# Input form จากผู้ใช้
# -------------------------------
st.subheader("📝 กรอกข้อมูลผู้ป่วยเพื่อพยากรณ์")

user_input = {}
for col in X.columns:
    if X[col].dtype in ["int64", "float64"]:
        user_input[col] = st.number_input(
            f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean())
        )
    else:
        options = list(df[col].astype("category").cat.categories)
        user_input[col] = st.selectbox(f"{col}", options)

# -------------------------------
# พยากรณ์จาก input ผู้ใช้
# -------------------------------
if st.button("พยากรณ์"):
    input_df = pd.DataFrame([user_input])

    # preprocess ให้ตรงกับ X
    for col in input_df.columns:
        if input_df[col].dtype == "object":
            input_df[col] = input_df[col].astype("category").cat.codes
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    y_pred = dtree.predict(input_df)[0]
    labels = {0: "เสียชีวิต", 1: "รอดชีวิต"}
    st.success(f"🔮 ผลการพยากรณ์: {labels[y_pred]}")

# -------------------------------
# ความแม่นยำของโมเดล
# -------------------------------
y_predict = dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f"🎯 ความแม่นยำของโมเดล: {score*100:.2f} %")

# -------------------------------
# แสดงกราฟโครงสร้าง Decision Tree
# -------------------------------
fig, ax = plt.subplots(figsize=(16, 10))
tree.plot_tree(
    dtree,
    feature_names=X.columns,
    class_names=["เสียชีวิต", "รอดชีวิต"],
    filled=True,
    ax=ax,
)
st.pyplot(fig)
