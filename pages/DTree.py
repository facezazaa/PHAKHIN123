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
    root_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(root_dir, "Data", "cirrhosis.csv")

    df = pd.read_csv(file_path)
    st.subheader("👀 ข้อมูลตัวอย่าง")
    st.write(df.head(10))
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์ cirrhosis.csv กรุณาตรวจสอบว่าอยู่ในโฟลเดอร์ 'Data'")
    st.stop()

# -------------------------------
# กำหนด target และ features
# -------------------------------
target_col = "Status"  # คอลัมน์ผลลัพธ์
if target_col not in df.columns:
    st.error(f"⚠️ ไม่พบคอลัมน์ '{target_col}' ใน dataset")
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
# ประเมินความแม่นยำ
# -------------------------------
y_predict = dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f"🎯 ความแม่นยำของโมเดล: {score*100:.2f} %")

# -------------------------------
# ทำนายผู้ป่วยจาก dataset จริง (สุ่ม 1 แถว)
# -------------------------------
st.subheader("🔍 ตัวอย่างการทำนายจากข้อมูลจริง")

rand_row = df.sample(1, random_state=42)
st.write("ข้อมูลผู้ป่วย (ก่อนแปลง):")
st.write(rand_row)

# preprocess แถวนี้ให้ตรงกับ X
row_proc = rand_row.drop(columns=[target_col])
for col in row_proc.columns:
    if row_proc[col].dtype == "object":
        row_proc[col] = row_proc[col].astype("category").cat.codes
row_proc = row_proc.reindex(columns=X.columns, fill_value=0)

pred = dtree.predict(row_proc)[0]
st.success(f"✅ ผลการทำนาย: {pred}")

# -------------------------------
# แสดงโครงสร้าง Decision Tree
# -------------------------------
st.subheader("🌳 โครงสร้าง Decision Tree")

fig, ax = plt.subplots(figsize=(16, 10))
tree.plot_tree(
    dtree,
    feature_names=X.columns,
    class_names=[str(c) for c in dtree.classes_],
    filled=True,
    ax=ax,
)
st.pyplot(fig)
