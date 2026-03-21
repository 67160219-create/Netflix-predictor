import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib

print("⏳ กำลังโหลดและเตรียมข้อมูล...")
# โหลดข้อมูล
df = pd.read_csv('tv_shows.csv')

# คลีนข้อมูล
df['Age'] = df['Age'].fillna('Unknown')
df['IMDb'] = df['IMDb'].astype(str).str.replace('/10', '').replace('nan', np.nan).astype(float)
df['IMDb'] = df['IMDb'].fillna(df['IMDb'].median())
df['Rotten Tomatoes'] = df['Rotten Tomatoes'].astype(str).str.replace('/100', '').replace('nan', np.nan).astype(float)
df['Rotten Tomatoes'] = df['Rotten Tomatoes'].fillna(df['Rotten Tomatoes'].median())

# แยก X, y
X = df[['Year', 'Age', 'IMDb', 'Rotten Tomatoes']]
y = df['Netflix']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# เตรียม Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Year', 'IMDb', 'Rotten Tomatoes']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Age'])
    ])

# รายชื่อโมเดลทั้ง 4 ตัว และชื่อไฟล์ที่จะเซฟ (ให้ตรงกับ app.py)
models_to_train = {
    'model_pipeline.pkl': RandomForestClassifier(n_estimators=100, random_state=42),
    'model_gb.pkl': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'model_svm.pkl': SVC(probability=True, random_state=42),
    'model_lr.pkl': LogisticRegression(random_state=42)
}

print("\n🚀 กำลัง Train และ Save โมเดลทั้ง 4 ตัว ด้วยเครื่อง Local ของคุณ...")

for filename, model in models_to_train.items():
    # มัดรวม Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Save ทับไฟล์เดิมด้วยเวอร์ชันของเครื่องนี้
    joblib.dump(pipeline, filename)
    print(f"✅ สร้างไฟล์ {filename} เสร็จเรียบร้อยแล้ว!")

print("\n🎉 สร้างโมเดลใหม่ครบทั้ง 4 ตัวแล้ว! กลับไปรัน Streamlit ได้เลยครับ")