import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# นำเข้าอัลกอริทึมทั้ง 4 ตัว
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

print("⏳ กำลังโหลดและเตรียมข้อมูล...")

# 1. โหลดและเตรียมข้อมูล
df = pd.read_csv('tv_shows.csv')
df['Age'] = df['Age'].fillna('Unknown')
df['IMDb'] = df['IMDb'].astype(str).str.replace('/10', '').replace('nan', np.nan).astype(float)
df['IMDb'] = df['IMDb'].fillna(df['IMDb'].median())
df['Rotten Tomatoes'] = df['Rotten Tomatoes'].astype(str).str.replace('/100', '').replace('nan', np.nan).astype(float)
df['Rotten Tomatoes'] = df['Rotten Tomatoes'].fillna(df['Rotten Tomatoes'].median())

X = df[['Year', 'Age', 'IMDb', 'Rotten Tomatoes']]
y = df['Netflix']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. สร้างตัวแปลงข้อมูล (Preprocessor)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Year', 'IMDb', 'Rotten Tomatoes']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Age'])
    ])

# 3. เตรียมโมเดลทั้ง 4 ตัวเข้าแข่งขัน
models = {
    '1. Logistic Regression': LogisticRegression(random_state=42),
    '2. Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    '3. Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    '4. Support Vector Machine (SVM)': SVC(probability=True, random_state=42)
}

print("\n🏁 เริ่มการแข่งขันของโมเดล!\n" + "-"*40)

# 4. วนลูปเทรนและทดสอบทีละตัว
results = []
for name, model in models.items():
    # มัดรวม Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict และประเมินผล
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    results.append({'Model': name, 'Accuracy (%)': acc * 100})
    print(f"✅ {name} เทรนเสร็จแล้ว! (Accuracy: {acc*100:.2f}%)")

# 5. สรุปผลลัพธ์
print("\n🏆 สรุปผลการแข่งขัน (ตารางคะแนน):")
results_df = pd.DataFrame(results).sort_values(by='Accuracy (%)', ascending=False).reset_index(drop=True)
print(results_df.to_string())

print("\n💡 คำแนะนำ: เลือกโมเดลที่ได้อันดับ 1 ไปใช้ใน app.py ของคุณได้เลย!")