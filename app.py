import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------
# 1. ตั้งค่าหน้าเว็บให้เต็มจอ
# ------------------------------------------------
st.set_page_config(page_title="Netflix Predictor", page_icon="🎬", layout="centered")

# **ลิสต์รูปภาพพื้นหลัง Demo สำหรับหนังดังๆ**
DEMO_BACKGROUNDS = {
    "Breaking Bad": "https://i.pinimg.com/original/b1/7d/53/b17d53b236a6f6f1400e979a78189c4a.jpg",
    "Stranger Things": "https://w0.peakpx.com/wallpaper/559/314/wallpaper-stranger-things-4-original-poster.jpg",
    "Sherlock": "https://i.pinimg.com/736x/ec/4c/32/ec4c32b50428882a98e859b85433f0b2.jpg",
    "Dexter": "https://e0.peakpx.com/wallpaper/1010/823/wallpaper-dexter-wallpaper.jpg",
    "House of Cards": "https://r4.wallpaperflare.com/wallpaper/709/962/834/house-of-cards-wallpaper-ef800a40d58841452425667a4e604f76.jpg",
}

# ภาพพื้นหลังมาตรฐาน (โรงหนังเบลอๆ โทนดาร์ก)
DEFAULT_BACKGROUND = "https://w0.peakpx.com/wallpaper/276/75/wallpaper-cinematic-movie-theater-blurred-background.jpg"

# ------------------------------------------------
# 2. ฟังก์ชันโหลดข้อมูลและโมเดล
# ------------------------------------------------
@st.cache_resource
def load_model(model_name):
    models = {
        'Gradient Boosting': 'model_gb.pkl',
        'Random Forest': 'model_pipeline.pkl',
        'Support Vector Machine (SVM)': 'model_svm.pkl',
        'Logistic Regression': 'model_lr.pkl'
    }
    return joblib.load(models[model_name])

@st.cache_data
def load_data():
    df = pd.read_csv('tv_shows.csv')
    df['Age'] = df['Age'].fillna('Unknown')
    df['IMDb'] = df['IMDb'].astype(str).str.replace('/10', '').replace('nan', np.nan).astype(float)
    df['IMDb'] = df['IMDb'].fillna(df['IMDb'].median())
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].astype(str).str.replace('/100', '').replace('nan', np.nan).astype(float)
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].fillna(df['Rotten Tomatoes'].median())
    return df

df_movies = load_data()

# ------------------------------------------------
# 3. ส่วนหัวของเว็บ (Main Area) - บังคับกึ่งกลาง 100%
# ------------------------------------------------
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    
    <div style="text-align: center; padding-top: 1rem; padding-bottom: 2rem;">
        <h1 style="font-family: 'Bebas Neue', sans-serif; color: #E50914; font-size: 4.5rem; letter-spacing: 4px; text-shadow: 2px 2px 10px rgba(229, 9, 20, 0.4); margin: 0; line-height: 1;">
            NETFLIX <span style="color: #FFFFFF;">PREDICTOR</span>
        </h1>
        <p style="font-family: 'Inter', sans-serif; color: #B3B3B3; font-size: 1.2rem; font-weight: 300; margin-top: 10px; letter-spacing: 0.5px;">
            ระบบทำนายโอกาสฉายภาพยนตร์บน Netflix
        </p>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 4. ส่วนตัวเลือกทางซ้าย (Sidebar)
# ------------------------------------------------
st.sidebar.header("⚙️ ตั้งค่าระบบ")

selected_model_name = st.sidebar.selectbox(
    "🤖 1. เลือกโมเดลทำนาย", 
    ['Gradient Boosting', 'Random Forest', 'Support Vector Machine (SVM)', 'Logistic Regression']
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎬 2. ข้อมูลภาพยนตร์/ซีรีส์")

movie_titles = ["-- กรอกข้อมูลเอง (Manual) --"] + df_movies['Title'].dropna().tolist()
selected_title = st.sidebar.selectbox("🔍 ค้นหาชื่อเรื่อง", movie_titles)

if selected_title != "-- กรอกข้อมูลเอง (Manual) --":
    movie_info = df_movies[df_movies['Title'] == selected_title].iloc[0]
    def_year = int(movie_info['Year'])
    def_age = str(movie_info['Age'])
    def_imdb = float(movie_info['IMDb'])
    def_rt = float(movie_info['Rotten Tomatoes'])
    
    selected_bg_url = DEMO_BACKGROUNDS.get(selected_title, DEFAULT_BACKGROUND)
    st.sidebar.success(f"โหลดข้อมูล: {selected_title}")
else:
    def_year, def_age, def_imdb, def_rt = 2015, 'Unknown', 7.5, 80.0
    selected_bg_url = DEFAULT_BACKGROUND

# ------------------------------------------------
# 🎨 Custom CSS (บังคับธีม Netflix)
# ------------------------------------------------
st.markdown(f"""
    <style>
        /* บังคับใช้ฟอนต์ Inter ทั้งเว็บ */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif !important;
        }}
    
        /* ซ่อนลายน้ำ Streamlit แต่เก็บจุดสามจุดไว้ */
        footer {{visibility: hidden;}}
        
        /* แถบด้านบนสุดให้โปร่งใส แต่ยังกดเมนูได้ */
        [data-testid="stHeader"] {{
            background-color: transparent !important;
        }}

        /* พื้นหลังหลัก ไล่สีดำทึบแบบ Netflix ทับรูปภาพ */
        .stApp {{
            background: linear-gradient(to bottom, rgba(20, 20, 20, 0.7) 0%, rgba(20, 20, 20, 0.95) 60%, rgba(20, 20, 20, 1) 100%), url("{selected_bg_url}");
            background-size: cover;
            background-position: center top;
            background-attachment: fixed;
            transition: background-image 0.5s ease-in-out;
        }}

        /* แถบเมนูด้านซ้าย (Sidebar) สีดำสนิท */
        [data-testid="stSidebar"] {{
            background-color: #000000 !important;
            border-right: 1px solid #333;
        }}

        /* กล่องเนื้อหาตรงกลาง */
        .block-container {{
            background-color: rgba(0, 0, 0, 0.65) !important;
            border-radius: 8px;
            padding: 2.5rem 3rem !important;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-top: 2rem;
            margin-bottom: 2rem;
            max-width: 750px !important;
        }}

        /* บังคับปุ่มกดให้เต็มกล่องและอยู่ตรงกลาง */
        div.stButton > button:first-child {{
            background-color: #E50914 !important;
            color: #FFFFFF !important;
            width: 100% !important; /* บังคับกว้าง 100% */
            border-radius: 4px !important; 
            padding: 0.8rem 0;
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: 1px;
            border: none;
            transition: all 0.2s ease;
            margin-top: 10px;
        }}
        
        div.stButton > button:first-child:hover {{
            background-color: #C11119 !important; 
        }}
        
        /* สีข้อความระบบให้เป็นสีขาว/เทา */
        h2, h3, label, .st-emotion-cache-10trblm {{
            color: #FFFFFF !important;
        }}
        p, span {{
            color: #B3B3B3 !important;
        }}
    </style>
""", unsafe_allow_html=True)

# แถบเลื่อนปรับค่า
st.sidebar.markdown("---")
st.sidebar.subheader("📊 3. ปรับแต่งสถิติ")
year = st.sidebar.slider("📅 ปีที่ฉาย", 1900, 2030, def_year)
age_options = ['Unknown', 'all', '7+', '13+', '16+', '18+']
age_index = age_options.index(def_age) if def_age in age_options else 0
age = st.sidebar.selectbox("👪 เรทอายุ", age_options, index=age_index)
imdb = st.sidebar.slider("⭐ คะแนน IMDb", 0.0, 10.0, def_imdb, step=0.1)
rotten_tomatoes = st.sidebar.slider("🍅 คะแนน Rotten Tomatoes", 0.0, 100.0, def_rt, step=1.0)

# ------------------------------------------------
# 5. ปุ่มทำนายและการแสดงผลลัพธ์
# ------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

# ชื่อโมเดลตรงกลางหน้าจอ
st.markdown(f"<p style='text-align: center; color: #808080; font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase;'>Model: <span style='font-weight: bold; color: #FFFFFF;'>{selected_model_name}</span></p>", unsafe_allow_html=True)

if st.button("PREDICT"):
    
    input_data = pd.DataFrame({'Year': [year], 'Age': [age], 'IMDb': [imdb], 'Rotten Tomatoes': [rotten_tomatoes]})
    
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] * 100
        
        if prediction[0] == 1:
            st.markdown(f"""
            <div style="background-color: #181818; border-left: 4px solid #E50914; padding: 20px; border-radius: 4px; margin-top: 20px;">
                <div style="display: flex; align-items: center;">
                    <div style="font-size: 40px; margin-right: 20px;">🍿</div>
                    <div>
                        <h2 style="font-size: 24px; font-weight: bold; color: #FFFFFF; margin: 0; letter-spacing: 1px;">AVAILABLE ON NETFLIX</h2>
                        <p style="color: #B3B3B3; font-size: 16px; margin-top: 5px;">ภาพยนตร์เรื่องนี้มีโอกาสสูงที่จะฉายบนแพลตฟอร์ม</p>
                        <div style="margin-top: 10px;">
                            <span style="background-color: rgba(229, 9, 20, 0.2); color: #E50914; font-size: 14px; padding: 4px 12px; border-radius: 4px; font-weight: bold;">
                                MATCH: {probability:.0f}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #181818; border-left: 4px solid #808080; padding: 20px; border-radius: 4px; margin-top: 20px;">
                <div style="display: flex; align-items: center;">
                    <div style="font-size: 40px; margin-right: 20px;">🚫</div>
                    <div>
                        <h2 style="font-size: 24px; font-weight: bold; color: #FFFFFF; margin: 0; letter-spacing: 1px;">NOT AVAILABLE</h2>
                        <p style="color: #B3B3B3; font-size: 16px; margin-top: 5px;">ภาพยนตร์เรื่องนี้ไม่น่าจะมีฉายบนแพลตฟอร์ม</p>
                        <div style="margin-top: 10px;">
                            <span style="background-color: rgba(128, 128, 128, 0.2); color: #B3B3B3; font-size: 14px; padding: 4px 12px; border-radius: 4px; font-weight: bold;">
                                PROBABILITY: {probability:.0f}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
