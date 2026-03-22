import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------
# 1. ตั้งค่าหน้าเว็บให้เต็มจอและซ่อนเมนูที่ไม่จำเป็น
# ------------------------------------------------
st.set_page_config(page_title="Netflix Predictor", page_icon="🎀", layout="centered")

# **ลิสต์รูปภาพพื้นหลัง Demo สำหรับหนังดังๆ**
DEMO_BACKGROUNDS = {
    "Breaking Bad": "https://i.pinimg.com/original/b1/7d/53/b17d53b236a6f6f1400e979a78189c4a.jpg",
    "Stranger Things": "https://w0.peakpx.com/wallpaper/559/314/wallpaper-stranger-things-4-original-poster.jpg",
    "Sherlock": "https://i.pinimg.com/736x/ec/4c/32/ec4c32b50428882a98e859b85433f0b2.jpg",
    "Dexter": "https://e0.peakpx.com/wallpaper/1010/823/wallpaper-dexter-wallpaper.jpg",
    "House of Cards": "https://r4.wallpaperflare.com/wallpaper/709/962/834/house-of-cards-wallpaper-ef800a40d58841452425667a4e604f76.jpg",
}

# ภาพพื้นหลังมาตรฐาน (โรงหนังเบลอๆ)
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
# 3. ส่วนหัวของเว็บ (Main Area) - ธีมชมพูน่ารัก
# ------------------------------------------------
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    
    <div class="text-center mb-8 mt-4">
        <h1 class="text-5xl font-extrabold tracking-wider uppercase" style="font-family: 'Inter', sans-serif; color: #ff1493; text-shadow: 2px 2px 4px rgba(255, 105, 180, 0.2);">
            <span style="color: #ff69b4;">NETFLIX</span> PREDICTOR 🎀
        </h1>
        <p class="mt-4 text-xl font-medium tracking-wide" style="font-family: 'Inter', sans-serif; color: #ff69b4;">
            ระบบทำนายโอกาสฉายภาพยนตร์บน Netflix ✨
        </p>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 4. ส่วนตัวเลือกทางซ้าย (Sidebar)
# ------------------------------------------------
st.sidebar.header("🌸 ตั้งค่าและข้อมูลภาพยนตร์")

selected_model_name = st.sidebar.selectbox(
    " 1. เลือกโมเดลทำนาย", 
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
    st.sidebar.success(f"โหลดสถิติของเรื่อง {selected_title} สำเร็จ! 💖")
else:
    def_year, def_age, def_imdb, def_rt = 2015, 'Unknown', 7.5, 80.0
    selected_bg_url = DEFAULT_BACKGROUND

# ------------------------------------------------
#  Custom CSS (บังคับธีมสีชมพูพาสเทล อ่านง่ายสำหรับทุกคน)
# ------------------------------------------------
st.markdown(f"""
    <style>
        /* บังคับใช้ฟอนต์ Inter ทั้งเว็บ */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif !important;
        }}
    
        #MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}}
        
        /* 🌟 ซ่อนแถบด้านบนสุดให้โปร่งใส กลืนไปกับพื้นหลัง 🌟 */
        [data-testid="stHeader"] {{
            background-color: transparent !important;
        }}

        /* พื้นหลังหลัก ไล่สีชมพูพาสเทลทับรูปโรงหนัง (ให้ดูละมุนขึ้น) */
        .stApp {{
            background: linear-gradient(to bottom, rgba(255, 240, 245, 0.85) 0%, rgba(255, 192, 203, 0.95) 100%), url("{selected_bg_url}");
            background-size: cover;
            background-position: center top;
            background-attachment: fixed;
            transition: background-image 0.6s ease-in-out;
        }}

        /* แถบเมนูด้านซ้าย (Sidebar) สีชมพูอ่อนๆ */
        [data-testid="stSidebar"] {{
            background-color: rgba(255, 228, 225, 0.95) !important;
            border-right: 2px solid #ffb6c1;
        }}

        /* กล่องเนื้อหาตรงกลาง สีขาวขอบมนดูสะอาดตา */
        .block-container {{
            background-color: rgba(255, 255, 255, 0.9) !important;
            border-radius: 24px;
            padding: 3rem !important;
            box-shadow: 0 10px 40px rgba(255, 105, 180, 0.15);
            border: 2px solid #ffe4e1;
            margin-top: 1rem;
            margin-bottom: 2rem;
            max-width: 800px !important;
        }}

        /* แต่งปุ่มกดให้เป็นสีชมพู HotPink สไตล์น่ารัก */
        div.stButton > button:first-child {{
            background-color: #ff69b4 !important;
            color: white !important;
            width: 100%;
            border-radius: 50px !important; /* ปุ่มมนๆ */
            padding: 0.8rem 0;
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: 1px;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 105, 180, 0.4);
            margin-top: 15px;
        }}
        
        div.stButton > button:first-child:hover {{
            background-color: #ff1493 !important;
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(255, 20, 147, 0.5);
        }}
        
        /* 🌟 บังคับสีตัวหนังสือทั้งหมดให้เป็นสีเทาเข้ม จะได้อ่านง่ายสำหรับทุกคน 🌟 */
        h1, h2, h3, p, span, label, .st-emotion-cache-10trblm {{
            color: #4a4a4a !important;
        }}
    </style>
""", unsafe_allow_html=True)

# แถบเลื่อนปรับค่า
st.sidebar.markdown("---")
st.sidebar.subheader(" 3. ปรับแต่งสถิติ")
year = st.sidebar.slider(" ปีที่ฉาย", 1900, 2030, def_year)
age_options = ['Unknown', 'all', '7+', '13+', '16+', '18+']
age_index = age_options.index(def_age) if def_age in age_options else 0
age = st.sidebar.selectbox(" เรทอายุ", age_options, index=age_index)
imdb = st.sidebar.slider(" คะแนน IMDb", 0.0, 10.0, def_imdb, step=0.1)
rotten_tomatoes = st.sidebar.slider(" คะแนน Rotten Tomatoes", 0.0, 100.0, def_rt, step=1.0)

# ------------------------------------------------
# 5. ปุ่มทำนายและการแสดงผลลัพธ์
# ------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
model = load_model(selected_model_name)
st.markdown(f"<p class='text-sm tracking-wide text-center' style='color: #ff69b4 !important;'>MODELS IN USE: <span class='font-bold'>{selected_model_name.upper()}</span></p>", unsafe_allow_html=True)

if st.button(" PREDICT AVAILABILITY "):
    
    input_data = pd.DataFrame({'Year': [year], 'Age': [age], 'IMDb': [imdb], 'Rotten Tomatoes': [rotten_tomatoes]})
    
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] * 100
        
        if prediction[0] == 1:
            st.markdown(f"""
            <div class="bg-green-50 border-l-4 border-green-400 rounded-lg p-6 mt-4 shadow-lg">
                <div class="flex items-center">
                    <div class="text-5xl mr-6">🍿</div>
                    <div>
                        <h2 class="text-3xl font-extrabold text-green-700 mb-1 tracking-wide">YES! </h2>
                        <p class="text-green-600 text-lg">ภาพยนตร์เรื่องนี้น่าจะมีฉายบน <span class="font-bold">NETFLIX</span></p>
                        <div class="mt-4">
                            <span class="inline-block bg-green-200 text-green-800 text-sm px-3 py-1 rounded-full font-bold uppercase tracking-wider">
                                ความน่าจะเป็น: {probability:.1f}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="bg-red-50 border-l-4 border-red-400 rounded-lg p-6 mt-4 shadow-lg">
                <div class="flex items-center">
                    <div class="text-5xl mr-6">🎬</div>
                    <div>
                        <h2 class="text-3xl font-extrabold text-red-700 mb-1 tracking-wide">NO </h2>
                        <p class="text-red-600 text-lg">ภาพยนตร์เรื่องนี้น่าจะ <span class="font-bold">ไม่มี</span> ฉายบน NETFLIX</p>
                        <div class="mt-4">
                            <span class="inline-block bg-red-200 text-red-800 text-sm px-3 py-1 rounded-full font-bold uppercase tracking-wider">
                                ความน่าจะเป็น (ที่จะมีฉาย): {probability:.1f}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
