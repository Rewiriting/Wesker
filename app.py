import os
import sys
from unittest.mock import MagicMock

# 1. Жесткая блокировка всего, что ломает импорт
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. Создаем моки ДО импорта transformers
class MockModel:
    def __init__(self, *args, **kwargs):
        pass
    
    def encode(self, texts, **kwargs):
        import numpy as np
        if isinstance(texts, str):
            return np.random.randn(384).astype('float32')
        return np.random.randn(len(texts), 384).astype('float32')

sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers.SentenceTransformer'] = MockModel
sys.modules['sentence_transformers.model_card'] = MagicMock()

# 3. Глушим transformers
import transformers
transformers.trainer = MagicMock()
transformers.integrations = MagicMock()
transformers.integrations.deepspeed = MagicMock()

# 4. Все остальные импорты
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import random
import json
from sklearn.preprocessing import MultiLabelBinarizer, normalize
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- НАСТРОЙКИ EMAIL ---
EMAIL_SETTINGS = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "4736250@gmail.com",  
    "sender_password": "uumm rpji hfix rvob",  
    "use_tls": True
}
def send_user_data_email(user_email, username, password, secret_word):
    """Отправляет email с данными пользователя"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SETTINGS["sender_email"]
        msg['To'] = user_email
        msg['Subject'] = "🔐 Ваши данные для входа - КИНО AI"
        
        body = f"""
        <html>
        <body>
            <h2>🔐 Ваши данные для входа</h2>
            <p>Здравствуйте, <b>{username}</b>!</p>
            <p>Вы запросили восстановление доступа к сервису <b>КИНО AI</b>.</p>
            <div style="background: #f0f0f0; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <p><b>👤 Логин:</b> <code style="background: #fff; padding: 4px 8px; border-radius: 4px;">{username}</code></p>
                <p><b>🔑 Пароль:</b> <code style="background: #fff; padding: 4px 8px; border-radius: 4px;">{password}</code></p>
                <p><b>🔐 Кодовое слово:</b> <code style="background: #fff; padding: 4px 8px; border-radius: 4px;">{secret_word}</code></p>
            </div>
            <p><b>⚠️ Важно:</b> Храните эти данные в безопасности.</p>
            <p>Если вы не запрашивали восстановление доступа, проигнорируйте это письмо.</p>
            <hr>
            <p style="color: #888; font-size: 12px;">С уважением, команда КИНО AI</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        if EMAIL_SETTINGS["use_tls"]:
            server = smtplib.SMTP(EMAIL_SETTINGS["smtp_server"], EMAIL_SETTINGS["smtp_port"])
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(EMAIL_SETTINGS["smtp_server"], EMAIL_SETTINGS["smtp_port"])
        
        server.login(EMAIL_SETTINGS["sender_email"], EMAIL_SETTINGS["sender_password"])
        server.send_message(msg)
        server.quit()
        
        return True, "Данные успешно отправлены!"
    except Exception as e:
        return False, f"Ошибка при отправке письма: {str(e)}"
    
class SimpleEmbedder:
    def __init__(self, model_name=None):
        self.dim = 384
        self.genre_embeddings = {}
        genres = ['Боевик', 'Приключения', 'Комедия', 'Драма', 'Фантастика', 'Триллер', 'Ужасы', 'Мультфильм', 
                  'Мелодрама', 'Детектив', 'Военный', 'Исторический', 'Криминал', 'Фэнтези']
        for genre in genres:
            self.genre_embeddings[genre] = np.random.randn(self.dim).astype('float32')
            self.genre_embeddings[genre] /= np.linalg.norm(self.genre_embeddings[genre])
    
    def encode(self, texts, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = np.zeros((len(texts), self.dim), dtype='float32')
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            for genre, emb in self.genre_embeddings.items():
                if genre.lower() in text_lower:
                    embeddings[i] += emb * np.random.uniform(0.5, 1.5)
            
            if np.all(embeddings[i] == 0):
                embeddings[i] = np.random.randn(self.dim).astype('float32')
            
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        
        return embeddings

SentenceTransformer = SimpleEmbedder

# --- ФУНКЦИИ СОХРАНЕНИЯ ---
USERS_FILE = "users_db.json"

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"admin": {"pass": "1234", "email": "admin@mail.ru", "secret": "админ"}}
    return {"admin": {"pass": "1234", "email": "admin@mail.ru", "secret": "админ"}}

def save_users(users_dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users_dict, f, ensure_ascii=False, indent=4)

def show_my_ratings():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Мои оценки")
    
    rated_df = st.session_state.db[st.session_state.db['my_rating'] > 0]
    
    if not rated_df.empty:
        st.sidebar.write(f"Оценено фильмов: **{len(rated_df)}**")
        st.sidebar.write(f"Средний балл: **{rated_df['my_rating'].mean():.1f}**")
        
        with st.sidebar.expander("Посмотреть все"):
            for _, row in rated_df.sort_values(by='my_rating', ascending=False).iterrows():
                st.sidebar.write(f"⭐ {row['my_rating']} — **{row['title']}**")
    else:
        st.sidebar.info("Вы еще не поставили ни одной оценки.")

class FastMovieRecommender:
    def __init__(self, df):
        self.df = df.copy().reset_index(drop=True)
        self.model = SimpleEmbedder()
        self._prepare_features()
    
    def _prepare_features(self):
        with st.spinner("Подготовка данных для рекомендаций..."):
            mlb = MultiLabelBinarizer()
            genre_feats = mlb.fit_transform(self.df['genre'].fillna('').str.split('|')).astype('float32')
            
            texts = []
            for _, row in self.df.iterrows():
                text = f"{row['genre']} {row.get('description', '')}"
                texts.append(text)
            
            desc_feats = self.model.encode(texts, show_progress_bar=False).astype('float32')
            
            combined = np.hstack([genre_feats * 1.5, desc_feats])
            self.features = normalize(combined, axis=1).astype('float32')
            
            self.index = faiss.IndexFlatIP(self.features.shape[1])
            self.index.add(self.features)
    
    def get_content_based_recommendations(self, movie_id, top_n=5):
        """Рекомендации на основе похожести контента"""
        idx = self.df[self.df['movieId'] == movie_id].index[0]
        query_vector = self.features[idx:idx+1]
        
        distances, indices = self.index.search(query_vector, top_n + 1)
        
        recommendations = []
        for i, dist in zip(indices[0][1:], distances[0][1:]):
            if i < len(self.df):
                movie = self.df.iloc[i]
                recommendations.append({
                    'movieId': movie['movieId'],
                    'title': movie['title'],
                    'year': movie['year'],
                    'genre': movie['genre'],
                    'similarity': float(dist),
                    'my_rating': float(movie['my_rating'])
                })
        
        return pd.DataFrame(recommendations)
    
    def get_user_profile_recommendations(self, top_n=10):
        """Рекомендации на основе профиля пользователя"""
        rated_movies = self.df[self.df['my_rating'] > 0]
        
        if rated_movies.empty:
            return self._get_popular_recommendations(top_n)
        
        user_profile = np.zeros(self.features.shape[1], dtype='float32')
        total_weight = 0
        
        for _, movie in rated_movies.iterrows():
            idx = self.df[self.df['movieId'] == movie['movieId']].index[0]
            weight = movie['my_rating'] / 10.0
            user_profile += self.features[idx] * weight
            total_weight += weight
        
        if total_weight > 0:
            user_profile /= total_weight
        
        user_profile = user_profile.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(user_profile, top_n + len(rated_movies))
        
        recommendations = []
        seen_ids = set(rated_movies['movieId'].values)
        
        for i, dist in zip(indices[0], distances[0]):
            if i < len(self.df):
                movie = self.df.iloc[i]
                if movie['movieId'] not in seen_ids:
                    recommendations.append({
                        'movieId': movie['movieId'],
                        'title': movie['title'],
                        'year': movie['year'],
                        'genre': movie['genre'],
                        'score': float(dist),
                        'my_rating': 0.0
                    })
                    if len(recommendations) >= top_n:
                        break
        
        return pd.DataFrame(recommendations)
    
    def _get_popular_recommendations(self, top_n=10):
        """Популярные рекомендации для новых пользователей"""
        available = self.df[self.df['my_rating'] == 0].copy()
        return available.head(top_n)
    
    def get_genre_filtered_recommendations(self, selected_genres=None, top_n=10):
        """Рекомендации с фильтром по жанрам"""
        available_movies = self.df[self.df['my_rating'] == 0].copy()
        
        if available_movies.empty:
            return pd.DataFrame()
        
        if selected_genres and len(selected_genres) > 0:
            pattern = "|".join(selected_genres)
            available_movies = available_movies[available_movies['genre'].str.contains(pattern, na=False)]
        
        rated_movies = self.df[self.df['my_rating'] > 0]
        
        if not rated_movies.empty and not available_movies.empty:
            user_favorites = "|".join(rated_movies.sort_values(by='my_rating', ascending=False)['genre'].head(3))
            fav_list = list(set(user_favorites.split('|')))
            
            def score_movie(genres):
                return sum(1 for g in fav_list if g in genres)
            
            available_movies['score'] = available_movies['genre'].apply(score_movie)
            available_movies = available_movies.sort_values(by=['score', 'year'], ascending=False)
        
        return available_movies.head(top_n)
    
    def get_similar_to_rated(self, movie_id, top_n=5):
        """Получить похожие фильмы на только что оцененный"""
        return self.get_content_based_recommendations(movie_id, top_n)

def full_custom_html():
    urls = [
        "https://i.pinimg.com/736x/f9/7e/47/f97e475a44e0ceeba88683cbb23bfd3f.jpg",
        "https://i.pinimg.com/736x/7f/f5/db/7ff5dbe1895bc9aebc5646ed6dc728b3.jpg",
        "https://i.pinimg.com/736x/8d/0e/da/8d0edaab679ca5bae5f15489ffad3769.jpg",
        "https://i.pinimg.com/736x/74/fa/68/74fa6885bb5f81d12f3355f91b548774.jpg",
        "https://i.pinimg.com/736x/5d/23/e8/5d23e8cea42ba4a5513f6ab32f0b40c7.jpg",
        "https://i.pinimg.com/1200x/62/4d/ed/624dedec0ec3cc46727fd47567b68620.jpg",
        "https://i.pinimg.com/1200x/a6/b3/5b/a6b35b26e05dbca66f27c17c6e781df8.jpg",
        "https://i.pinimg.com/1200x/82/70/71/827071f872e7d7c131e23a69e1111752.jpg",
        "https://i.pinimg.com/736x/45/9d/4b/459d4b5ec6b037a746db7f2262b625bb.jpg"
    ]
    
    generated_css = ""
    generated_html = ""
    for i in range(20):
        u = random.choice(urls)
        left = random.randint(-10, 105)
        duration = random.uniform(12, 28)
        delay = random.uniform(0, 30)
        size = random.randint(70, 220)
        start_rot = random.randint(-45, 45)
        end_rot = start_rot + random.randint(-90, 90)
        
        generated_css += f"""
        .p-{i} {{
            left: {left}%;
            width: {size}px;
            height: {int(size*1.4)}px;
            animation: fall-{i} {duration}s linear infinite -{delay}s;
            opacity: {random.uniform(0.1, 0.4)};
        }}
        @keyframes fall-{i} {{
            0% {{ top: -400px; transform: rotate({start_rot}deg); }}
            100% {{ top: 115vh; transform: rotate({end_rot}deg); }}
        }}
        """
        generated_html += f'<div class="falling-poster p-{i}"><img src="{u}"></div>'
    
    return {"css": generated_css, "html": generated_html}

def check_password():
    if st.session_state.get("password_correct", False):
        return True
    
    data = full_custom_html()
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
        
        .stApp {{
            background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 25%, #0a1a3a 50%, #1a0a2e 75%, #0a0a1a 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }}
        
        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        .stTabs {{
            background: rgba(10, 15, 30, 0.8) !important;
            padding: 40px !important;
            border-radius: 30px !important;
            backdrop-filter: blur(20px);
            border: 2px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 0 50px rgba(0, 212, 255, 0.2), inset 0 0 50px rgba(0, 212, 255, 0.05);
            max-width: 500px;
            margin: 5vh auto !important;
            animation: slideInFade 0.8s ease-out;
        }}
        
        @keyframes slideInFade {{
            0% {{
                opacity: 0;
                transform: translateY(-30px) scale(0.95);
            }}
            100% {{
                opacity: 1;
                transform: translateY(0) scale(1);
            }}
        }}
        
        .battle-image-container {{
            text-align: center;
            margin-bottom: 20px;
            animation: pulse 2s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        .battle-image {{
            border-radius: 20px;
            box-shadow: 0 0 40px rgba(255, 215, 0, 0.3);
            max-width: 100%;
            height: auto;
        }}
        
        .falling-poster {{ position: fixed; z-index: 0; pointer-events: none; }}
        .falling-poster img {{ width: 100%; height: 100%; object-fit: cover; border-radius: 10px; }}
        
        div.stButton > button {{
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 180, 255, 0.1)) !important;
            color: #00d4ff !important;
            border: 2px solid #00d4ff !important;
            border-radius: 15px !important;
            font-family: 'Orbitron', sans-serif !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        }}
        
        div.stButton > button:hover {{
            background: linear-gradient(135deg, #00d4ff, #0088ff) !important;
            color: #000 !important;
            transform: scale(1.05) !important;
            box-shadow: 0 0 40px rgba(0, 212, 255, 0.6) !important;
        }}
        
        .stTextInput > div > div > input {{
            background: rgba(255, 255, 255, 0.05) !important;
            border: 2px solid rgba(0, 212, 255, 0.3) !important;
            border-radius: 15px !important;
            color: #fff !important;
            padding: 12px !important;
            font-size: 16px !important;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: #00d4ff !important;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        }}
        
        h1 {{
            font-family: 'Orbitron', sans-serif !important;
            font-weight: 900 !important;
            font-size: 3em !important;
            text-align: center;
            color: white;
            text-shadow: 0 0 20px #00d4ff, 0 0 40px #00d4ff, 0 0 60px #0088ff;
            animation: titleGlow 2s ease-in-out infinite;
        }}
        
        @keyframes titleGlow {{
            0%, 100% {{ text-shadow: 0 0 20px #00d4ff, 0 0 40px #00d4ff; }}
            50% {{ text-shadow: 0 0 40px #00d4ff, 0 0 80px #00d4ff, 0 0 120px #0088ff; }}
        }}
        
        p {{
            color: rgba(255, 255, 255, 0.8) !important;
        }}
        
        label {{
            color: #00d4ff !important;
            font-weight: 600 !important;
        }}
        
        .stRadio > div {{
            background: rgba(0, 212, 255, 0.1) !important;
            border-radius: 15px !important;
            padding: 10px !important;
        }}
        
        {data['css']}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(data['html'], unsafe_allow_html=True)
    
    # Контейнер с битвой
    st.markdown("""
    <div class="battle-image-container">
        <img src="https://i.pinimg.com/originals/8f/2c/3f/8f2c3f1cf4f3c0e5c7b8a9d0e1f2a3b4c.gif" 
             class="battle-image" 
             style="max-width: 400px; border-radius: 20px; box-shadow: 0 0 40px rgba(255, 215, 0, 0.5);">
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 style="text-align:center; color:white; text-shadow: 0 0 15px #00d4ff;">🎬 КИНО AI</h1>', unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["🚀 ВХОД", "✨ РЕГИСТРАЦИЯ", "🔑 ВОССТАНОВЛЕНИЕ"])
    
    with t1:
        st.markdown('<p style="color:#00d4ff; text-align:center; font-size:18px;"></p>', unsafe_allow_html=True)
        u = st.text_input("👤 Логин", key="l_i", placeholder="Введите ваш логин")
        p = st.text_input("🔒 Пароль", type="password", key="p_i", placeholder="Введите пароль")
        if st.button("⚡ ВОЙТИ", use_container_width=True):
            db = st.session_state["users_db"]
            if u in db and db[u]["pass"] == p:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("❌ Неверный логин или пароль")
    
    with t2:
        st.markdown('<p style="color:#ffd700; text-align:center; font-size:18px;">🌟 Создайте свой аккаунт</p>', unsafe_allow_html=True)
        new_u = st.text_input("👤 Придумайте логин", key="reg_u", placeholder="Ваш уникальный логин")
        new_e = st.text_input("📧 Ваша почта", key="reg_e", placeholder="example@mail.ru")
        new_p = st.text_input("🔒 Придумайте пароль", type="password", key="reg_p", placeholder="Надежный пароль")
        new_s = st.text_input("🔐 Кодовое слово", key="reg_s", placeholder="Для восстановления доступа")
        
        if st.button("🌟 ЗАРЕГИСТРИРОВАТЬСЯ", use_container_width=True):
            if new_u and new_p and new_e and new_s:
                if "@" not in new_e:
                    st.error("❌ Введите корректную почту!")
                else:
                    db = st.session_state["users_db"]
                    if new_u in db:
                        st.error("❌ Этот логин уже занят!")
                    else:
                        new_data = {"pass": new_p, "email": new_e, "secret": new_s.lower()}
                        st.session_state["users_db"][new_u] = new_data
                        save_users(st.session_state["users_db"])
                        st.success(f"✅ Аккаунт {new_u} успешно создан!")
                        st.balloons()
            else:
                st.warning("⚠️ Заполните все поля")
    
    with t3:
        st.markdown('<p style="color:#ff6b6b; text-align:center; font-size:18px;">🔍 Восстановление доступа</p>', unsafe_allow_html=True)
    
    # Выбор способа восстановления
        recovery_method = st.radio(
            "📋 Выберите способ восстановления:",
            ["🔐 Показать на экране (если забыли почту)", "📧 Отправить на почту (если помните почту)"],
            horizontal=True
        )
    
        if recovery_method == "🔐 Показать на экране (если забыли почту)":
            val = st.text_input("🔍 Введите кодовое слово", key="rec_val", placeholder="Ваше кодовое слово")
        
        if st.button("🔓 ПОКАЗАТЬ ДОСТУП", use_container_width=True):
            db = st.session_state["users_db"]
            found = False
            for login, info in db.items():
                if info["secret"] == val.lower():
                    st.success(f"✅ Доступ найден!")
                    st.info(f"👤 Логин: **{login}**\n🔑 Пароль: **{info['pass']}**\n📧 Почта: **{info['email']}**")
                    found = True
                    break
            if not found:
                st.error("❌ Пользователь с таким кодовым словом не найден")
    
        else:  # Отправить на почту
            st.markdown('<p style="color:#00d4ff; font-size:14px;">📧 Введите email, который вы указывали при регистрации</p>', unsafe_allow_html=True)
            email_for_reset = st.text_input("📧 Ваш email", key="reset_email", placeholder="example@mail.ru")
        
        if st.button("📨 ОТПРАВИТЬ ДАННЫЕ НА ПОЧТУ", use_container_width=True):
            if email_for_reset:
                db = st.session_state["users_db"]
                found_user = None
                user_data = None
                
                # Ищем пользователя по email
                for login, info in db.items():
                    if info["email"].lower() == email_for_reset.lower():
                        found_user = login
                        user_data = info
                        break
                
                if found_user and user_data:
                    # Отправляем email со всеми данными
                    success, message = send_user_data_email(
                        email_for_reset,
                        found_user,
                        user_data["pass"],
                        user_data["secret"]
                    )
                    
                    if success:
                        st.success(f"✅ Данные отправлены на почту {email_for_reset}!")
                        st.info("📧 Проверьте папку 'Спам', если письмо не пришло.")
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error("❌ Пользователь с таким email не найден")
            else:
                st.warning("⚠️ Введите email")

@st.cache_data
def get_movies_data():
    """Генерация русской базы фильмов"""
    russian_movies = [
        "Брат", "Брат 2", "Игла", "Асса", "Кин-дза-дза!", "Москва слезам не верит",
        "Ирония судьбы", "Бриллиантовая рука", "Иван Васильевич меняет профессию",
        "Джентльмены удачи", "Операция Ы", "Кавказская пленница", "Любовь и голуби",
        "Служебный роман", "Девчата", "В бой идут одни старики", "Они сражались за Родину",
        "Сталкер", "Солярис", "Андрей Рублев", "Зеркало", "Левиафан", "Нелюбовь",
        "Дурак", "Аритмия", "Как я провел этим летом", "Остров", "Возвращение",
        "Русалка", "Питер FM", "Прогулка", "Ночной дозор", "Дневной дозор",
        "9 рота", "Турецкий гамбит", "Статский советник", "Адмирал", "Викинг",
        "Движение вверх", "Легенда №17", "Экипаж", "Притяжение", "Вторжение",
        "Т-34", "Салют-7", "Время первых", "Лед", "Тренер", "Текст", "Холоп",
        "Полицейский с Рублевки", "Горько!", "Елки", "О чем говорят мужчины",
        "День радио", "Изображая жертву", "Кислород", "Жить", "Духless",
        "Метро", "Мажор", "Чернобыль", "Битва за Севастополь", "28 панфиловцев",
        "Тихий Дон", "Война и мир", "Анна Каренина", "Преступление и наказание",
        "Мастер и Маргарита", "Собачье сердце", "12 стульев", "Золотой теленок",
        "Место встречи изменить нельзя", "Семнадцать мгновений весны", "Щит и меч",
        "Гостья из будущего", "Приключения Электроника", "Человек-амфибия",
        "Чародеи", "Формула любви", "Обыкновенное чудо", "Тот самый Мюнхгаузен"
    ]
    
    international_movies = [
        "Побег из Шоушенка", "Крестный отец", "Темный рыцарь", "Криминальное чтиво",
        "Властелин колец", "Бойцовский клуб", "Форрест Гамп", "Матрица",
        "Начало", "Список Шиндлера", "Молчание ягнят", "Гладиатор",
        "Интерстеллар", "Аватар", "Отступники", "Социальная сеть",
        "Одержимость", "Джанго освобожденный", "Бесславные ублюдки",
        "Поймай меня если сможешь", "Остров проклятых", "Помни",
        "Престиж", "Темный рыцарь: Возрождение легенды", "Мстители",
        "Железный человек", "Первый мститель", "Стражы Галактики",
        "Человек-паук", "Люди Икс", "Дэдпул", "Логан", "Джокер",
        "Паразиты", "Олдбой", "Магнолия", "Бойцовский клуб",
        "Большой Лебовски", "Старикам тут не место", "Нефть",
        "Выживший", "Волк с Уолл-стрит", "Хороший, плохой, злой",
        "Однажды в Америке", "Леон", "Пятый элемент", "Никита",
        "Бегущий по лезвию", "Чужой", "Хищник", "Терминатор",
        "Назад в будущее", "Парк Юрского периода", "Звездные войны"
    ]
    
    genres_pool = ['Боевик', 'Приключения', 'Комедия', 'Драма', 'Фантастика', 'Триллер', 
                   'Ужасы', 'Мультфильм', 'Мелодрама', 'Детектив', 'Военный', 
                   'Исторический', 'Криминал', 'Фэнтези']
    
    data = []
    movie_id = 1
    
    for title in russian_movies[:100]:
        year = random.randint(1960, 2024)
        genre_count = random.randint(1, 3)
        genre = '|'.join(random.sample(genres_pool, genre_count))
        
        data.append({
            'movieId': movie_id,
            'title': title,
            'year': year,
            'genre': genre,
            'my_rating': 0.0
        })
        movie_id += 1
    
    for title in international_movies[:100]:
        year = random.randint(1990, 2024)
        genre_count = random.randint(1, 3)
        genre = '|'.join(random.sample(genres_pool, genre_count))
        
        data.append({
            'movieId': movie_id,
            'title': title,
            'year': year,
            'genre': genre,
            'my_rating': 0.0
        })
        movie_id += 1
    
    return pd.DataFrame(data)

def display_similar_movies(recommender, movie_id, movie_title, movie_rating):
    """Отображает похожие фильмы на основе оцененного"""
    if movie_rating >= 7.0:
        with st.spinner(f"🔍 Ищем фильмы похожие на '{movie_title}'..."):
            similar = recommender.get_similar_to_rated(movie_id, top_n=3)
            
            if not similar.empty:
                st.markdown("---")
                st.markdown(f"### 🎯 Похожие фильмы на **{movie_title}**")
                st.caption("На основе вашей высокой оценки")
                
                for _, sim_row in similar.iterrows():
                    with st.expander(f"🎥 {sim_row['title']} ({sim_row['year']}) - Похожесть: {sim_row['similarity']:.1%}"):
                        st.write(f"**Жанры:** {sim_row['genre']}")
                        
                        sim_r_key = f"rate_similar_{sim_row['movieId']}_{random.randint(1000, 9999)}"
                        movie_filter = st.session_state.db['movieId'] == sim_row['movieId']
                        current_sim_rating = float(st.session_state.db.loc[movie_filter, 'my_rating'].iloc[0])
                        
                        rating = st.slider("Оцените фильм", 0.0, 10.0, current_sim_rating, 0.5, key=sim_r_key)
                        
                        if rating != current_sim_rating:
                            st.session_state.db.loc[movie_filter, 'my_rating'] = rating
                            st.toast(f"⭐ Оценка {rating} сохранена!")

def main_app():
    st.title("🎬 Персональный Movie AI")
    
    if 'db' not in st.session_state:
        st.session_state.db = get_movies_data()
    
    # Добавляем стили для главного интерфейса
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 25%, #0a1a3a 50%, #1a0a2e 75%, #0a0a1a 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .stSidebar {
            background: linear-gradient(180deg, rgba(10, 10, 26, 0.95) 0%, rgba(26, 10, 46, 0.95) 100%) !important;
            border-right: 2px solid rgba(0, 212, 255, 0.3) !important;
            box-shadow: 5px 0 30px rgba(0, 212, 255, 0.1) !important;
        }
        
        div.stButton > button {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 180, 255, 0.1)) !important;
            color: #00d4ff !important;
            border: 2px solid #00d4ff !important;
            border-radius: 15px !important;
            transition: all 0.3s ease !important;
        }
        
        div.stButton > button:hover {
            background: linear-gradient(135deg, #00d4ff, #0088ff) !important;
            color: #000 !important;
            transform: scale(1.05) !important;
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.6) !important;
        }
        
        .stSlider > div > div > div {
            background: linear-gradient(90deg, #00d4ff, #0088ff) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Инициализируем рекомендательную систему
    recommender = FastMovieRecommender(st.session_state.db)
    
    # --- ВКЛАДКИ ДЛЯ РАЗНЫХ ТИПОВ РЕКОМЕНДАЦИЙ ---
    tab1, tab2, tab3 = st.tabs(["🎯 Умные рекомендации", "🔍 Поиск по жанрам", "📊 Аналитика"])
    
    with tab1:
        st.subheader("🤖 Рекомендации на основе ваших оценок")
        
        rated_count = len(st.session_state.db[st.session_state.db['my_rating'] > 0])
        
        if rated_count == 0:
            st.info("👋 Вы еще не поставили ни одной оценки! Оцените несколько фильмов в боковой панели, чтобы получить персонализированные рекомендации.")
        else:
            st.success(f"✅ У вас {rated_count} оценок. Система готова предложить рекомендации!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎲 Персональные рекомендации", use_container_width=True):
                    with st.spinner("Анализируем ваши предпочтения..."):
                        recommendations = recommender.get_user_profile_recommendations(top_n=10)
                        
                        if not recommendations.empty:
                            st.session_state.personal_recs = recommendations
                        else:
                            st.warning("Не удалось найти подходящие рекомендации")
            
            with col2:
                if st.button("⭐ Похожие на лучшие оценки", use_container_width=True):
                    with st.spinner("Ищем похожие фильмы..."):
                        best_rated = st.session_state.db[st.session_state.db['my_rating'] > 0].nlargest(1, 'my_rating')
                        
                        if not best_rated.empty:
                            best_movie = best_rated.iloc[0]
                            similar = recommender.get_content_based_recommendations(best_movie['movieId'], top_n=5)
                            
                            if not similar.empty:
                                st.session_state.personal_recs = similar
                                st.info(f"🎯 На основе фильма: **{best_movie['title']}** (ваша оценка: {best_movie['my_rating']})")
            
            if 'personal_recs' in st.session_state and not st.session_state.personal_recs.empty:
                st.write("### 🎯 Рекомендованные фильмы:")
                
                for _, row in st.session_state.personal_recs.iterrows():
                    with st.expander(f"🎥 {row['title']} ({row['year']})"):
                        st.write(f"**Жанры:** {row['genre']}")
                        
                        if 'similarity' in row:
                            st.write(f"**Похожесть:** {row['similarity']:.2%}")
                        if 'score' in row:
                            st.write(f"**Рейтинг соответствия:** {row['score']:.2f}")
                        
                        r_key = f"rate_personal_{row['movieId']}"
                        movie_filter = st.session_state.db['movieId'] == row['movieId']
                        current_rating = float(st.session_state.db.loc[movie_filter, 'my_rating'].iloc[0])
                        
                        rating = st.slider("Оцените фильм", 0.0, 10.0, current_rating, 0.5, key=r_key)
                        
                        if rating != current_rating:
                            st.session_state.db.loc[movie_filter, 'my_rating'] = rating
                            st.toast(f"⭐ Оценка {rating} сохранена!")
                            if rating >= 7.0:
                                display_similar_movies(recommender, row['movieId'], row['title'], rating)
    
    with tab2:
        st.subheader("🔍 Поиск по жанрам")
        genres_pool = ['Боевик', 'Приключения', 'Комедия', 'Драма', 'Фантастика', 'Триллер', 
                       'Ужасы', 'Мультфильм', 'Мелодрама', 'Детектив', 'Военный', 
                       'Исторический', 'Криминал', 'Фэнтези']
        
        selected_genres = st.multiselect(
            "Выберите жанры (можно несколько):",
            options=genres_pool,
            key="genre_multiselect"
        )
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            if st.button("🔍 НАЙТИ ФИЛЬМЫ", use_container_width=True):
                with st.spinner("Поиск фильмов..."):
                    recommendations = recommender.get_genre_filtered_recommendations(selected_genres)
                    st.session_state.genre_recs = recommendations
        
        with col2:
            if st.button("🗑️ Очистить", use_container_width=True):
                if "genre_recs" in st.session_state:
                    del st.session_state.genre_recs
                if "genre_similar_shown" in st.session_state:
                    del st.session_state.genre_similar_shown
        
        if "genre_recs" in st.session_state:
            recs = st.session_state.genre_recs
            
            if not recs.empty:
                st.write(f"### 🎯 Найдено фильмов: {len(recs)}")
                
                if "genre_similar_shown" not in st.session_state:
                    st.session_state.genre_similar_shown = {}
                
                for _, row in recs.iterrows():
                    r_key = f"rate_genre_{row['movieId']}"
                    
                    movie_filter = st.session_state.db['movieId'] == row['movieId']
                    db_rating = float(st.session_state.db.loc[movie_filter, 'my_rating'].iloc[0])
                    
                    with st.expander(f"🎥 {row['title']} ({row['year']})"):
                        st.write(f"**Жанры:** {row['genre']}")
                        
                        if 'score' in row:
                            st.write(f"**Совпадение с интересами:** {row['score']}/3")
                        
                        rating = st.slider("Оцените фильм", 0.0, 10.0, db_rating, 0.5, key=r_key)
                        
                        if rating != db_rating:
                            st.session_state.db.loc[movie_filter, 'my_rating'] = rating
                            st.toast(f"⭐ Оценка {rating} сохранена!")
                            
                            movie_key = f"movie_{row['movieId']}"
                            if rating >= 7.0 and movie_key not in st.session_state.genre_similar_shown:
                                st.session_state.genre_similar_shown[movie_key] = True
                                display_similar_movies(recommender, row['movieId'], row['title'], rating)
            else:
                st.warning("Ничего не найдено. Попробуйте другие жанры.")
    
    with tab3:
        st.subheader("📊 Аналитика ваших предпочтений")
        
        rated_movies = st.session_state.db[st.session_state.db['my_rating'] > 0]
        
        if not rated_movies.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Оценено фильмов", len(rated_movies))
            
            with col2:
                avg_rating = rated_movies['my_rating'].mean()
                st.metric("Средняя оценка", f"{avg_rating:.1f}")
            
            with col3:
                favorite_genre = rated_movies.loc[rated_movies['my_rating'].idxmax(), 'genre']
                st.metric("Лучший фильм", f"⭐ {rated_movies['my_rating'].max()}")
            
            st.write("### Распределение оценок")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(rated_movies['my_rating'], bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Оценка')
            ax.set_ylabel('Количество фильмов')
            ax.set_title('Распределение ваших оценок')
            st.pyplot(fig)
            
            st.write("### 🏆 Топ-5 любимых фильмов")
            top_5 = rated_movies.nlargest(5, 'my_rating')
            
            for _, row in top_5.iterrows():
                st.write(f"⭐ {row['my_rating']} — **{row['title']}** ({row['year']}) — {row['genre']}")
                
                with st.expander(f"🎯 Фильмы похожие на '{row['title']}'"):
                    similar = recommender.get_similar_to_rated(row['movieId'], top_n=3)
                    if not similar.empty:
                        for _, sim_row in similar.iterrows():
                            st.write(f"• **{sim_row['title']}** ({sim_row['year']}) - {sim_row['genre']} (Похожесть: {sim_row['similarity']:.1%})")
        else:
            st.info("Оцените несколько фильмов, чтобы увидеть аналитику!")

# --- ЗАПУСК ПРИЛОЖЕНИЯ ---
if "users_db" not in st.session_state:
    st.session_state["users_db"] = load_users()

if check_password():
    if 'db' not in st.session_state:
        st.session_state.db = get_movies_data()
    
    if 'quick_rate_movies' not in st.session_state:
        unrated = st.session_state.db[st.session_state.db['my_rating'] == 0]
        if not unrated.empty:
            sample_size = min(5, len(unrated))
            st.session_state.quick_rate_movies = unrated.sample(sample_size).copy()
        else:
            st.session_state.quick_rate_movies = pd.DataFrame()
    
    with st.sidebar:
        # Добавляем стили для боковой панели
        st.markdown("""
        <style>
            .sidebar-header {
                background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 136, 255, 0.1));
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
                border: 2px solid rgba(0, 212, 255, 0.3);
            }
        </style>
        <div class="sidebar-header">
            <h3 style="color: #00d4ff; margin: 0;">⭐ БЫСТРАЯ ОЦЕНКА</h3>
            <p style="color: rgba(255,255,255,0.7); margin: 10px 0 0 0; font-size: 14px;">
                Оценивайте фильмы и получайте рекомендации!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Выйти", use_container_width=True):
            st.session_state["password_correct"] = False
            st.rerun()
        
        if st.button("🔄 Обновить список фильмов", use_container_width=True):
            unrated = st.session_state.db[st.session_state.db['my_rating'] == 0]
            if not unrated.empty:
                sample_size = min(5, len(unrated))
                st.session_state.quick_rate_movies = unrated.sample(sample_size).copy()
            else:
                st.session_state.quick_rate_movies = pd.DataFrame()
        
        st.markdown("---")
        
        quick_movies = st.session_state.quick_rate_movies
        
        if not quick_movies.empty:
            for idx, movie_row in quick_movies.iterrows():
                # Стилизованный контейнер для каждого фильма
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 136, 255, 0.05));
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 10px;
                    border-left: 3px solid #00d4ff;
                ">
                    <p style="color: #fff; font-weight: bold; margin: 0 0 10px 0;">
                        🎬 {movie_row['title']} ({movie_row['year']})
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                movie_filter = st.session_state.db['movieId'] == movie_row['movieId']
                current_rating = float(st.session_state.db.loc[movie_filter, 'my_rating'].iloc[0])
                
                new_rating = st.slider(
                    f"Оценка",
                    0.0, 10.0,
                    current_rating,
                    0.5,
                    key=f"sidebar_slider_{movie_row['movieId']}",
                    label_visibility="collapsed"
                )
                
                if new_rating != current_rating:
                    st.session_state.db.loc[movie_filter, 'my_rating'] = new_rating
                    st.toast(f"⭐ Оценка {new_rating} для '{movie_row['title']}' сохранена!")
        else:
            st.info("🎉 Все фильмы оценены!")
            if st.button("🔄 Сбросить все оценки"):
                st.session_state.db['my_rating'] = 0.0
                st.session_state.quick_rate_movies = st.session_state.db[st.session_state.db['my_rating'] == 0].sample(min(5, len(st.session_state.db)))
                st.rerun()
        
        show_my_ratings()
    
    main_app()