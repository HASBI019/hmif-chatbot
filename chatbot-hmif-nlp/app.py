import streamlit as st
import pickle
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Chatbot HMIF",
    page_icon="🤖",
    layout="wide"
)

# ================= GLOBAL CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #F8FAFC, #E0F2FE);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A, #1E293B);
}

/* CHAT MESSAGE */
[data-testid="stChatMessage"] * {
    color: #0F172A !important;
}

[data-testid="stChatMessage"][aria-label="user"] {
    background: #DBEAFE;
    border-radius: 18px;
    padding: 14px;
    margin-bottom: 12px;
}

[data-testid="stChatMessage"][aria-label="assistant"] {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 14px;
    margin-bottom: 12px;
    border: 1px solid #E5E7EB;
}

/* CHAT INPUT */
.stChatInputContainer textarea {
    border-radius: 20px;
}

/* BUTTON CUSTOM */
.stButton > button {
    border-radius: 999px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
    background: linear-gradient(135deg, #2563EB, #3B82F6);
    color: white !important;
    border: none !important;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(37,99,235,0.35);
}

/* FOOTER FIXED */
.hmif-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: linear-gradient(135deg, #0F172A, #1E293B);
    color: #CBD5E1;
    text-align: center;
    font-size: 12px;
    padding: 8px 0;
    z-index: 999;
}

.hmif-footer a {
    color: #38BDF8;
    text-decoration: none;
    margin: 0 8px;
    font-weight: 600;
}

.hmif-footer a:hover {
    text-decoration: underline;
}

/* BIAR KONTEN NGGAK KETUTUP FOOTER */
section.main {
    padding-bottom: 70px;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    h1 {
        font-size: 24px !important;
    }
    .hmif-footer {
        font-size: 11px;
    }
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    # Cek berbagai kemungkinan lokasi folder models
    possible_paths = [
        "models/",                   
        "chatbot-hmif-nlp/models/",  
        "../models/"                 
    ]
    
    path_found = ""
    for p in possible_paths:
        if os.path.exists(p + "tfidf_vectorizer.pkl"):
            path_found = p
            break
            
    if path_found == "":
        return None, None

    try:
        with open(path_found + "tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open(path_found + "hmif_chatbot_data.pkl", "rb") as f:
            df = pickle.load(f)
        return vectorizer, df
    except Exception as e:
        return None, None

vectorizer, df = load_model()
stemmer = StemmerFactory().create_stemmer()

# ================= LOGIKA CHATBOT =================
def chatbot_response(text):
    text_lower = text.lower()

    # --- BAGIAN 1: MANUAL OVERRIDE (Hanya untuk Permikomnas & Sejarah) ---
    # SAYA SUDAH MENGHAPUS BAGIAN "KETUA" DARI SINI
    # Jadi bot sekarang akan mencari info Ketua langsung dari CSV/Model
    
    if "permikomnas" in text_lower:
        return "Permikomnas adalah Perhimpunan Mahasiswa Informatika dan Komputer Nasional. Wadah ini menyatukan mahasiswa IT dari seluruh Indonesia untuk berkolaborasi dan berkembang bersama."
    
    if "sejarah" in text_lower and "hmif" in text_lower:
        return "HMIF didirikan pada tanggal 20 Februari dan diresmikan pada tanggal 20 Februari 2021, sebagai wadah aspirasi mahasiswa Informatika."
    
    # --- BAGIAN 2: DATA DARI MODEL AI (.PKL) ---
    if vectorizer is None:
        return "⚠️ Maaf, otak bot (Model AI) belum ketemu path-nya."

    try:
        # 1. Stemming: Mengubah "Ketuanya siapa?" jadi "ketua siapa" biar cocok sama CSV
        text_stemmed = stemmer.stem(text_lower)
        
        # 2. Hitung Kemiripan
        vec = vectorizer.transform([text_stemmed])
        sim = cosine_similarity(vec, vectorizer.transform(df["clean_question"]))
        
        best_match_index = np.argmax(sim)
        score = sim[0][best_match_index]

        # 3. Cek Ambang Batas (Threshold)
        # Kalau kemiripan di bawah 10%, anggap bot tidak tahu
        if score < 0.1:
            return "Maaf, saya kurang paham. Coba gunakan kata kunci lain seperti 'Ketua Himpunan' atau 'Program Kerja'."
            
        return df.iloc[best_match_index]["answer"]
        
    except Exception:
        return "Maaf, ada gangguan teknis pada sistem pemrosesan kata."

# ================= SIDEBAR =================
with st.sidebar:
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        logo_path = None
        if os.path.exists("logo_hmif.png"):
            logo_path = "logo_hmif.png"
        elif os.path.exists("assets/logo_hmif.png"):
            logo_path = "assets/logo_hmif.png"
        elif os.path.exists("chatbot-hmif-nlp/app/assets/logo_hmif.png"):
            logo_path = "chatbot-hmif-nlp/app/assets/logo_hmif.png"
            
        if logo_path:
            st.image(logo_path, width=140)
        else:
            st.markdown("<div style='text-align:center; font-size:40px;'>🤖</div>", unsafe_allow_html=True)

    st.markdown("""
    <h2 style='color:#38BDF8;text-align:center;'>HMIF Assistant</h2>
    <p style='color:#CBD5E1;text-align:center;font-size:14px;'>
    Chatbot Informasi<br>Himpunan Mahasiswa Informatika
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    st.info("""
**Topik Populer**
- Sejarah HMIF  
- Pengurus  
- Program Kerja  
- Permikomnas
""")

    if st.button("🗑 Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# ================= MAIN PAGE =================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown("""
    <h1 style='text-align:center;color:#0F172A;'>🤖 Chat HMIF</h1>
    <p style='text-align:center;color:#475569;'>
    Asisten digital HMIF — cepat, informatif, dan santai
    </p>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;color:#0F172A;font-weight:600;margin-top:20px;">
        ✨ Coba pertanyaan ini:
        </div>
        """, unsafe_allow_html=True)

        contoh = [
            "Apa itu HMIF?",
            "Siapa Ketua HMIF?",
            "Visi Misi HMIF"
        ]

        for c in contoh:
            if st.button(c):
                st.session_state.messages.append({"role":"user","content":c})
                st.session_state.messages.append({"role":"assistant","content":chatbot_response(c)})
                st.rerun()

    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

# ================= INPUT FIELD =================
if prompt := st.chat_input("Tulis pertanyaan..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.spinner("Mengetik jawaban..."):
        reply = chatbot_response(prompt)
    st.session_state.messages.append({"role":"assistant","content":reply})
    st.rerun()

# ================= FOOTER =================
st.markdown("""
<div class="hmif-footer">
© 2025 HMIF • Chatbot HMIF |
<a href="https://www.instagram.com/hmif_sttcipasung" target="_blank">Instagram</a> |
<a href="https://www.tiktok.com/@hmif_sttcipasung" target="_blank">Tiktok</a>
</div>
""", unsafe_allow_html=True)
