import streamlit as st
import os
import sys
import time
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --- AYARLAR ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME
except ImportError:
    CHROMA_DB_DIR = os.path.join(parent_dir, 'data', 'chroma_db')
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="ÇÜ Mevzuat Asistanı", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
    .reportview-container {background: #f0f2f6}
    h1 {color: #1e3d59;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Çukurova Üniversitesi Mevzuat Asistanı")
st.caption("DeepSeek-V3 Motoru & RAG Teknolojisi ile Güçlendirilmiştir")

# --- YAN MENÜ ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/tr/e/e6/%C3%87ukurova_%C3%9Cniversitesi_logosu.png", width=100)
    st.markdown("### ⚙️ Sistem Durumu")
    st.success("Motor: DeepSeek V3")
    st.info("Hafıza: ChromaDB (Türkçe)")
    if st.button("Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

# --- KAYNAKLARI YÜKLE ---
@st.cache_resource
def load_resources():
    if not DEEPSEEK_API_KEY:
        st.error("API Key Eksik!")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    llm = ChatOpenAI(
        model=MODEL_NAME, 
        api_key=DEEPSEEK_API_KEY, 
        base_url=BASE_URL, 
        temperature=0.3
    )
    return vector_db, llm

vector_db, llm = load_resources()

# --- SOHBET GEÇMİŞİ BAŞLAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Ben mevzuat asistanıyım. Sorularınızı bekliyorum."}]

# --- GEÇMİŞ MESAJLARI GÖSTER ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 1. Düşünme Logları
        if "steps" in msg and msg["steps"]:
            with st.expander("🧠 Düşünme Süreci (Loglar)", expanded=False):
                for step in msg["steps"]:
                    st.write(step)
        
        # 2. Chunklar (Kaynaklar)
        if "chunks" in msg and msg["chunks"]:
            with st.expander(f"📚 Kullanılan Kaynaklar ({len(msg['chunks'])} Parça)", expanded=False):
                for i, c in enumerate(msg['chunks']):
                    st.markdown(f"**--- Parça {i+1} ---**")
                    st.caption(c) # Daha temiz görünsün diye caption yaptık

        # 3. Asıl Cevap
        st.markdown(msg["content"])

# --- ANA İŞLEM DÖNGÜSÜ ---
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # KULLANICI MESAJI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ASİSTAN CEVABI
    with st.chat_message("assistant"):
        response_container = st.empty()
        process_logs = []
        chunk_texts = []
        
        # CANLI DURUM KUTUSU
        with st.status("🧠 **Analiz Başlatılıyor...**", expanded=True) as status:
            
            # ADIM 1: ARAMA
            log1 = "🔍 **Adım 1:** Soru analiz ediliyor ve veritabanı taranıyor..."
            st.write(log1)
            process_logs.append(log1)
            
            docs = vector_db.similarity_search(prompt, k=20)
            chunk_texts = [d.page_content for d in docs]
            
            log2 = f"✅ **Bulundu:** {len(docs)} adet yönetmelik maddesi çekildi."
            st.write(log2)
            process_logs.append(log2)
            
            # --- İŞTE BURASI: Chunkları CANLI gösteriyoruz ---
            st.write("📂 **Adım 2:** Ham veriler inceleniyor...")
            with st.expander("👀 Bulunan Ham Verileri (Chunkları) İncele", expanded=False):
                for i, txt in enumerate(chunk_texts):
                    st.markdown(f"**--- Parça {i+1} ---**")
                    st.caption(txt)
            process_logs.append("📂 **Adım 2:** Ham veriler (Chunklar) bağlama eklendi.")
            # ------------------------------------------------
            
            # ADIM 3: API
            log3 = "🚀 **Adım 3:** DeepSeek-V3 API'ye bağlanılıyor..."
            st.write(log3)
            process_logs.append(log3)
            
            context_text = "\n\n".join(chunk_texts)
            
            system_prompt = ChatPromptTemplate.from_template("""
            Sen Çukurova Üniversitesi uzmanı bir asistanısın.
            Aşağıdaki YÖNETMELİK PARÇALARINI (CONTEXT) kullanarak öğrencinin sorusunu cevapla.
            
            KURALLAR:
            1. Cevabın profesyonel ve net olsun.
            2. Cevabını MUTLAKA aşağıdaki metinlere dayandır.
            
            CONTEXT:
            {context}
            
            SORU: {question}
            """)
            
            chain = system_prompt | llm
            try:
                response = chain.invoke({"context": context_text, "question": prompt})
                full_response = response.content
            except Exception as e:
                full_response = f"Hata oluştu: {e}"
                st.error("API Hatası!")

            log4 = "💡 **Sonuç:** Cevap başarıyla üretildi."
            st.write(log4)
            process_logs.append(log4)
            
            status.update(label="✅ Analiz Tamamlandı!", state="complete", expanded=True)

        # CEVABI YAZDIR
        response_container.markdown(full_response)

        # GEÇMİŞE KAYDET
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "chunks": chunk_texts, # Chunkları kaydet
            "steps": process_logs  # Logları kaydet
        })