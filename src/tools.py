from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys

# Yol ayarlarını yap
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def get_tools():
    print("🛠️  Araçlar (Tools) hazırlanıyor...")
    
    # 1. Embedding Modelini Hazırla
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    
    # 2. Vektör Veritabanını (Chroma) Oku
    if not os.path.exists(config.CHROMA_DB_DIR):
        raise ValueError(f"❌ Veritabanı bulunamadı: {config.CHROMA_DB_DIR}. Lütfen önce 'python src/ingest.py' çalıştır.")

    vector_db = Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    
    # 3. Retriever (Arama Motoru) Oluştur
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Her aramada en alakalı 5 parçayı getir
    )
    
    # 4. Ajanın Kullanacağı Aracı (Tool) Tanımla
    search_tool = create_retriever_tool(
        retriever,
        "uni_rules_search", # Ajanın göreceği isim
        "Çukurova Üniversitesi yönetmelikleri, sınavlar, krediler ve akademik kurallar hakkında bilgi arar. Soruları cevaplamak için MUTLAKA önce bu aracı kullan."
    )
    
    print("✅ Araçlar başarıyla yüklendi.")
    return [search_tool]