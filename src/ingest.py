import os
import sys
import shutil

# --- NAVİGASYON AYARI (HATA ÇÖZÜCÜ) ---
# Kodun çalıştığı yerin bir üst klasörünü sisteme tanıtıyoruz.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# --------------------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Ayarları güvenli bir şekilde çekiyoruz
try:
    from src.config import CHROMA_DB_DIR, DATA_PATH, EMBEDDING_MODEL_NAME
except ImportError:
    # Eğer src bulunamazsa (farklı çalıştırma şekilleri için) direkt config'den al
    from config import CHROMA_DB_DIR, DATA_PATH, EMBEDDING_MODEL_NAME

def main():
    print("----------------------------------------------------")
    print(f"🌍 TÜRKÇE MODEL İLE KURULUM BAŞLIYOR: {EMBEDDING_MODEL_NAME}")
    print("----------------------------------------------------")

    # 1. TEMİZLİK
    print(f"🧹 Eski veritabanı temizleniyor...")
    if os.path.exists(CHROMA_DB_DIR):
        try:
            shutil.rmtree(CHROMA_DB_DIR)
            print("✅ Temizlik tamam.")
        except Exception as e:
            print(f"⚠️ Silme uyarısı: {e}")

    # 2. PDF ARAMA (Alt klasörler dahil)
    print(f"📂 Dosyalar taranıyor: {DATA_PATH}")
    pdf_files = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print("❌ HATA: Hiç PDF bulunamadı! 'data' klasörüne PDF yüklediğine emin ol.")
        return

    print(f"📄 Bulunan PDF Sayısı: {len(pdf_files)}")

    # 3. OKUMA
    docs = []
    for pdf in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf)
            docs.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Okuma hatası ({os.path.basename(pdf)}): {e}")

    # 4. PARÇALAMA (Türkçe için optimize edilmiş ayarlar)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, 
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    
    # Boş sayfaları ele
    quality_splits = [d for d in splits if d.page_content and len(d.page_content) > 20]
    
    print(f"🧩 Toplam {len(quality_splits)} parça veri işleniyor...")

    # 5. KAYDETME
    print("💾 Veritabanı oluşturuluyor (Model indirilirken biraz bekletebilir)...")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    # Çökmemesi için 100'er 100'er yükle
    batch_size = 100
    for i in range(0, len(quality_splits), batch_size):
        batch = quality_splits[i:i+batch_size]
        vector_db.add_documents(batch)
        print(f"   ↳ %{int((i+batch_size)/len(quality_splits)*100)} yüklendi...")

    print("\n✅ VERİTABANI HAZIR! Şimdi testi başlatabilirsin.")

if __name__ == "__main__":
    main()