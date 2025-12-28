import os

# Yolları Tanımla
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# Veri yolları
DATA_PATH = os.path.join(PARENT_DIR, 'data')
CHROMA_DB_DIR = os.path.join(PARENT_DIR, 'data', 'chroma_db')

# --- TÜRKÇE USTASI MODEL ---
# İngilizce model yerine bunu kullanıyoruz.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"