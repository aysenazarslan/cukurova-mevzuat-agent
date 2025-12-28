import config
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class RagEngine:
    def __init__(self):
        # Embedding modelini yükle
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        
        # ChromaDB'ye bağlan
        self.db = Chroma(
            persist_directory=str(config.CHROMA_DB_DIR),
            embedding_function=self.embeddings
        )
        
        # Retriever (Getirici) ayarları
        # k=3: En alakalı 3 parçayı getirir
        self.retriever = self.db.as_retriever(search_kwargs={"k": 10})

    def retrieve(self, query: str) -> str:
        """
        Sorguyu alır, veritabanından ilgili parçaları bulur 
        ve ham metin (string) olarak döner.
        """
        try:
            docs = self.retriever.invoke(query)
            
            if not docs:
                return "Belgelerde bu konuyla ilgili bilgi bulunamadı."
            
            # Bulunan parçaları birleştirip formatlayalım
            result_text = ""
            for i, doc in enumerate(docs):
                result_text += f"\n--- Döküman Parçası {i+1} ---\n"
                result_text += doc.page_content
                
            return result_text
            
        except Exception as e:
            return f"Arama sırasında hata oluştu: {str(e)}"

# Test bloğu: Bu dosyayı doğrudan çalıştırırsan arama testi yapar.
if __name__ == "__main__":
    engine = RagEngine()
    test_sorusu = "Ders kaydı ne zaman yapılır?"
    print(f"Soru: {test_sorusu}")
    cevap = engine.retrieve(test_sorusu)
    print("Bulunan İçerik:", cevap)