import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Modülleri bulabilmesi için yol ayarı
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import initialize_agent

# --- API AYARLARI ---
app = FastAPI(
    title="Çukurova Üniversitesi AI Asistanı",
    description="RAG ve ReAct tabanlı Akıllı Asistan API",
    version="1.0"
)

# --- MODEL TANIMLARI ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# --- AJANI BAŞLAT ---
print("🤖 Ajan hafızaya yükleniyor...")
try:
    agent_executor = initialize_agent()
    print("✅ Ajan Hazır!")
except Exception as e:
    print(f"❌ Ajan Yükleme Hatası: {e}")
    agent_executor = None

# --- ENDPOINTLER ---

@app.get("/")
def home():
    return {"status": "online", "message": "ÇÜ Asistan API Çalışıyor. /docs adresine giderek test edebilirsin."}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Ajan aktif değil.")
    
    try:
        # Ajanı çalıştır
        result = agent_executor.invoke({"input": request.question})
        
        # Sonucu döndür
        return QueryResponse(answer=result["output"])
    except Exception as e:
        return QueryResponse(answer=f"Bir hata oluştu: {str(e)}")