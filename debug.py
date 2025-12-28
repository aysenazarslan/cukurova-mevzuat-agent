import sys
import os
from dotenv import load_dotenv

# 1. Ortam Değişkenlerini Yükle
load_dotenv()

# 2. Yolları Ayarla
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("🔍 DEBUG MODU BAŞLATILIYOR...")

# 3. API Key Kontrolü
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ HATA: GROQ_API_KEY bulunamadı! .env dosyasını kontrol et.")
    sys.exit()
else:
    print(f"✅ API Key mevcut: {api_key[:5]}...{api_key[-5:]}")

# 4. Ajanı Başlatmayı Dene
print("🤖 Ajan başlatılıyor...")
try:
    from src.agent import initialize_agent
    agent = initialize_agent()
    print("✅ Ajan başarıyla yüklendi.")
except Exception as e:
    print(f"❌ AJAN YÜKLEME HATASI: {e}")
    sys.exit()

# 5. Basit Bir Test Sorusu Sor
print("❓ Test sorusu soruluyor: 'Doktora kredisi kaç?'")
try:
    response = agent.invoke({"input": "Doktora kredisi en az kaçtır?"})
    print("\n📝 CEVAP GELDİ:")
    print(response['output'])
except Exception as e:
    print("\n❌ ÇALIŞMA ZAMANI HATASI (İşte aradığımız suçlu bu):")
    print(e)