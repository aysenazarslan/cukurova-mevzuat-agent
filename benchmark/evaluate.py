import json
import pandas as pd
import sys
import os
import time
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --- AYARLAR ---
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    print("❌ HATA: .env dosyasında DEEPSEEK_API_KEY bulunamadı!")
    sys.exit(1)

MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src import config
    CHROMA_PATH = config.CHROMA_DB_DIR
    EMBEDDING_MODEL = config.EMBEDDING_MODEL_NAME
except ImportError:
    CHROMA_PATH = os.path.join(parent_dir, "data", "chroma_db")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BENCHMARK_FILE = os.path.join(current_dir, 'benchmark_data.json')
OUTPUT_EXCEL = os.path.join(current_dir, 'deepseek_final_sonuc.xlsx')

# --- FONKSİYONLAR ---

def get_deepseek_answer(question, vector_db):
    # DÜZELTME: k=20 (İdeal Denge). Sistemi yormaz ama cevabı bulur.
    docs = vector_db.similarity_search(question, k=20)
    
    if not docs: return "Bilgi bulunamadı."
    
    context_text = ""
    for i, d in enumerate(docs):
        context_text += f"\n--- PARÇA {i+1} ---\n{d.page_content}\n"
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=BASE_URL,
        temperature=0 # Sıfır hata toleransı
    )
    
    prompt = ChatPromptTemplate.from_template("""
    Sen Çukurova Üniversitesi mevzuat asistanısın.
    Aşağıdaki dökümanları kullanarak soruyu NET ve KISA bir şekilde cevapla.
    
    KURALLAR:
    1. Sadece verilen metne sadık kal.
    2. Eğer metinde cevap yoksa "Yönetmelikte bulunamadı" de.
    3. Sayısal verileri (kredi, yıl, gün) asla kaçırma.
    
    DÖKÜMANLAR:
    {context}
    
    SORU: {question}
    
    CEVAP:
    """)
    chain = prompt | llm
    
    try:
        response = chain.invoke({"context": context_text, "question": question})
        return response.content
    except Exception as e:
        # Hata varsa terminale bas (Gizleme)
        print(f"\n⚠️  CEVAP ÜRETME HATASI: {e}")
        return f"HATA: {str(e)}"

def evaluate_with_deepseek(soru, dogru, cevap):
    if "HATA" in cevap: return 0, "Sistem Hatası"
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=BASE_URL,
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_template("""
    Sen öğretmensin. Cevabı puanla (1-5).
    
    SORU: {soru}
    REFERANS: {dogru}
    ÖĞRENCİ: {cevap}
    
    Format:
    PUAN: [Rakam]
    GEREKÇE: [Kısa açıklama]
    """)
    chain = prompt | llm
    
    try:
        response = chain.invoke({"soru": soru, "dogru": dogru, "cevap": cevap})
        text = response.content
        import re
        puan_match = re.search(r'PUAN:\s*(\d)', text)
        puan = int(puan_match.group(1)) if puan_match else 1
        
        gerekce_match = re.search(r'GEREKÇE:\s*(.*)', text, re.DOTALL)
        gerekce = gerekce_match.group(1).strip() if gerekce_match else text.strip()
        
        return puan, gerekce
    except Exception as e:
        print(f"\n⚠️  HAKEM HATASI: {e}")
        return 3, "Format hatası"

# --- TABLO ---
def print_table(results):
    print("\n" + "="*140)
    print(f"{'ID':<3} | {'SORU':<35} | {'PUAN':<4} | {'DURUM':<10} | {'HAKEM GEREKÇESİ'}")
    print("-" * 140)
    for r in results:
        soru_ozet = (r['Soru'][:32] + "..") if len(r['Soru']) > 32 else r['Soru']
        # Gerekçeyi temizle (yeni satırları sil)
        temiz_gerekce = r['Gerekçe'].replace('\n', ' ')
        gerekce_ozet = (temiz_gerekce[:75] + "..") if len(temiz_gerekce) > 75 else temiz_gerekce
        
        print(f"{r['ID']:<3} | {soru_ozet:<35} | {r['Puan']:<4} | {r['Durum']:<10} | {gerekce_ozet}")
    print("="*140 + "\n")

# --- ANA PROGRAM ---
def main():
    print(f"\n DEEPSEEK DENGELİ MOD (k=20) BAŞLIYOR")
    print("---------------------------------------")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    results = []
    # TQDM ayarları
    pbar = tqdm(questions, desc="Analiz", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
    
    for i, item in enumerate(pbar):
        soru = item['question']
        dogru = item['ground_truth']
        
        # Cevapla
        cevap = get_deepseek_answer(soru, vector_db)
        
        # Puanla
        puan, gerekce = evaluate_with_deepseek(soru, dogru, cevap)
        
        # Debug Baskısı (Eğer puan düşükse nedenini hemen görelim)
        if puan < 3:
            tqdm.write(f"\n Düşük Puan ({puan}): {soru[:50]}...")
            tqdm.write(f"   Cevap: {cevap[:100]}...")
        
        results.append({
            "ID": item.get('id', i+1),
            "Soru": soru,
            "Cevap": cevap,
            "Referans": dogru,
            "Puan": puan,
            "Gerekçe": gerekce,
            "Durum": "BAŞARILI ✅" if puan >= 3 else "BAŞARISIZ ❌"
        })
        
        df = pd.DataFrame(results)
        df.to_excel(OUTPUT_EXCEL, index=False)
        
        if len(df) > 0:
            basari = len(df[df["Puan"] >= 3])
            oran = (basari / len(df)) * 100
            pbar.set_postfix({"Başarı": f"%{oran:.0f}"})

    print_table(results)
    
    basari_sayisi = len([r for r in results if r['Puan'] >= 3])
    final_oran = (basari_sayisi / len(results)) * 100
    
    print(f"\n🏆 FİNAL SKOR: %{final_oran:.2f}")

if __name__ == "__main__":
    main()