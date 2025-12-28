# 🎓 Çukurova Üniversitesi Mevzuat Asistanı (AI-Powered RAG Agent)

![Status](https://img.shields.io/badge/Status-Completed-success)
![Success Rate](https://img.shields.io/badge/Benchmark_Score-%2593.33-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![DeepSeek](https://img.shields.io/badge/Engine-DeepSeek_V3-orange)


> Çukurova Üniversitesi yönetmeliklerini saniyeler içinde tarayan, analiz eden ve öğrencilerin sorularını kanıtlarıyla birlikte cevaplayan Yeni Nesil Yapay Zeka Asistanı.

---

##  Proje Hakkında:

Bu proje, öğrencilerin ve akademik personelin yüzlerce sayfalık PDF yönetmelikleri arasında kaybolmasını önlemek amacıyla geliştirilmiştir. **Retrieval-Augmented Generation (RAG)** mimarisi kullanılarak, statik PDF dosyaları interaktif ve zeki bir sohbet botuna dönüştürülmüştür.

Sistem, klasik anahtar kelime aramasının ötesine geçerek; **DeepSeek-V3** motoru ile muhakeme (reasoning) yapar ve sorulara **%93.33 doğrulukla** yanıt verir.

###  Temel Özellikler
* ** Yüksek Doğruluk:** Karmaşık ve koşullu sorularda dahi yüksek başarı oranı.
* ** Açıklanabilir YZ (X-Ray Modu):** Sistemin cevabı üretirken hangi yönetmelik maddelerini okuduğunu şeffaf bir şekilde gösterir.
* ** Türkçe Semantik Arama:** `paraphrase-multilingual-MiniLM-L12-v2` modeli ile Türkçeyi anlamsal olarak kavrar (Örn: "Mazeret" = "Rapor").
* ** Hızlı ve Maliyet Etkin:** DeepSeek-V3 motoru ile yüksek performans/maliyet oranı.

---

##  Proje Mimarisi:

Sistem 3 ana katmandan oluşur:
1.  **Ingestion (Veri İşleme):** PDF'ler okunur, parçalanır (Chunking) ve Vektör Veritabanına (ChromaDB) kaydedilir.
2.  **Retrieval (Bilgi Erişim):** Kullanıcı sorusu vektöre çevrilir ve en alakalı 20 yönetmelik maddesi bulunur.
3.  **Generation (Üretim):** Bulunan kanıtlar ve soru LLM'e (DeepSeek) gönderilir, cevap üretilir.

---

##  Dizin Yapısı:

```text
uni_react_agent/
├── data/                       # Veri Katmanı
│   ├── *.pdf                   # Ham Yönetmelik Dosyaları
│   └── chroma_db/              # Vektör Veritabanı (Embeddingler)
│
├── src/                        # Kaynak Kodlar
│   ├── ingest.py               # Veritabanı Oluşturucu (ETL)
│   ├── app.py                  # Streamlit Web Arayüzü
│   └── config.py               # Ayar Dosyası
│
├── benchmark/                  # Test ve Raporlama
│   ├── evaluate.py             # Başarı Ölçüm Scripti
│   ├── benchmark_data.json     # Test Soruları (Ground Truth)
│   └── deepseek_final_sonuc.xlsx # Detaylı Sonuç Raporu
│
├── .env                        # API Anahtarları
├── requirements.txt            # Kütüphaneler
└── README.md                   # Dokümantasyon

Kurulum ve Çalıştırma:
Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin.

1. Ön Hazırlık
Projeyi klonlayın ve sanal ortamı kurun:

# Sanal ortam oluştur
python -m venv venv

# Sanal ortamı aktif et (Windows)
.\venv\Scripts\activate

# Gerekli kütüphaneleri yükle
pip install -r requirements.txt

API Anahtarı:
Ana dizinde .env adında bir dosya oluşturun ve DeepSeek API anahtarınızı ekleyin

Veritabanını Kurma (Ingest):
PDF dosyalarını işleyip veritabanına kaydetmek için:

python src/ingest.py

Uygulamayı Başlatma
Web arayüzünü açmak için:

streamlit run src/app.py

Benchmark Sonuçları:
Sistem, 15 adet zorlu senaryo sorusu (Sayısal veri, koşullu durum, süreç analizi) ile test edilmiştir.

Metrik,Değer
Toplam Soru,15
Başarılı Cevap,14
Tam Puan (5/5),12
Genel Başarı Oranı,%93.33 

Detaylı test sonuçları ve hakem yorumları benchmark/deepseek_final_sonuc.xlsx dosyasında mevcuttur.

Kullanılan Teknolojiler:
-Dil: Python 3.11
-LLM: DeepSeek-V3
-Framework: LangChain
-Vector DB: ChromaDB
-Embedding: HuggingFace (Multilingual)
-Frontend: Streamlit
-PDF Parser: PyMuPDF

Lisans:
Bu proje eğitim ve akademik amaçlarla geliştirilmiştir.