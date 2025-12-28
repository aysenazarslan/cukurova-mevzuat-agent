import sys
import os

# src klasörünü python yoluna ekle (import hatası almamak için)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agent import initialize_agent

def main():
    print("🎓 Çukurova Üniversitesi AI Asistanı Başlatılıyor...")
    try:
        agent_executor = initialize_agent()
        print("✅ Sistem Hazır! (Çıkmak için 'q' yazın)")
        print("-" * 50)
    except Exception as e:
        print(f"❌ Başlatma Hatası: {e}")
        return

    while True:
        try:
            # Kullanıcıdan soru al
            user_input = input("\n👤 Sorunuz: ")
            
            if user_input.lower() in ['q', 'exit', 'çıkış']:
                print("👋 Görüşmek üzere!")
                break
            
            if not user_input.strip():
                continue

            print("🤖 Düşünüyor ve Araştırıyor...\n")
            
            # Ajana soruyu gönder
            response = agent_executor.invoke({"input": user_input})
            
            # Cevabı yazdır
            print(f"\n💡 Cevap: {response['output']}")
            print("-" * 50)
            
        except Exception as e:
            print(f"bir hata oluştu: {e}")

if __name__ == "__main__":
    main()