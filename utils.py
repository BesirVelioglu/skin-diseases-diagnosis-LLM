import requests
from config import GROQ_API_KEY

def get_llm_response(user_input, confidence=None):
    """
    Groq API'sine HTTP isteği göndererek kullanıcının sorusuna yanıt alır
    
    Args:
        user_input (str): Kullanıcının sorusu veya contextual prompt
        confidence (float, optional): Model güven skoru (varsa)
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # System prompt - dermatoloji uzmanı rolü
    system_prompt = """Sen yardımcı bir dermatoloji uzmanısın. Cilt sağlığı konularında bilgi veren, hasta dostu ve profesyonel bir asistansın.

ÖNEMLİ: HER ZAMAN TÜRKÇE YANIT VER. İngilizce kesinlikle kullanma.

Görevlerin:
- Kullanıcının cilt sağlığı sorularını TÜRKÇE yanıtlamak
- Tıbbi terimleri Türkçe olarak anlaşılır şekilde açıklamak  
- Türkçe genel bilgilendirme yapmak
- Her zaman profesyonel tıbbi yardım almanın önemini Türkçe vurgulamak

Önemli kurallar:
- Kesin tanı koyma, sadece bilgilendirme yap
- Hasta dostu, anlayışlı bir Türkçe dil kullan
- Tıbbi terimleri Türkçe açıkla
- Her yanıtta "bu bilgiler genel bilgilendirme amaçlıdır" uyarısını Türkçe ver
- Yanıtının tamamı Türkçe olmalı

Dil kuralları:
- Tüm yanıtlar sadece Türkçe
- İngilizce kelime kullanma
- Türk hastalar için uygun dil
- Medikal terimler varsa Türkçe karşılıklarını kullan"""

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            error_message = f"API Hatası (Status Code: {response.status_code}): {response.text}"
            return f"⚠️ Üzgünüm, şu anda bir teknik sorun yaşıyorum. Lütfen daha sonra tekrar deneyin.\n\nHata detayı: {error_message}"
            
    except requests.exceptions.Timeout:
        return "⚠️ Yanıt süresi aşıldı. Lütfen tekrar deneyin."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Bağlantı hatası: {str(e)}"
    except Exception as e:
        return f"⚠️ Beklenmeyen bir hata oluştu: {str(e)}"

def format_confidence(confidence):
    """
    Güven skorunu yüzde formatına çevirir
    """
    return f"{confidence:.2%}" 