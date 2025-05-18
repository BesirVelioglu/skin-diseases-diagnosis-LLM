import requests
from config import GROQ_API_KEY

def get_llm_response(disease_name, confidence):
    """
    Groq API'sine HTTP isteği göndererek hastalık hakkında detaylı bilgi alır
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Bir dermatoloji uzmanı olarak yanıt ver. 
    Hastanın cilt görüntüsü analiz edildi ve {confidence:.2%} doğruluk oranıyla '{disease_name}' tespit edildi.
    
    Lütfen aşağıdaki bilgileri sağla:
    1. Bu hastalığın kısa bir açıklaması
    2. Yaygın semptomları
    3. Olası tedavi yöntemleri
    4. Önleyici tedbirler
    5. Ne zaman bir doktora başvurulmalı
    
    Yanıtını maddeler halinde, anlaşılır ve hasta dostu bir dille ver."""

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": "Sen yardımcı bir dermatoloji uzmanısın. Tıbbi terimleri kullanırken mutlaka açıklamalarını da ekle ve hasta dostu bir dil kullan."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        error_message = f"API Hatası (Status Code: {response.status_code}): {response.text}"
        raise Exception(error_message)

def format_confidence(confidence):
    """
    Güven skorunu yüzde formatına çevirir
    """
    return f"{confidence:.2%}" 