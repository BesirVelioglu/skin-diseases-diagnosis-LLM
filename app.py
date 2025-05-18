import streamlit as st
from PIL import Image
import os
from model import DermNetModel
from utils import get_llm_response, format_confidence

# Sayfa yapılandırması
st.set_page_config(
    page_title="DermNet AI Asistan",
    page_icon="🏥",
    layout="wide"
)

# CSS stil tanımlamaları
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Başlık ve açıklama
st.title("🏥 DermNet AI Asistan")
st.markdown("""
Bu uygulama, cilt hastalıklarını yapay zeka kullanarak tespit eder ve size detaylı bilgi sunar.
Lütfen analiz etmek istediğiniz cilt bölgesinin fotoğrafını yükleyin.
""")

# Model yükleme
@st.cache_resource
def load_model():
    return DermNetModel("ResNet50_dermnet23.pth")

try:
    model = load_model()
    st.success("Model başarıyla yüklendi! ✅")
except Exception as e:
    st.error(f"Model yüklenirken bir hata oluştu: {str(e)}")
    st.stop()

# Dosya yükleme alanı
uploaded_file = st.file_uploader("Bir görüntü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görüntüyü göster
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Yüklenen Görüntü")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    # Analiz butonu
    if st.button("Analiz Et"):
        with st.spinner("Görüntü analiz ediliyor..."):
            # Model tahmini
            result = model.predict(image)
            disease_name = result['class']
            confidence = result['confidence']
            
            with col2:
                st.subheader("Analiz Sonuçları")
                with st.container(border=True):
                    st.markdown(f"**Tespit Edilen Hastalık:** {disease_name}")
                    st.markdown(f"**Doğruluk Oranı:** {format_confidence(confidence)}")
            
            # LLM yanıtı
            st.subheader("Detaylı Bilgi")
            with st.spinner("Detaylı bilgiler hazırlanıyor..."):
                llm_response = get_llm_response(disease_name, confidence)
                with st.container(border=True):
                    st.markdown(llm_response)
            
            # Uyarı mesajı
            st.warning("""
            ⚠️ **Önemli Not:** Bu uygulama bir tıbbi tanı aracı değildir. 
            Sonuçlar sadece bilgilendirme amaçlıdır ve kesin tanı için mutlaka bir dermatoloji uzmanına başvurunuz.
            """)

# Footer
st.markdown("---")
st.markdown("Developed by [@SametKaras](https://github.com/SametKaras) and [@BesirVelioglu](https://github.com/BesirVelioglu)") 