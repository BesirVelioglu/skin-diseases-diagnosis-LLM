"""
Streamlit frontend UI for a skin‑disease chatbot assistant.
Integrates with DermNet model and LLM backend for real functionality.
Save this as `app.py` and run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from PIL import Image
from model import DermNetModel
from utils import get_llm_response, format_confidence

# ─────────────────────────  PAGE CONFIG  ──────────────────────────
st.set_page_config(
    page_title="DermNet AI Asistan",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────  CSS OVERRIDES  ───────────────────────
# Dark, modern, minimalist styling + narrow chat width
st.markdown(
    """
    <style>
        /* Global tweaks */
        html, body, [class*="css"], .stApp {background-color:#0e1117; color:#e6e6e7; font-family:'Inter', sans-serif;}
        /* Title */
        .app-title {font-size:2.3rem; font-weight:700; margin-bottom:0.2rem;}
        /* Subtitle */
        .subtext {font-size:0.95rem; opacity:0.75; margin-bottom:1.2rem;}
        /* Uploader */
        .block-container .stFileUploader>label {display:none;} /* hide default label */
        /* Chat container */
        .chat-wrapper {max-width:840px; margin:auto;}
        /* Chat bubbles */
        .stChatMessage.user {background:#1f1f28; border-radius:18px 18px 0 18px; padding:12px 16px; margin-bottom:8px;}
        .stChatMessage.assistant {background:#16212d; border-radius:18px 18px 18px 0; padding:12px 16px; margin-bottom:8px;}
        .stChatMessage .avatar {display:none;} /* Hide avatars for cleaner look */
        /* Input */
        .stChatInput>div>textarea {background:#1f1f28; color:#e6e6e7; border:1px solid #333; border-radius:12px; padding:0.8rem 1rem;}
        /* Remove hamburger */
        header.css-18ni7ap.e8zbici2 {display:none;}
        /* Loading spinner */
        .stSpinner {color: #e6e6e7;}
        /* Success/Error messages */
        .stSuccess {background-color: #1e4d3e; border: 1px solid #28a745; color: #90ee90;}
        .stError {background-color: #4d1e1e; border: 1px solid #dc3545; color: #ff9999;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────  SESSION STATE INIT  ──────────────────────
if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, str]] = [
        {
            "role": "assistant", 
            "content": "👋 Merhaba! Ben DermNet AI Asistanınızım. Size cilt sağlığınız konusunda yardımcı olmak için buradayım.\n\n📷 Bir fotoğraf yükleyerek analiz yaptırabilir veya cilt sağlığı ile ilgili sorularınızı sorabilirsiniz."
        }
    ]

if "image_analysis" not in st.session_state:
    # Will store {"disease": str, "confidence": float}
    st.session_state.image_analysis: dict[str, float] | None = None

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "model" not in st.session_state:
    st.session_state.model = None

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "current_image_hash" not in st.session_state:
    st.session_state.current_image_hash = None

# ─────────────────────────  MODEL LOADING  ────────────────────────
@st.cache_resource(show_spinner=False)
def load_dermatology_model():
    """Load the trained DermNet model"""
    try:
        model = DermNetModel("ResNet50_dermnet23.pth")
        return model
    except Exception as e:
        # Cache edilebilir hata objesi döndür
        return {"error": str(e)}

def get_model():
    """Get model with proper error handling"""
    cached_result = load_dermatology_model()
    if isinstance(cached_result, dict) and "error" in cached_result:
        raise Exception(cached_result["error"])
    return cached_result

# ───────────────────────────  HEADER  ─────────────────────────────
with st.container():
    st.markdown('<div class="app-title">DermNet AI Asistan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtext">Cilt hastalıklarıyla ilgili sorularınızı sorun veya bir fotoğraf yükleyin. Yüklediğiniz görüntü, eğitimli modelle analiz edilir ve sonuç sohbete eklenir.</div>',
        unsafe_allow_html=True,
    )

# ───────────────────────  IMAGE UPLOADER  ────────────────────────
with st.container():
    uploaded_file = st.file_uploader(
        "Bir fotoğraf yükleyin (JPG / PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        import hashlib
        
        # Generate hash for the uploaded file to track if it's new
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Görüntü", width=400)

        # Check if this is a new image that hasn't been analyzed
        is_new_image = file_hash != st.session_state.current_image_hash
        
        if is_new_image:
            # Reset analysis state for new image
            st.session_state.analysis_complete = False
            st.session_state.current_image_hash = file_hash
            
            # Load model if not already loaded
            if not st.session_state.model_loaded:
                with st.spinner("🔄 Model yükleniyor..."):
                    try:
                        model = get_model()
                        st.session_state.model = model
                        st.session_state.model_loaded = True
                        st.success("✅ Model başarıyla yüklendi!")
                    except Exception as e:
                        st.error(f"❌ Model yüklenemedi: {str(e)}")
                        st.stop()

            # Run prediction with real model
            if st.session_state.model_loaded and st.session_state.model and not st.session_state.analysis_complete:
                with st.spinner("🔍 Görüntü analiz ediliyor..."):
                    try:
                        result = st.session_state.model.predict(image)
                        disease = result['class']
                        confidence = result['confidence'] * 100  # Convert to percentage
                        
                        st.session_state.image_analysis = {"disease": disease, "confidence": confidence}

                        # Push classifier result as assistant message
                        diagnosis_msg = (
                            f"📊 **Tahmini Teşhis:** {disease}\n"
                            f"🔎 **Model Güveni:** {confidence:.2f}%\n\n"
                            "Bu sonuca göre size nasıl yardımcı olabilirim? Tedavi seçenekleri, önleyici tedbirler veya bu durum hakkında merak ettiğiniz konuları sorabilirsiniz."
                        )
                        st.session_state.messages.append({"role": "assistant", "content": diagnosis_msg})
                        st.session_state.analysis_complete = True
                        st.success("✅ Analiz tamamlandı!")
                        
                        # Force page refresh to show the new message
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Analiz sırasında hata oluştu: {str(e)}")
        else:
            # Same image, show that analysis is already done
            if st.session_state.analysis_complete:
                st.info("Bu görüntü zaten analiz edildi. Sonuçları aşağıdaki sohbet alanında görebilirsiniz.")

# ──────────────────────────  CHAT AREA  ───────────────────────────
with st.container():
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    
    # Debug info (can be removed later)
    if st.session_state.messages:
        st.caption(f"💬 {len(st.session_state.messages)} mesaj mevcut")

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    prompt = st.chat_input("Sorunuzu yazın ve Enter'a basın…")

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate LLM response with context
        with st.spinner("💭 Yanıt hazırlanıyor..."):
            try:
                # Prepare context for LLM
                if st.session_state.image_analysis:
                    disease_name = st.session_state.image_analysis["disease"]
                    confidence = st.session_state.image_analysis["confidence"] / 100  # Convert back to decimal
                    
                    # Create contextual prompt
                    contextual_prompt = f"""Kullanıcı sorusu: {prompt}

Bağlam: Az önce analiz edilen fotoğrafta {disease_name} tespit edildi (güven: {confidence:.2%}).

Bu bağlamda kullanıcının sorusunu TÜRKÇE yanıtla. Eğer soru analizle ilgiliyse bu sonucu referans al. 

ÖNEMLİ: Yanıtın tamamen Türkçe olsun, İngilizce kelime kullanma."""
                    
                    assistant_reply = get_llm_response(contextual_prompt, confidence)
                else:
                    # No image analysis context, general dermatology assistance
                    general_prompt = f"""Kullanıcı sorusu: {prompt}

Cilt sağlığı konusunda genel bilgilendirme yap. 

ÖNEMLİ: Yanıtın tamamen Türkçe olsun, İngilizce kelime kullanma."""
                    
                    assistant_reply = get_llm_response(general_prompt, None)
                    
            except Exception as e:
                assistant_reply = f"⚠️ Yanıt oluşturulurken bir hata oluştu: {str(e)}\n\nLütfen sorunuzu yeniden deneyin."

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        # Re‑render last two messages so user sees immediate feedback
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────  DISCLAIMER  ───────────────────────────
st.markdown(
    """
    <div style="margin-top:2.5rem; padding:0.75rem 1rem; border-radius:8px; background:#38340955; font-size:0.9rem;">
        <strong>⚠️ Önemli Not:</strong> Bu uygulama bir tıbbi tanı aracı değildir. Sonuçlar yalnızca bilgilendirme amaçlıdır;
        kesin tanı ve tedavi için mutlaka bir dermatoloji uzmanına başvurunuz.
    </div>
    """,
    unsafe_allow_html=True,
)

