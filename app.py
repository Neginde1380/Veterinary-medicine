from retriever import load_faiss_index, load_documents, search_faiss_index
from sentence_transformers import SentenceTransformer
import requests
import json
import time
import streamlit as st


# === CONFIG ===
FAISS_INDEX_PATH = "bge_m3_faiss.index"
DOCUMENTS_PATH = "bge_m3_documents.json"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# api_key = "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"
# apexion-ai/Nous-1-2B  طول کشید تا جواب داد 
api_key = st.secrets["OPENROUTER_API_KEY"]                                                                                                                                                     
TOP_K = 1  # Number of retrieved passages to include

# === Load models and data ===
@st.cache_resource
def load_assets():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index = load_faiss_index(FAISS_INDEX_PATH)
    documents = load_documents(DOCUMENTS_PATH)
    return embedder, index, documents

embedder, index, documents = load_assets()

def call_llm(query, context, max_retries=3):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",  
        "Content-Type": "application/json",
    }

    data = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
        "messages": [
            {
                "role": "system",
                "content": (
                  "You are a Persian assistant to answer veterinary questions."
                  " Answer the question based on the retrieved information."
                ),
            },
            {
                "role": "user",
                "content":f"Question: {query}\n\n Retrieved Information:\n{context}\n\n Answer:" 
            },
        ]
    }
    for attempt in range(1, max_retries + 1):
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        elif response.status_code == 429:
            wait_time = 10 * attempt
            print(f"⏳ Rate limited. Waiting {wait_time} seconds before retry ({attempt}/{max_retries})...")
            time.sleep(wait_time)
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            break
    return None


# === Streamlit UI ===
st.set_page_config(page_title="اداره کل دامپزشکی استان اصفهان", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap');
/* Remove default Streamlit padding and margin */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 800px !important;
}
/* Hide Streamlit menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
html, body, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif !important;
    direction: rtl;
    background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 50%, #f9fbe7 100%);
    margin: 0;
    padding: 0;
}
.stApp {
    background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 50%, #f9fbe7 100%);
    min-height: 100vh;
}
/* Header Section with gradient and icon */
.header-container {
    background: linear-gradient(135deg, #2e7d32 0%, #388e3c 50%, #4caf50 100%);
    padding: 2rem 1rem;
    border-radius: 0 0 25px 25px;
    margin: -1rem -1rem 2rem -1rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(46, 125, 50, 0.3);
    position: relative;
    overflow: hidden;
}
.header-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="rgba(255,255,255,0.1)"/></svg>') repeat;
    animation: float 20s linear infinite;
@keyframes float {
    0% { transform: translateX(-100px) translateY(-100px); }
    100% { transform: translateX(100px) translateY(100px); }
}
.header-title {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
}

.header-subtitle {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.3rem !important;
    font-weight: 400 !important;
    margin: 0.5rem 0 0 0 !important;
    position: relative;
    z-index: 1;
}
.header-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
}
/* Main content card */
.main-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    margin-bottom: 2rem;
}
/* Input styling */
.stTextArea textarea {
    text-align: right !important;
    font-size: 16px !important;
    padding: 20px !important;
    border-radius: 15px !important;
    background: rgba(255, 255, 255, 0.9) !important;
    border: 2px solid #e0e0e0 !important;
    color: #2c2c2c !important;
    font-family: 'Vazirmatn', sans-serif !important;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.stTextArea textarea:focus {
    border-color: #4caf50 !important;
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1) !important;
}
/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%) !important;
    color: white !important;
    border-radius: 15px !important;
    padding: 16px 40px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    font-family: 'Vazirmatn', sans-serif !important;
    border: none !important;
    box-shadow: 0 8px 20px rgba(46, 125, 50, 0.3) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 25px rgba(46, 125, 50, 0.4) !important;
}
/* Answer section */
.answer-container {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border-radius: 15px;
    padding: 2rem;
    margin: 1.5rem 0;
    border-left: 5px solid #4caf50;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    font-size: 16px;
    line-height: 1.8;
    color: #2c2c2c;
}
.answer-title {
    color: #2e7d32 !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    margin-bottom: 1rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
/* Context section */
.context-container {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 15px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    border: 1px solid rgba(76, 175, 80, 0.2);
}
.context-item {
    background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e8 100%);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    border-left: 3px solid #4caf50;
    font-size: 14px;
    line-height: 1.6;
}
/* Loading spinner */
.stSpinner > div {
    text-align: center;
    color: #4caf50 !important;
}
/* Info box */
.stInfo {
    background: linear-gradient(135deg, #e3f2fd 0%, #f1f8e9 100%) !important;
    border-radius: 15px !important;
    border-left: 5px solid #4caf50 !important;
    padding: 1.5rem !important;
    font-size: 16px !important;
}
/* Checkbox */
.stCheckbox {
    margin-top: 1rem;
}
.stCheckbox label {
    font-size: 15px !important;
    color: #2e7d32 !important;
    font-weight: 500 !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .header-title {
        font-size: 2rem !important;
    }
    
    .header-subtitle {
        font-size: 1.1rem !important;
    }
    
    .main-card {
        padding: 1.5rem;
        margin: 0 0.5rem;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .main-card, .answer-container, .context-container {
        background: rgba(30, 30, 30, 0.95) !important;
        color: #e0e0e0 !important;
    }
    
    .stTextArea textarea {
        background: rgba(40, 40, 40, 0.9) !important;
        color: #f0f0f0 !important;
        border-color: #555 !important;
    }
}
</style>
""", unsafe_allow_html=True)
                                                                                                                                                                                                                                                            



# === Centered Title with Markdown ===
st.markdown("""
<div class="header-container">
    <div class="header-icon">🏥</div>
st.markdown("<h1>اداره کل دامپزشکی استان اصفهان</h1>", unsafe_allow_html=True)
<p class="header-subtitle">دستیار هوشمندن</p>
</div>
""", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
# === UI Elements ===
query = st.text_area("✍️ چه سوالی درباره دامپزشکی داری؟", height=120)

if st.button("🚀 پرسیدن سوال", use_container_width=True) and query.strip():
    with st.spinner("🔍 در حال پردازش..."):
        retrieved = search_faiss_index(query, embedder, index, documents, k=TOP_K)
        context_text = "\n\n".join([r["content"] for r in retrieved])
        answer = call_llm(query, context_text)
        if answer:
            clean_answer = answer.replace("```", "").replace("---", "").strip()
            st.markdown("""
            <div class="answer-container">
                <div class="answer-title">✅ پاسخ:</div>
                    <div>{}</div>
            </div>
            """.format(clean_answer), unsafe_allow_html=True)
        else:
            st.error(f"❌ API Error {answer.status_code}: {answer.text}")

    show_context = st.checkbox("📚 نمایش منابع و مراجع استفاده شده")
    if show_context:
        st.markdown('<div class="context-container">', unsafe_allow_html=True)
        st.markdown("**📖 منابع مورد استفاده:**")
        for i, doc in enumerate(retrieved, 1):
            st.markdown(f'<div class="context-item"><strong>{i}.</strong> {doc["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #666;">
    <small>دستیار هوشمند دامپزشکی استان اصفهان |نسخه آزمایشی </small>
</div>
""", unsafe_allow_html=True)    

