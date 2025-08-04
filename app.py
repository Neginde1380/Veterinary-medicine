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
    background: #4caf50; /* رنگ ساده بدون گرادینت */
    padding: 1.5rem 1rem;
    border-radius: 0 0 20px 20px;
    margin: -1rem -1rem 2rem -1rem;
    text-align: center;
    box-shadow: none; /* حذف سایه */
    position: relative;
}
.header-container::before {
    content: none;  /* حذف انیمیشن بک‌گراند */
}

/* Main content card */
.main-card {
    background: #ffffff;
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: none; /* حذف افکت سایه */
    border: 1px solid #e0e0e0;
}
/* Input styling */
.stTextArea textarea {
    background: #f9f9f9 !important;
    border: 1px solid #ccc !important;
    box-shadow: none !important;
}

.stButton > button {
    background: #4caf50 !important;
    box-shadow: none !important;
    transition: none !important;
}
.stButton > button:hover {
    transform: none !important;
    box-shadow: none !important;
}
.answer-container {
    background: #fdfdfd;
    border: 1px solid #4caf50;
    box-shadow: none;
}
.context-container {
    background: #f7f7f7;
    border: 1px solid #d0d0d0;
    box-shadow: none;
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

