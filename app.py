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
        "model": "google/gemma-3-12b-it:free",
        #deepseek/deepseek-r1-0528-qwen3-8b:free
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
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif !important;
    direction: rtl;
    background-color: #f5f5f5;
    color: #212121;
    margin: 0;
    padding: 0;
}

.stApp {
    background-color: #f5f5f5;
}

.block-container {
    max-width: 800px;
    padding: 1rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.header-container {
    text-align: center;
    margin-bottom: 2rem;
}

.header-container h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #2e7d32;
    margin-bottom: 0.3rem;
}

.header-container p {
    font-size: 1.1rem;
    color: #4e4e4e;
    margin-top: 0;
}

.stTextArea textarea {
    text-align: right;
    font-size: 16px;
    padding: 1rem;
    border-radius: 10px;
    background: #ffffff;
    border: 1px solid #cccccc;
}

.stButton > button {
    background-color: #4caf50;
    color: white;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    font-size: 16px;
    font-weight: bold;
    border: none;
    width: 100%;
}

.stButton > button:hover {
    background-color: #388e3c;
}

.answer-container {
    background-color: #ffffff;
    border-left: 5px solid #4caf50;
    padding: 1.5rem;
    margin-top: 2rem;
    border-radius: 10px;
    font-size: 16px;
    line-height: 1.8;
    color: #212121;
}
.context-container {
    background: #fff;
    padding: 1rem;
    margin-top: 1rem;
    border-radius: 10px;
    border: 1px solid #ddd;
}

.context-item {
    margin-bottom: 0.5rem;
    line-height: 1.6;
}
@media (prefers-color-scheme: dark) {
    html, body, .stApp {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    .stTextArea textarea {
        background-color: #2c2c2c;
        color: #f0f0f0;
        border: 1px solid #555;
    }
    .stButton > button {
        background-color: #66bb6a;
        color: black;
    }
    .answer-container {
        background-color: #2a2a2a;
        color: #f0f0f0;
        border-left: 5px solid #81c784;
    }
}            
</style>
""", unsafe_allow_html=True)
                                                                                                                                                                                                                                                            



# === Centered Title with Markdown ===
st.markdown("""
<div class="header-container">
    <h1>اداره کل دامپزشکی استان اصفهان</h1>
    <p style="font-weight: bold; font-size: 1.2rem; margin-top: 1rem;">دستیار هوشمند</p>
</div>
""", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
# === UI Elements ===
query = st.text_area("✍️ چه سوالی درباره دامپزشکی داری؟", height=120)

if st.button(" پرسیدن سوال", use_container_width=True) and query.strip():
    with st.spinner("🔍 در حال پردازش..."):
        retrieved = search_faiss_index(query, embedder, index, documents, k=TOP_K)
        context_text = "\n\n".join([r["content"] for r in retrieved])
        answer = call_llm(query, context_text)
        if answer:
            clean_answer = answer.replace("```", "").replace("---", "").strip()
            st.markdown("""
            <div class="answer-container">
                <div class="answer-title" style="font-weight: bold; margin-bottom: 1rem;">✅ پاسخ:</div>        
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

