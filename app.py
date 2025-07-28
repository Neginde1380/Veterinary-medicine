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
api_key = st.secrets["api_key"]                                                                                                                                                     
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
st.set_page_config(page_title="چت‌بات دامپزشکی", page_icon="🐾")

# === Persian Font & Styling ===
st.markdown("""
    <style>
    @font-face {
        font-family: 'Vazirmatn';
        src: url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@latest/dist/webfonts/Vazirmatn-Regular.woff2') format('woff2');
        font-weight: normal;
    }

    @font-face {
        font-family: 'Vazirmatn';
        src: url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@latest/dist/webfonts/Vazirmatn-Bold.woff2') format('woff2');
        font-weight: bold;
    }

    html, body, [class*="css"] {
        font-family: 'Vazirmatn', sans-serif !important;
        direction: rtl;
        font-size: 18px;
    }

    .stApp {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9, #ffffff);
        padding: 2rem;
    }

    .stTextInput input, .stTextArea textarea {
        text-align: right !important;
        font-size: 18px !important;
        padding: 16px !important;
        border-radius: 10px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%) !important;
        color: white !important;
        border-radius: 10px;
        padding: 14px 32px !important;
        font-size: 17px !important;
        font-weight: bold;
        font-family: 'Vazirmatn', sans-serif !important;
    }

    h1 {
        color: #1b5e20 !important;
        text-align: center;
        font-size: 36px !important;
        margin-bottom: 2rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .stTextArea, .stCheckbox, .stButton, .stMarkdown {
        margin-bottom: 1.5rem;
    }

    .stMarkdown {
        font-size: 18px !important;
        line-height: 1.9 !important;
    }
    </style>
""", unsafe_allow_html=True)

# === Centered Title with Markdown ===
st.markdown("<h1>🐾 دستیار هوشمند دامپزشکی</h1>", unsafe_allow_html=True)

# === UI Elements ===
query = st.text_area("✍️ چه سوالی درباره دامپزشکی داری؟", height=120)
show_context = st.checkbox("نمایش متن‌های مرتبط بازیابی‌شده")

if st.button("پرسیدن سوال") and query.strip():
    with st.spinner("🔍 در حال پردازش..."):
        retrieved = search_faiss_index(query, embedder, index, documents, k=TOP_K)
        context_text = "\n\n".join([r["content"] for r in retrieved])
        answer = call_llm(query, context_text)
        if answer:
           clean_answer = answer.replace("```", "").replace("---", "").strip()
           st.success(f"✅ پاسخ:\n\n{clean_answer}")
        else:
            st.error(f"❌ API Error {answer.status_code}: {answer.text}")

        if show_context:
            st.markdown("---")
            st.markdown("### 📚 متون بازیابی‌شده:")
            for i, doc in enumerate(retrieved, 1):
                st.markdown(f"**{i}.** {doc['content']}")
else:
    st.info("منتظر سوال شما هستم...")
