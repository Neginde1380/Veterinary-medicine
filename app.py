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
st.set_page_config(page_title="اداره کل دامپزشکی استان اصفهان", layout="centered")

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
    color: var(--text-color, #212121);
    background-color: var(--background-color, #f9fbe7);
}

.stApp {
    padding: 2rem;
    background: var(--background-color, #f9fbe7);
}

.stTextInput input, .stTextArea textarea {
    text-align: right !important;
    font-size: 18px !important;
    padding: 16px !important;
    border-radius: 10px;
    background-color: rgba(255, 255, 255, 0.85) !important;
    color: #212121 !important;
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

h1, h3 {
    text-align: center;
    font-family: 'Vazirmatn', sans-serif !important;
    margin-bottom: 1rem;
}

h1 {
    font-size: 36px;
    color: #2e7d32;
}

h3 {
    font-size: 22px;
    color: #558b2f;
    margin-bottom: 2rem;
}

.stTextArea, .stCheckbox, .stButton, .stMarkdown {
    margin-bottom: 1.5rem;
}

.stMarkdown {
    font-size: 18px !important;
    line-height: 1.9 !important;
    background-color: rgba(255, 255, 255, 0.85);
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 0 6px rgba(0, 0, 0, 0.1);
    color: #212121 !important;
}

/* Dark mode override (Streamlit doesn’t natively support CSS media queries, but we do this in safe way) */
@media (prefers-color-scheme: dark) {
    html, body, .stApp {
        background-color: #121212 !important;
        color: #e0e0e0 !important;
    }
    .stTextInput input, .stTextArea textarea, .stMarkdown {
        background-color: #1e1e1e !important;
        color: #f5f5f5 !important;
    }
}
</style>
""", unsafe_allow_html=True)


# === Centered Title with Markdown ===
st.markdown("<h1>اداره کل دامپزشکی استان اصفهان</h1>", unsafe_allow_html=True)
st.markdown("<h3>دستیار هوشمند</h3>", unsafe_allow_html=True)


# === UI Elements ===
query = st.text_area("✍️ چه سوالی درباره دامپزشکی داری؟", height=140)

if st.button("پرسیدن سوال") and query.strip():
    with st.spinner("🔍 در حال پردازش..."):
        retrieved = search_faiss_index(query, embedder, index, documents, k=TOP_K)
        context_text = "\n\n".join([r["content"] for r in retrieved])
        answer = call_llm(query, context_text)
        if answer:
            clean_answer = answer.replace("```", "").replace("---", "").strip()
            st.markdown("### ✅ پاسخ:")
            st.markdown(
                f"<div style='background-color:rgba(255,255,255,0.85); color:#212121; padding:16px; border-radius:10px;'>{clean_answer}</div>",
                unsafe_allow_html=True
            )
        else:
            st.error(f"❌ API Error {answer.status_code}: {answer.text}")

    show_context = st.checkbox("نمایش متن‌های مرتبط بازیابی‌شده")
    if show_context:
        st.markdown("---")
        st.markdown("### 📚 متون بازیابی‌شده:")
        for i, doc in enumerate(retrieved, 1):
            st.markdown(f"**{i}.** {doc['content']}")
else:
    st.info("منتظر سوال شما هستم...")

