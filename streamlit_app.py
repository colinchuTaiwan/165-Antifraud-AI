import os
import time
import streamlit as st
import chromadb
from google import genai
from google.genai import types
import random

# =========================
# 1. API 初始化（改 Cloud 寫法）
# =========================
@st.cache_resource
def get_genai_client():
    api_key = st.secrets.get("GEMINI_API_KEY")  # ✅ 改這裡
    if not api_key:
        st.error("請在 Streamlit Secrets 設定 GEMINI_API_KEY")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_genai_client()

GEN_MODEL_ID = "gemini-1.5-flash"   # ✅ 穩定版
EMBED_MODEL_ID = "text-embedding-004"

# =========================
# 2. 安全 API 呼叫
# =========================
def safe_api_call(call_type, **kwargs):
    max_retries = 5

    for i in range(max_retries):
        try:
            if call_type == 'embed':
                res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=kwargs['text']
                )
                return res.embeddings[0].values

            elif call_type == 'generate':
                return client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=kwargs['prompt'],
                )

        except Exception as e:
            wait_time = min(2 ** (i + 2), 30) + random.uniform(0, 1)
            st.warning(f"API 錯誤，重試中 ({i+1}/5)...")
            time.sleep(wait_time)

    st.error("❌ API 多次失敗")
    st.stop()


# =========================
# 3. Chroma（改為 In-Memory 避免爆）
# =========================
@st.cache_resource
def get_vector_db():
    return chromadb.Client()   # ❗ 不用 Persistent


# =========================
# 4. UI
# =========================
st.set_page_config(
    page_title="165 防詐分析系統",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 165 智慧防詐分析系統（Cloud版）")

user_input = st.text_area(
    "請輸入可疑訊息：",
    height=150
)

if st.button("🔍 分析"):
    if not user_input.strip():
        st.warning("請輸入內容")
        st.stop()

    with st.spinner("分析中..."):

        query_vec = safe_api_call('embed', text=user_input)

        prompt = f"""
你是一位資深防詐專家，分析以下內容：

{user_input}

請輸出：
1. 詐騙類型
2. 判斷理由
3. 關鍵風險
4. 建議行動
"""

        res = safe_api_call('generate', prompt=prompt)

        st.markdown("## 💡 分析結果")
        st.write(res.text)

st.caption("⚠️ 僅供參考")
