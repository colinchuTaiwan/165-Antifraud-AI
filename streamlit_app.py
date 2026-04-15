import os
import time
import random
import streamlit as st
import chromadb
from google import genai
from google.genai import types

# =========================
# 1. 初始化與路徑設定
# =========================
# 取得目前程式碼所在的目錄，確保在 Streamlit Cloud 讀取 chroma_crime_db 不會出錯
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(CURRENT_DIR, "chroma_crime_db")

@st.cache_resource
def get_genai_client():
    # 改用 st.secrets 讀取，符合 Streamlit Cloud 規範
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ 請在 Streamlit Cloud 的 Settings -> Secrets 中設定 GEMINI_API_KEY")
        st.stop()
    # 鎖定 v1beta 版本以支援指定的模型 ID
    return genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

client = get_genai_client()

# 鎖定您指定的模型 ID
GEN_MODEL_ID = "gemini-flash-latest"
EMBED_MODEL_ID = "gemini-embedding-001"

def safe_api_call(call_type, **kwargs):
    """具備指數退避與隨機抖動機制的 API 呼叫"""
    max_retries = 5
    for i in range(max_retries):
        try:
            if call_type == 'embed':
                res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=kwargs['text'],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                return res.embeddings[0].values
            
            elif call_type == 'generate':
                return client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=kwargs['prompt'],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                        ]
                    )
                )
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in [429, 503, 500]:
                if i == max_retries - 1:
                    st.error("🚫 API 配額已耗盡或伺服器持續無回應。請稍後再試。")
                    st.stop()
                
                wait_time = min(2 ** (i + 2), 32) + random.uniform(0, 1)
                error_type = {429: "配額用完", 503: "服務忙碌", 500: "伺服器錯誤"}.get(status, "API 錯誤")
                st.toast(f"⏳ {error_type}！第 {i+1} 次重試將於 {int(wait_time)} 秒後開始...", icon="⚠️")
                time.sleep(wait_time)
                continue
            
            st.error(f"❌ 發生不可預期的 API 錯誤：{str(e)}")
            st.stop()
    return None

# =========================
# 2. 核心解析邏輯
# =========================
def parse_cases_from_doc(raw_text):
    """確保『案類標題』、『內容』與『特徵』完整組合"""
    lines = raw_text.split('\n')
    processed_cases = []
    current_case = []

    for line in lines:
        line = line.strip()
        if not line: continue
        
        if "【案例內容】" in line:
            pass 
        elif "【核心特徵】" in line:
            current_case.append(line)
            processed_cases.append("\n".join(current_case))
            current_case = []
            continue
            
        current_case.append(line)
    
    if current_case:
        processed_cases.append("\n".join(current_case))
    return processed_cases

@st.cache_resource
def get_vector_db():
    if not os.path.exists(CHROMA_PATH):
        st.error(f"❌ 找不到資料庫目錄：{CHROMA_PATH}。請確保已將 chroma_crime_db 上傳至 GitHub。")
        st.stop()
    return chromadb.PersistentClient(path=CHROMA_PATH)

# =========================
# 3. Streamlit UI
# =========================
st.set_page_config(page_title="165 智慧防詐分析系統", page_icon="🚨", layout="wide")
st.title("🚨 165 智慧防詐分析系統")
user_input = st.text_area("請輸入可疑訊息或對話內容：", height=150, placeholder="例如：收到簡訊說帳戶異常，要點擊連結...")

if st.button("🔍 啟動全方位剖析", use_container_width=True):
    if not user_input.strip():
        st.warning("請輸入內容後再開始分析。")
        st.stop()

    with st.spinner("分析官正在檢索案例並對照防詐教材..."):
        try:
            # A. 向量化
            query_vec = safe_api_call('embed', text=user_input)
            db = get_vector_db()
            
            # B. 檢索「歷史案例庫」
            case_col = db.get_collection("165_cases")
            case_results = case_col.query(query_embeddings=[query_vec], n_results=1)
            
            top_cases_ctx = ""
            all_cases = []
            if case_results['documents'] and len(case_results['documents'][0]) > 0:
                raw_doc = case_results['documents'][0][0]
                all_cases = parse_cases_from_doc(raw_doc)
                top_cases_ctx = "\n\n---\n\n".join(all_cases[:3])
            
            # C. 檢索「防詐教材庫」
            kb_col = db.get_collection("anti_fraud_kb")
            kb_results = kb_col.query(query_embeddings=[query_vec], n_results=2)
            kb_ctx = "\n\n".join(kb_results['documents'][0]) if kb_results['documents'] else "（無可用教材）"

            # D. 生成 AI 報告
            prompt = f"""你是一位資深刑事防詐分析官。請結合『歷史案例』與『官方防詐教材』分析民眾輸入。
【參考歷史案例】:
{top_cases_ctx}
【官方防詐教材知識】:
{kb_ctx}
【民眾輸入內容】:
{user_input}

請依照此結構回覆：
## 💡 刑事分析報告
您好，我是「165 刑事防詐分析官」。針對您所遇到的情況，這是一起典型的詐騙手法。
### 🚩 專家研判
...
### ⚡ 關鍵破綻
...
### 📘 防詐教室
...
### 🛡️ 具體行動建議
...
"""
            response = safe_api_call('generate', prompt=prompt)

            if response:
                st.subheader("💡 綜合分析報告")
                st.markdown(response.text)
                
                # E. 底部參考面板
                st.divider()
                st.subheader("📌 歷史案例 (Top 5)")
                for idx, case_text in enumerate(all_cases[:5]):
                    with st.expander(f"🏆 案例 {idx+1}", expanded=(idx==0)):
                        st.info(case_text)
            
        except Exception as e:
            st.error(f"系統執行錯誤: {e}")

st.divider()
st.caption("⚠️ 分析結果僅供參考，如有疑慮請撥打 165 反詐騙專線。")
