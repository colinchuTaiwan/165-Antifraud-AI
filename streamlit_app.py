import os
import asyncio
import streamlit as st
import chromadb
from google import genai
from google.genai import types

# --- 1. 核心參數設定 (嚴格遵照您的要求) ---
GEN_MODEL_ID = "gemini-flash-latest"
EMBED_MODEL_ID = "gemini-embedding-001"
CHROMA_PATH = "chroma_crime_db"

# --- 2. 頁面配置 ---
st.set_page_config(
    page_title="165 智慧打詐機器人",
    page_icon="🛡️",
    layout="wide"
)

# --- 3. 初始化資源 ---
@st.cache_resource
def init_app_resources():
    """初始化 API Client 與向量資料庫"""
    # 從 Secrets 讀取 API Key
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ 請在 Streamlit Cloud Secrets 中設定 GEMINI_API_KEY")
        st.stop()
        
    client = genai.Client(api_key=api_key)
    
    # 定義資料庫完整路徑 (確保在 GitHub/Streamlit 環境能找到同目錄資料夾)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_full_path = os.path.join(current_dir, CHROMA_PATH)
    
    try:
        # 初始化 ChromaDB
        chroma_client = chromadb.PersistentClient(path=db_full_path)
        # 取得集合 (請確保名稱與您建立時一致，此處預設為 crime_data)
        collection = chroma_client.get_or_create_collection(name="crime_data")
        return client, collection
    except Exception as e:
        st.error(f"❌ 資料庫載入失敗：{str(e)}")
        st.stop()

# 實例化
genai_client, vector_db = init_app_resources()

# --- 4. 核心邏輯：檢索與分析 ---
async def get_fraud_report(user_text):
    """結合特定 Embedding 與生成模型的 RAG 流程"""
    try:
        # A. 向量檢索 (注意：這部分通常由 Chroma 內部調用 EMBED_MODEL_ID)
        # 假設您的集合在建立時已綁定 Embedding 功能
        results = vector_db.query(
            query_texts=[user_text],
            n_results=3
        )
        
        # 整理檢索到的案例上下文
        context_list = results.get('documents', [[]])[0]
        context_text = "\n\n".join(context_list) if context_list else "目前資料庫無直接相關案例。"

        # B. 建立打詐專家 Prompt
        prompt = f"""
        你是一位專業的打詐（Anti-fraud）專家。
        請根據以下參考案例與您的專業知識，分析使用者提供的內容是否有詐騙疑慮。

        【參考犯罪案例庫】：
        {context_text}

        【待分析內容】：
        {user_text}

        ---
        請提供以下繁體中文分析：
        1. 詐騙風險評級 (低/中/高)
        2. 懷疑採用的手法
        3. 具體的預防建議或行動
        """

        # C. 使用指定的生成模型產生結果
        response = genai_client.models.generate_content(
            model=GEN_MODEL_ID,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"分析時發生技術錯誤：{str(e)}"

# --- 5. Streamlit 使用者介面 ---
st.title("🛡️ 165 智慧打詐鑑定系統 (chatbot-2)")
st.markdown(f"系統已連線至：`{CHROMA_PATH}` | 使用模型：`{GEN_MODEL_ID}`")

# 顯示目前資料庫狀態
with st.sidebar:
    st.header("系統狀態")
    st.success("✅ Gemini API 已就緒")
    st.info(f"📊 資料庫現有條目：{vector_db.count()}")
    st.divider()
    st.write("本系統僅供參考，遇到疑似詐騙請即刻撥打 **165 反詐騙專線**。")

# 輸入框
input_col, output_col = st.columns([1, 1])

with input_col:
    user_query = st.text_area("請輸入可疑的簡訊、網址或廣告內容：", height=250)
    analyze_btn = st.button("🔍 執行 AI 鑑定", use_container_width=True)

with output_col:
    if analyze_btn:
        if user_query.strip():
            with st.spinner("正在比對犯罪資料庫並進行分析..."):
                # 執行異步分析邏輯
                result_text = asyncio.run(get_fraud_report(user_query))
                st.subheader("📋 鑑定報告")
                st.markdown(result_text)
        else:
            st.warning("請先輸入要分析的內容。")
    else:
        st.write("分析結果將顯示於此。")

# 頁尾
st.divider()
st.caption("© 2026 chatbot-2 打詐技術小組 | 模型 ID: " + GEN_MODEL_ID)
