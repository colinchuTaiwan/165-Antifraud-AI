import os
import json
import time
import streamlit as st
from google import genai
from google.genai import types

# =========================
# 1. API 初始化
# =========================
@st.cache_resource
def get_genai_client():
    # 優先從 Streamlit Secrets 讀取 (部署在 streamlit.io 時使用)
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        # 本地測試時從環境變數讀取
        api_key = os.getenv("GEMINI_API_KEY")
        
    if not api_key:
        st.error("❌ 請設定 GEMINI_API_KEY (Secrets 或環境變數)")
        st.stop()
    return genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

client = get_genai_client()
# 使用您指定的 Flash 模型，處理長筆錄速度最快
GEN_MODEL_ID = "gemini-flash-latest"

# =========================
# 2. 核心提取邏輯
# =========================
def extract_report_to_json(text):
    """將雜亂的筆錄文字精確提取為結構化 JSON"""
    # 根據您提供的圖片欄位精確定義 Prompt
    prompt = f"""
    你是一位專業的刑事紀錄員。請將以下「報案筆錄」中的關鍵個資提取為 JSON 格式，以便自動填入行政表格。
    
    【JSON 欄位對應】：
    - 姓名 (name)
    - 出生年月日 (birth_date, 格式: YYYY-MM-DD)
    - 性別 (gender)
    - 身分證字號 (id_number)
    - 教育程度 (education)
    - 職業 (occupation)
    - Email (email)
    - 聯絡電話 (phone)
    - 現居地址 (current_address)
    - 戶籍地址 (permanent_address)

    【規則】：
    1. 若筆錄未提及，請填寫 "未提供"。
    2. 僅輸出 JSON 內容，禁止包含 ```json 等任何 Markdown 標記。
    3. 確保地址完整性（含縣市、區、路名）。

    【待解析筆錄內容】：
    {text}
    """
    
    try:
        response = client.models.generate_content(
            model=GEN_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # 設為最低溫，確保結果一致且不亂編
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ]
            )
        )
        # 清理並解析 JSON
        clean_res = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_res)
    except Exception as e:
        st.error(f"解析發生錯誤：{str(e)}")
        return None

# =========================
# 3. Streamlit UI 介面
# =========================
st.set_page_config(page_title="筆錄自動填表助理", page_icon="📝")
st.title("📝 刑事筆錄資料提取工具")
st.markdown("將筆錄文字直接轉化為 **JSON 格式**，方便自動化填入表格系統。")

# 輸入框：放寬高度方便貼上長筆錄
raw_report = st.text_area("請輸入或貼上初步筆錄文字：", height=400)

if st.button("🚀 執行自動提取", use_container_width=True):
    if raw_report.strip():
        with st.spinner("AI 正在辨識關鍵字並整理結構..."):
            extracted_json = extract_report_to_json(raw_report)
            
            if extracted_json:
                st.subheader("✅ 提取完成 (JSON 結果)")
                # 顯示美化過的 JSON
                st.json(extracted_json)
                
                # 提供下載與快速複製
                json_str = json.dumps(extracted_json, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 下載 JSON 檔案",
                    data=json_str.encode('utf-8'),
                    file_name="record_fill.json",
                    mime="application/json"
                )
                
                st.success("提取成功！您可以將此 JSON 用於後端填表腳本。")
    else:
        st.warning("請先輸入筆錄內容。")

st.divider()
st.caption("© 2026 colinchuTaiwan | 本系統僅供刑事辦案行政輔助使用")
