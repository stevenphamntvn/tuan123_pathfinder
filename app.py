# file: app.py
# Phiên bản hoàn chỉnh: Giao diện tinh gọn, hiển thị chi phí ở góc trái.

# --- PHẦN SỬA LỖI QUAN TRỌNG CHO STREAMLIT CLOUD ---
# Ba dòng này phải nằm ở ngay đầu file
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------------------

import streamlit as st
import chromadb
import google.generativeai as genai
import os
import requests
import zipfile
from io import BytesIO

# --- PHẦN CẤU HÌNH ---
# API Key của bạn từ Google Cloud
GOOGLE_API_KEY = 'AIzaSyBOAgpJI1voNNxeOC6sS7y01EJRXWSK0YU' # !!! THAY API KEY CỦA BẠN VÀO ĐÂY !!!

# --- CẤU HÌNH TRIỂN KHAI ONLINE ---
# !!! QUAN TRỌNG: Dán đường dẫn tải trực tiếp file zip của bạn vào đây
DB_ZIP_URL = "https://drive.google.com/uc?export=download&id=1WpTztD-D21zN5fyXxtS7QPz5kFxJ9AIG"
DB_PATH = 'chroma_db'
COLLECTION_NAME = 'collection'

# --- BẢNG GIÁ VÀ LỰA CHỌN MÔ HÌNH ---
MODEL_PRICING = {
    "gemini-1.5-flash-latest": {
        "input": 0.35,
        "output": 1.05
    },
    "gemini-1.5-pro-latest": {
        "input": 3.50,
        "output": 10.50
    }
}
MODEL_OPTIONS = list(MODEL_PRICING.keys())

# --- TỶ GIÁ VÀ CÁC VAI TRÒ (PERSONA) CHO AI ---
USD_TO_VND_RATE = 25500  # Tỷ giá USD/VND (bạn có thể cập nhật)
PERSONAS = {
    "Tướng quân Chỉ đạo": "Bạn là một Tướng quân của Tuấn 123, đưa ra các chỉ dẫn, quy trình một cách dứt khoát, rõ ràng và đầy năng lượng.",
    "Chuyên gia Đào tạo": "Bạn là một chuyên gia đào tạo thân thiện, giải thích các tình huống, kỹ năng cho chuyên viên, chuyên gia một cách chi tiết, dễ hiểu, kèm theo ví dụ thực tế."
}
PERSONA_OPTIONS = list(PERSONAS.keys())


# --- HÀM TẢI VÀ GIẢI NÉN DATABASE ---
def setup_database():
    """Kiểm tra, tải về và giải nén database nếu cần."""
    if not os.path.exists(DB_PATH):
        st.info(f"Không tìm thấy database '{DB_PATH}'. Bắt đầu tải về từ cloud...")
        st.warning("Quá trình này chỉ diễn ra một lần và có thể mất vài phút.")
        
        if not DB_ZIP_URL or DB_ZIP_URL == "YOUR_DIRECT_DOWNLOAD_LINK_TO_THE_DB_ZIP_FILE":
            st.error("Lỗi cấu hình: Vui lòng cung cấp DB_ZIP_URL trong file app.py.")
            return False

        try:
            with st.spinner('Đang tải database...'):
                response = requests.get(DB_ZIP_URL)
                response.raise_for_status()
            with st.spinner('Đang giải nén...'):
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    z.extractall('.')
            st.success("Thiết lập database thành công! Đang tải lại...")
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi khi tải hoặc giải nén database: {e}")
            return False
    return True

# --- KHỞI TẠO DATABASE ---
@st.cache_resource
def load_db():
    """Kết nối tới database và trả về collection."""
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        st.error(f"Lỗi kết nối database: {e}")
        return None

# --- HÀM LOGIC XỬ LÝ CÂU HỎI ---
def get_ai_response(question, model, collection, model_name, system_instruction):
    """Lấy câu trả lời từ AI và thông tin sử dụng."""
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n---\n\n".join(results['documents'][0])
    
    prompt = f"""{system_instruction}

    Dựa vào thông tin tham khảo được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng.
    
    Thông tin tham khảo: {context}
    
    Câu hỏi: {question}"""
    
    response = model.generate_content(prompt)
    usage_info = None
    try:
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count
        response_tokens = usage.candidates_token_count
        
        price_input = MODEL_PRICING[model_name]["input"]
        price_output = MODEL_PRICING[model_name]["output"]
        
        input_cost = (prompt_tokens / 1_000_000) * price_input
        output_cost = (response_tokens / 1_000_000) * price_output
        total_cost_usd = input_cost + output_cost
        
        usage_info = {
            "cost_vnd": total_cost_usd * USD_TO_VND_RATE
        }
    except Exception:
        pass

    return response.text, usage_info

# --- GIAO DIỆN NGƯỜI DÙNG STREAMLIT ---
st.set_page_config(page_title="Pathfinder - Trợ lý Tuấn 123", page_icon="🌿")
st.title(" Pathfinder - Trợ lý Tuấn 123")

# Khởi tạo tổng chi phí trong session state
if 'total_session_cost_vnd' not in st.session_state:
    st.session_state.total_session_cost_vnd = 0.0

# Thanh bên để chọn mô hình và vai trò
with st.sidebar:
    st.header("Cấu hình")
    selected_model_name = st.selectbox(
        "Chọn mô hình AI:",
        options=MODEL_OPTIONS,
        index=0,
    )
    
    selected_persona_name = st.selectbox(
        "Chọn phong cách trả lời:",
        options=PERSONA_OPTIONS,
        index=1 # Mặc định chọn "Lương y trẻ"
    )
    system_instruction = PERSONAS[selected_persona_name]
    

# Bước 1: Đảm bảo database đã sẵn sàng
if setup_database():
    # Bước 2: Khởi tạo AI và DB
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm_model = genai.GenerativeModel(selected_model_name)
        collection = load_db()
    except Exception as e:
        st.error(f"Lỗi khởi tạo AI. Vui lòng kiểm tra API Key. Lỗi: {e}")
        llm_model = None
        collection = None

    # Bước 3: Hiển thị giao diện chat
    if llm_model and collection:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ví dụ: Bệnh Thái Dương là gì?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(f"AI ({selected_model_name}) đang suy nghĩ..."):
                    response_text, usage_info = get_ai_response(prompt, llm_model, collection, selected_model_name, system_instruction)
                    
                    # Chỉ hiển thị câu trả lời, không hiển thị nguồn
                    st.markdown(response_text)
                    
                    if usage_info:
                        # Cập nhật tổng chi phí
                        st.session_state.total_session_cost_vnd += usage_info['cost_vnd']
            
            # Lưu câu trả lời vào lịch sử chat
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # Chạy lại để cập nhật tổng chi phí
            st.rerun()

    # Hiển thị tổng chi phí ở góc dưới bên trái
    # Sử dụng HTML và CSS để định vị
    total_cost_display = f"""
    <div style="
        position: fixed;
        bottom: 10px;
        left: 10px; /* Đã đổi từ right sang left */
        background-color: #f0f2f6;
        padding: 5px 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        font-size: 0.8em;
        z-index: 1000;
        color: #333;
    ">
        API: {st.session_state.total_session_cost_vnd:,.0f} VNĐ
    </div>
    """
    st.markdown(total_cost_display, unsafe_allow_html=True)
