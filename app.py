# file: app.py
# Phiên bản hoàn chỉnh: Cập nhật vai trò mới cho Tuấn 123 Pathfinder.

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
import time

# --- PHẦN CẤU HÌNH ---
st.set_page_config(
    page_title="Pathfinder - Trợ lý Tuấn 123",
    page_icon="🤖",
    layout="wide"
)

# Sử dụng st.secrets để bảo mật API key khi triển khai online
try:
    # Lấy key từ Streamlit secrets (cho môi trường online)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    # Dùng cho local dev nếu không có file secrets.toml
    # !!! THAY THẾ BẰNG API KEY CỦA BẠN ĐỂ CHẠY LOCAL
    GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY_HERE'

# --- CẤU HÌNH TRIỂN KHAI ONLINE ---
# !!! QUAN TRỌNG: Dán đường dẫn tải trực tiếp file zip của bạn vào đây
DB_ZIP_URL = "https://drive.google.com/uc?export=download&id=1WpTztD-D21zN5fyXxtS7QPz5kFxJ9AIG"
DB_PATH = 'chroma_db'
COLLECTION_NAME = 'tuan123_collection'

# --- BẢNG GIÁ VÀ LỰA CHỌN MÔ HÌNH ---
USD_TO_VND_RATE = 25500
MODEL_PRICING = {
    "gemini-1.5-pro-latest": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash-latest": {"input": 0.35, "output": 1.05}
}

# --- KHỞI TẠO AI VÀ DATABASE ---

@st.cache_resource
def setup_database():
    """Tải và giải nén CSDL từ URL nếu chưa có."""
    if not os.path.exists(DB_PATH):
        st.info(f"Không tìm thấy CSDL tại '{DB_PATH}'. Bắt đầu tải từ Google Drive...")
        try:
            r = requests.get(DB_ZIP_URL, stream=True)
            if r.status_code == 200:
                with st.spinner("Đang tải file chroma_db.zip..."):
                    zip_file = BytesIO(r.content)
                with st.spinner("Đang giải nén CSDL..."):
                    with zipfile.ZipFile(zip_file, 'r') as z:
                        z.extractall('.')
                st.success("Tải và giải nén CSDL thành công!")
                # Chờ một chút để hệ thống file ổn định
                time.sleep(2)
            else:
                st.error(f"Lỗi tải file: Status code {r.status_code}. Hãy kiểm tra lại đường dẫn URL.")
                return None, None
        except Exception as e:
            st.error(f"Lỗi trong quá trình tải hoặc giải nén: {e}")
            return None, None
    
    try:
        with st.spinner("Đang kết nối tới cơ sở dữ liệu vector..."):
            client = chromadb.PersistentClient(path=DB_PATH)
            collection = client.get_collection(name=COLLECTION_NAME)
        st.success("Kết nối CSDL thành công!")
        return client, collection
    except Exception as e:
        st.error(f"Lỗi kết nối tới ChromaDB tại '{DB_PATH}': {e}")
        st.error("Hãy đảm bảo bạn đã chạy các script xử lý và lập chỉ mục dữ liệu trước đó.")
        return None, None

def configure_ai():
    """Cấu hình API key cho Google AI."""
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Lỗi cấu hình Google Generative AI: {e}")

# --- CÁC HÀM XỬ LÝ LÕI ---

def get_relevant_context(query, collection, n_results=5):
    """Tìm các đoạn văn bản liên quan nhất trong CSDL."""
    if collection is None:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0] if results['documents'] else []

def get_ai_response(query, model_name, collection, system_instruction):
    """Tạo câu trả lời từ AI dựa trên câu hỏi và ngữ cảnh."""
    relevant_docs = get_relevant_context(query, collection)
    
    context_str = "\n---\n".join(relevant_docs)
    
    prompt = f"""{system_instruction}

Dựa vào các thông tin, quy định, và kiến thức được cung cấp dưới đây để trả lời câu hỏi của người dùng một cách chính xác và chi tiết.

---
{context_str}
---

Câu hỏi của người dùng: {query}
"""
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Lấy thông tin sử dụng token
        usage_info = None
        if response.usage_metadata:
            usage = response.usage_metadata
            pricing = MODEL_PRICING[model_name]
            input_cost_usd = (usage.prompt_token_count / 1_000_000) * pricing["input"]
            output_cost_usd = (usage.candidates_token_count / 1_000_000) * pricing["output"]
            total_cost_usd = input_cost_usd + output_cost_usd
            total_cost_vnd = total_cost_usd * USD_TO_VND_RATE
            
            usage_info = {
                "prompt_tokens": usage.prompt_token_count,
                "response_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count,
                "cost_usd": total_cost_usd,
                "cost_vnd": total_cost_vnd,
                "model": model_name
            }
        
        return response.text, usage_info

    except Exception as e:
        return f"Đã có lỗi xảy ra khi gọi API của Google: {e}", None

# --- GIAO DIỆN NGƯỜI DÙNG (UI) ---

# Khởi tạo các thành phần
configure_ai()
client, collection = setup_database()

# Tiêu đề ứng dụng
st.title(" Pathfinder - Trợ lý AI Tuấn 123 🤖")
st.caption("Trợ lý được xây dựng dựa trên kho tri thức nội bộ của công ty.")

# Khởi tạo session state để lưu lịch sử chat và chi phí
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_session_cost_vnd" not in st.session_state:
    st.session_state.total_session_cost_vnd = 0.0

# Cấu hình sidebar
with st.sidebar:
    st.header("Cài đặt")
    
    model_name_map = {
        "Flash (Nhanh & Rẻ)": "gemini-1.5-flash-latest",
        "Pro (Mạnh hơn)": "gemini-1.5-pro-latest"
    }
    selected_model_display = st.selectbox(
        "Chọn mô hình AI:",
        options=list(model_name_map.keys())
    )
    selected_model_name = model_name_map[selected_model_display]

    st.subheader("Vai trò của AI")
    system_instruction = st.text_area(
        "Bạn muốn AI đóng vai gì?",
        "Bạn là một Trợ lý AI am hiểu sâu sắc về các quy trình, quy định và văn hóa của công ty bất động sản Tuấn 123. Nhiệm vụ của bạn là cung cấp câu trả lời chính xác, chi tiết và chuyên nghiệp cho các nhân viên dựa trên kho tri thức được cung cấp.",
        height=200
    )

    if st.button("Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.session_state.total_session_cost_vnd = 0.0
        st.rerun()

    st.divider()
    st.markdown(f"**Tổng chi phí phiên này:**")
    st.markdown(f"### {st.session_state.total_session_cost_vnd:,.0f} VNĐ")


# Giao diện chat chính
if collection is not None:
    # Hiển thị các tin nhắn đã có
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Nhập câu hỏi của bạn về quy trình, nghiệp vụ..."):
        # Thêm câu hỏi của người dùng vào lịch sử và hiển thị
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Lấy và hiển thị câu trả lời của AI
        with st.chat_message("assistant"):
            with st.spinner(f"Pathfinder ({selected_model_display}) đang suy nghĩ..."):
                response_text, usage_info = get_ai_response(prompt, selected_model_name, collection, system_instruction)
                
                # Tạo response đầy đủ bao gồm cả chi tiết sử dụng API
                full_response_to_display = response_text
                if usage_info:
                    st.session_state.total_session_cost_vnd += usage_info['cost_vnd']
                    usage_html = f"""
                    <br>
                    <details style="font-size: 0.8em; color: grey;">
                        <summary>Chi tiết sử dụng API</summary>
                        <p style="margin: 0; padding-left: 1em;">- Model: {usage_info['model']}<br>
                           - Chi phí: {usage_info['cost_vnd']:,.0f} VNĐ<br>
                           - Tokens: {usage_info['total_tokens']}</p>
                    </details>
                    """
                    full_response_to_display += usage_html

                st.markdown(full_response_to_display, unsafe_allow_html=True)
        
        # Lưu câu trả lời (bao gồm cả chi tiết) vào session state
        st.session_state.messages.append({"role": "assistant", "content": full_response_to_display})
        st.rerun()
else:
    st.warning("CSDL không khả dụng. Vui lòng kiểm tra lại cấu hình và đảm bảo đã chạy các bước chuẩn bị dữ liệu.")
