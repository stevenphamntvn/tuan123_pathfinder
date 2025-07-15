# file: app.py
# v2.4: Sửa lỗi chromadb.errors.InvalidArgumentError

# --- PHẦN SỬA LỖI QUAN TRỌNG CHO STREAMLIT CLOUD ---
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

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    GOOGLE_API_KEY = 'AIzaSyBOAgpJI1voNNxeOC6sS7y01EJRXWSK0YU'

# Cấu hình tải CSDL từ Dropbox/GitHub
DB_ZIP_URL = "https://www.dropbox.com/scl/fi/kzrz3y76uvhwlgdg9qhl7/chroma_db.zip?rlkey=h4bfzcx1opsts5vf34a4k23xm&st=ru7367m1&dl=1" # !!! DÁN LINK TẢI TRỰC TIẾP CỦA BẠN VÀO ĐÂY

DB_PATH = "chroma_db"
COLLECTION_NAME = "collection"
EMBEDDING_MODEL = "models/text-embedding-004" # Phải giống với file index_data

# Bảng giá
USD_TO_VND_RATE = 25500
MODEL_PRICING = {
    "gemini-1.5-pro-latest": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash-latest": {"input": 0.35, "output": 1.05}
}

# --- KHỞI TẠO AI VÀ DATABASE ---

@st.cache_resource
def setup_database():
    if not os.path.exists(DB_PATH):
        st.info(f"Bắt đầu tải CSDL từ link đã cung cấp...")
        try:
            response = requests.get(DB_ZIP_URL, stream=True)
            if response.status_code == 200:
                with st.spinner("Đang tải file CSDL..."):
                    zip_file = BytesIO(response.content)
                with st.spinner("Đang giải nén CSDL..."):
                    with zipfile.ZipFile(zip_file, 'r') as z:
                        z.extractall('.')
                st.success("Tải và giải nén thành công!")
                time.sleep(2)
            else:
                st.error(f"Lỗi tải file: Status code {response.status_code}.")
                return None, None
        except Exception as e:
            st.error(f"Lỗi tải hoặc giải nén: {e}")
            return None, None
    
    try:
        with st.spinner("Đang kết nối tới cơ sở dữ liệu vector..."):
            client = chromadb.PersistentClient(path=DB_PATH)
            collection = client.get_collection(name=COLLECTION_NAME)
        st.success("Kết nối CSDL thành công!")
        return client, collection
    except Exception as e:
        st.error(f"Lỗi kết nối tới ChromaDB: {e}")
        return None, None

def configure_ai():
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Lỗi cấu hình Google AI: {e}")

# --- CÁC HÀM XỬ LÝ LÕI ---

# --- HÀM ĐÃ ĐƯỢC NÂNG CẤP ---
def get_relevant_context(query, collection, n_results=5):
    """
    Tạo embedding cho câu hỏi và dùng nó để truy vấn CSDL.
    """
    if collection is None:
        return []
    
    # 1. Mã hóa câu hỏi của người dùng thành vector
    query_embedding = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY" # Task type dành cho việc truy vấn
    )["embedding"]

    # 2. Dùng vector câu hỏi để tìm kiếm trong CSDL
    results = collection.query(
        query_embeddings=[query_embedding], # Sử dụng query_embeddings thay vì query_texts
        n_results=n_results
    )
    return results['documents'][0] if results and results['documents'] else []

def get_ai_response(query, model_name, collection, system_instruction):
    """Tạo câu trả lời từ AI."""
    relevant_docs = get_relevant_context(query, collection)
    if not relevant_docs:
        return "Xin lỗi, tôi không tìm thấy thông tin nào liên quan đến câu hỏi của bạn trong cơ sở tri thức.", None
        
    context_str = "\n---\n".join(relevant_docs)
    prompt = f"""{system_instruction}\n\nDựa vào các thông tin, quy định, và kiến thức được cung cấp dưới đây để trả lời câu hỏi của người dùng.\n\n---NGỮ CẢNH---\n{context_str}\n---HẾT NGỮ CẢNH---\n\nCâu hỏi: {query}"""
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        usage_info = None
        if response.usage_metadata:
            usage = response.usage_metadata
            pricing = MODEL_PRICING[model_name]
            total_cost_usd = ((usage.prompt_token_count / 1_000_000) * pricing["input"]) + ((usage.candidates_token_count / 1_000_000) * pricing["output"])
            usage_info = {
                "total_tokens": usage.total_token_count,
                "cost_vnd": total_cost_usd * USD_TO_VND_RATE,
                "model": model_name
            }
        return response.text, usage_info
    except Exception as e:
        return f"Lỗi khi gọi API Google: {e}", None

# --- GIAO DIỆN NGƯỜI DÙNG (UI) ---

configure_ai()
client, collection = setup_database()

st.title(" Pathfinder - Trợ lý AI Tuấn 123 🤖")
st.caption("Trợ lý được xây dựng dựa trên kho tri thức nội bộ của công ty.")

if "messages" not in st.session_state: st.session_state.messages = []
if "total_session_cost_vnd" not in st.session_state: st.session_state.total_session_cost_vnd = 0.0

with st.sidebar:
    st.header("Cài đặt")
    model_name_map = {"Flash (Nhanh & Rẻ)": "gemini-1.5-flash-latest", "Pro (Mạnh hơn)": "gemini-1.5-pro-latest"}
    selected_model_display = st.selectbox("Chọn mô hình AI:", options=list(model_name_map.keys()))
    selected_model_name = model_name_map[selected_model_display]
    system_instruction = st.text_area("Vai trò của AI:", "Bạn là một Trợ lý AI am hiểu sâu sắc về các quy trình, quy định và văn hóa của công ty bất động sản Tuấn 123. Nhiệm vụ của bạn là cung cấp câu trả lời chính xác, chi tiết và chuyên nghiệp cho các nhân viên dựa trên kho tri thức được cung cấp.", height=200)
    if st.button("Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.session_state.total_session_cost_vnd = 0.0
        st.rerun()
    st.divider()
    st.markdown("**Tổng chi phí phiên này:**")
    st.markdown(f"### {st.session_state.total_session_cost_vnd:,.0f} VNĐ")

if collection is not None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Nhập câu hỏi của bạn về quy trình, nghiệp vụ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Pathfinder ({selected_model_display}) đang suy nghĩ..."):
                response_text, usage_info = get_ai_response(prompt, selected_model_name, collection, system_instruction)
                full_response_to_display = response_text
                if usage_info:
                    st.session_state.total_session_cost_vnd += usage_info['cost_vnd']
                    usage_html = f"""<br><details style="font-size: 0.8em; color: grey;"><summary>Chi tiết</summary><p style="margin: 0; padding-left: 1em;">- Model: {usage_info['model']}<br>- Chi phí: {usage_info['cost_vnd']:,.0f} VNĐ<br>- Tokens: {usage_info['total_tokens']}</p></details>"""
                    full_response_to_display += usage_html
                st.markdown(full_response_to_display, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response_to_display})
        st.rerun()
else:
    st.warning("CSDL không khả dụng. Vui lòng kiểm tra lại cấu hình.")
