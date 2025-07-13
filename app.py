# file: app.py
# Đã xóa các dòng __import__('pysqlite3') không cần thiết

import streamlit as st
import chromadb
import google.generativeai as genai

# --- PHẦN CẤU HÌNH ---
# API Key của bạn từ Google Cloud
GOOGLE_API_KEY = 'AIzaSyBOAgpJI1voNNxeOC6sS7y01EJRXWSK0YU' # !!! THAY API KEY CỦA BẠN VÀO ĐÂY !!!
# Đường dẫn tới cơ sở dữ liệu vector
DB_PATH = 'yhct_chroma_db'
# Tên của bộ sưu tập trong database
COLLECTION_NAME = 'yhct_collection'
# Tên mô hình bạn muốn sử dụng
MODEL_NAME = 'gemini-1.5-pro-latest'

# --- BẢNG GIÁ (Cập nhật tháng 7/2025 - Vui lòng kiểm tra lại giá trên trang của Google) ---
# Giá cho mỗi 1 triệu token
PRICE_PER_MILLION_INPUT_TOKENS = 3.50  # $3.50
PRICE_PER_MILLION_OUTPUT_TOKENS = 10.50 # $10.50

# --- KHỞI TẠO AI VÀ DATABASE ---
@st.cache_resource
def load_models_and_db():
    """Tải model AI và kết nối tới database chỉ một lần duy nhất."""
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm_model = genai.GenerativeModel(MODEL_NAME)
        
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        
        return llm_model, collection
    except Exception as e:
        st.error(f"Lỗi khởi tạo: {e}")
        return None, None

# --- HÀM LOGIC XỬ LÝ CÂU HỎI ---
def get_ai_response(question, model, collection):
    """Lấy câu trả lời từ AI và thông tin sử dụng API."""
    # 1. Tìm kiếm trong DB
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    
    retrieved_docs = results['documents'][0]
    retrieved_sources = [meta['source'] for meta in results['metadatas'][0]]

    # 2. Tạo prompt
    context = "\n\n---\n\n".join(retrieved_docs)
    prompt = f"""Dựa vào các thông tin tham khảo được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng.
    
    Thông tin tham khảo:
    {context}
    
    Câu hỏi: {question}
    """
    
    # 3. Gọi Gemini
    response = model.generate_content(prompt)
    
    # 4. Trích xuất thông tin sử dụng từ phản hồi của API
    usage_info = None
    try:
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count
        response_tokens = usage.candidates_token_count
        total_tokens = usage.total_token_count
        
        # Tính toán chi phí
        input_cost = (prompt_tokens / 1_000_000) * PRICE_PER_MILLION_INPUT_TOKENS
        output_cost = (response_tokens / 1_000_000) * PRICE_PER_MILLION_OUTPUT_TOKENS
        total_cost = input_cost + output_cost
        
        usage_info = {
            "model": MODEL_NAME,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "cost_usd": total_cost
        }
    except Exception as e:
        print(f"Không thể lấy thông tin sử dụng: {e}")

    return response.text, set(retrieved_sources), usage_info

# --- GIAO DIỆN NGƯỜI DÙNG STREAMLIT ---
st.set_page_config(page_title="Trợ lý Y học Cổ truyền", page_icon="🌿")
st.title("🌿 Trợ lý Y học Cổ truyền")
st.write("Đặt câu hỏi để tra cứu kiến thức từ kho dữ liệu y học cổ truyền.")

llm_model, collection = load_models_and_db()

if llm_model and collection:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ví dụ: Bệnh Thái Dương là gì?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI đang phân tích và tổng hợp..."):
                response_text, sources, usage_info = get_ai_response(prompt, llm_model, collection)
                
                # Tạo nội dung hiển thị chính
                source_markdown = "\n\n---\n**Nguồn tham khảo:**\n" + "\n".join([f"- `{s}`" for s in sources])
                full_response = response_text + source_markdown
                st.markdown(full_response)
                
                # Hiển thị thông tin sử dụng API trong một expander
                if usage_info:
                    with st.expander("Xem chi tiết sử dụng API"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Tokens Đầu vào", usage_info['prompt_tokens'])
                        col2.metric("Tokens Đầu ra", usage_info['response_tokens'])
                        col3.metric("Tổng Tokens", usage_info['total_tokens'])
                        col4.metric("Chi phí (USD)", f"${usage_info['cost_usd']:.6f}")
                        st.caption(f"Mô hình sử dụng: `{usage_info['model']}`")

        # Lưu lại toàn bộ nội dung đã hiển thị vào lịch sử chat
        if usage_info:
            usage_html = f"""
            <details>
                <summary>Xem chi tiết sử dụng API</summary>
                <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <p><b>Mô hình:</b> {usage_info['model']}</p>
                    <p><b>Tokens Đầu vào:</b> {usage_info['prompt_tokens']}</p>
                    <p><b>Tokens Đầu ra:</b> {usage_info['response_tokens']}</p>
                    <p><b>Tổng Tokens:</b> {usage_info['total_tokens']}</p>
                    <p><b>Chi phí (USD):</b> ${usage_info['cost_usd']:.6f}</p>
                </div>
            </details>
            """
            full_response += usage_html

        st.session_state.messages.append({"role": "assistant", "content": full_response})