import os
import shutil
import chromadb
import google.generativeai as genai
import re
import math

# --- PHẦN CẤU HÌNH ---
GOOGLE_API_KEY = 'AIzaSyBOAgpJI1voNNxeOC6sS7y01EJRXWSK0YU'  # !!! THAY API KEY CỦA BẠN VÀO ĐÂY !!!
DATA_DIR = 'data_output'
# --- ĐÃ ĐỒNG BỘ TÊN ---
DB_PATH = 'chroma_db'
COLLECTION_NAME = 'tuan123_collection'
# -------------------------

# --- CẤU HÌNH CHI PHÍ ---
EMBEDDING_MODEL = 'models/text-embedding-004'
PRICE_PER_MILLION_EMBEDDING_TOKENS = 0.20
USD_TO_VND_RATE = 25500

# Cấu hình cho việc chia nhỏ văn bản
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ... (Phần còn lại của file giữ nguyên) ...
# (Bạn không cần thay đổi phần còn lại, chỉ cần đảm bảo phần cấu hình ở trên là chính xác)

def split_text_into_chunks(text, chunk_size, chunk_overlap):
    """Chia một đoạn văn bản dài thành các đoạn nhỏ hơn."""
    words = re.split(r'(\s+)', text)
    chunks = []
    current_chunk_words = []
    current_length = 0

    for i in range(0, len(words), 2):
        word = words[i]
        space = words[i+1] if i+1 < len(words) else ""
        current_chunk_words.append(word + space)
        current_length += 1
        
        if current_length >= chunk_size:
            chunks.append("".join(current_chunk_words))
            overlap_start_index = max(0, len(current_chunk_words) - chunk_overlap * 2)
            current_chunk_words = current_chunk_words[overlap_start_index:]
            current_length = len(current_chunk_words)
            
    if current_chunk_words:
        chunks.append("".join(current_chunk_words))
        
    return chunks

def main():
    """Hàm chính điều phối việc lập chỉ mục dữ liệu vào ChromaDB."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == 'YOUR_API_KEY':
        print("Lỗi: Vui lòng thay YOUR_API_KEY bằng API Key của bạn.")
        return
    
    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Thư mục '{DATA_DIR}' không tồn tại. Vui lòng chạy các script trước đó.")
        return

    print("\nĐang quét và phân tích các file để ước tính chi phí lập chỉ mục...")
    total_tokens = 0
    all_content = ""
    for folder_name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        all_content += f.read()
    
    total_tokens = math.ceil(len(all_content.split()) / 0.75)
    estimated_cost_usd = (total_tokens / 1_000_000) * PRICE_PER_MILLION_EMBEDDING_TOKENS
    estimated_cost_vnd = estimated_cost_usd * USD_TO_VND_RATE

    print("\n--- BẢNG KÊ CHI PHÍ ƯỚC TÍNH (XÂY DỰNG BỘ NÃO AI) ---")
    print(f"Tổng số token cần xử lý ước tính: {total_tokens:,}")
    print(f"Mô hình Embedding sẽ sử dụng: {EMBEDDING_MODEL}")
    print(f"Chi phí ước tính: ${estimated_cost_usd:.4f} USD (~{estimated_cost_vnd:,.0f} VND)")
    print("-" * 60)

    proceed = input("Bạn có muốn tiếp tục không? (y/n): ").lower()
    if proceed != 'y':
        print("Đã hủy quá trình.")
        return

    genai.configure(api_key=GOOGLE_API_KEY)

    if os.path.exists(DB_PATH):
        print(f"Phát hiện database cũ tại '{DB_PATH}'. Sẽ xóa để tạo mới.")
        shutil.rmtree(DB_PATH)
        
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    documents = []
    metadatas = []
    ids = []
    doc_id = 1

    print("\nBắt đầu quá trình lập chỉ mục (indexing)... �")
    
    for folder_name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(folder_path, filename)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if content:
                        print(f"  - Đang chia nhỏ file: {folder_name}/{filename}")
                        chunks = split_text_into_chunks(content, 500, 50)
                        
                        for i, chunk in enumerate(chunks):
                            documents.append(chunk)
                            metadatas.append({'source': f"{folder_name}/{filename}", 'chunk': i})
                            ids.append(f"doc_{doc_id}_{i}")
                        
                        doc_id += 1

    if documents:
        print(f"\nĐang thêm {len(documents)} đoạn văn bản vào database...")
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print("--- LẬP CHỈ MỤC HOÀN TẤT ---")
        print(f"Bộ não AI của bạn đã được tạo tại thư mục: '{DB_PATH}'")
    else:
        print("Không tìm thấy tài liệu nào để lập chỉ mục.")

if __name__ == "__main__":
    main()
�
