import json
import os
import asyncio
from langchain_core.documents import Document
from typing import List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- Cấu hình ---
JSON_FILE_PATH = r"/data/plant.json"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_storage")
EMBED_MODEL = os.getenv("EMBED_MODEL", "AITeamVN/Vietnamese_Embedding")


def load_documents_from_json(json_path: str) -> List[Document]:
    """Đọc file JSON và chuyển đổi thành danh sách Document."""
    print(f"Đang tải dữ liệu từ: {json_path}")
    if not os.path.exists(json_path):
        print(f"Lỗi: Không tìm thấy file JSON tại: {json_path}")
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file JSON: {e}")
        return []

    docs_json = []
    for item in data.get("danh_sach_benh", []):
        text = (
            f"Tên bệnh: {item.get('ten_benh', '')}\n"
            f"Tên khoa học: {item.get('ten_khoa_hoc', '')}\n"
            f"Cây chủ: {item.get('cay_chu', '')}\n"
            f"Nguyên nhân: {item.get('nguyen_nhan', '')}\n"
            f"Triệu chứng: {item.get('trieu_chung', '')}\n"
            f"Phòng trừ: {item.get('phong_tru', '')}"
        )
        metadata = {
            "id": item.get("id", "unknown"),
            "source": json_path,
            "ten_benh": item.get('ten_benh', ''),
            "cay_chu": item.get('cay_chu', '')
        }
        doc = Document(page_content=text, metadata=metadata)
        docs_json.append(doc)
    print(f"Đã tải và xử lý thành công {len(docs_json)} tài liệu từ JSON.")
    return docs_json


def build_vector_store():
    """Hàm chính để tải, chia nhỏ và lưu trữ tài liệu vào Vector Store."""
    all_documents = load_documents_from_json(JSON_FILE_PATH)

    if not all_documents:
        print("Không có tài liệu nào để xử lý. Dừng lại.")
        return

    # 2. CHIA NHỎ
    print(f"\nĐang chia {len(all_documents)} tài liệu...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Chia mỗi đoạn 1000 ký tự
        chunk_overlap=200,  # Gối lên nhau 200 ký tự để giữ ngữ cảnh
        length_function=len
    )
    all_splits = text_splitter.split_documents(all_documents)
    print(f"Đã chia thành {len(all_splits)} đoạn (chunks).")

    # 3. NHÚNG
    print(f"Đang khởi tạo mô hình embedding (model: {EMBED_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},  # Dùng CPU
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Khởi tạo embedding model thành công.")

    # 4. LƯU TRỮ
    print(f"Đang tạo và lưu trữ Vector Store tại: {CHROMA_DB_PATH}")
    print("(Việc này có thể mất vài phút tùy thuộc vào số lượng tài liệu...)")
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_metadata={"hnsw:space": "cosine"}  # Chỉ định dùng Cosine
    )

    print("\n--- HOÀN THÀNH ---")
    print(f"Vector Store đã được tạo và lưu vĩnh viễn tại: {CHROMA_DB_PATH}")
    print(f"Tổng số vector đã được lưu: {vector_store._collection.count()}")


# --- Chạy hàm build ---
if __name__ == "__main__":
    build_vector_store()